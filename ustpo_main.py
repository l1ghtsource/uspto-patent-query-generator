from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import datetime
import math
import pickle
import random
import signal
import sys
import time
import whoosh_utils
from pathlib import Path
import polars as pl
from tqdm import tqdm
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Any


comp_data_dir = Path("/kaggle/input/uspto-explainable-ai")

meta = pl.scan_parquet(comp_data_dir / "patent_metadata.parquet")
meta = (
    meta.with_columns(
        pl.col("publication_date").dt.year().alias("year"),
        pl.col("publication_date").dt.month().alias("month"),
    )
    .filter(pl.col("publication_date") >= pl.date(1975, 1, 1))
    .rename({"cpc_codes": "cpc"})
    .collect()
)
test_nn = pl.scan_csv(comp_data_dir / "test.csv")

all_pub = test_nn.melt().collect().get_column("value").unique()
meta = meta.filter(pl.col("publication_number").is_in(all_pub))

patents = []
n_unique = meta.select(["year", "month"]).n_unique()
for (year, month), _ in tqdm(meta.group_by(["year", "month"]), total=n_unique):
    patent_path = comp_data_dir / f"patent_data/{year}_{month}.parquet"
    patent = pl.scan_parquet(patent_path).select(pl.exclude(["claims", "description"]))
    patents.append(patent)
patent: pl.LazyFrame = pl.concat(patents)
patent = patent.with_columns(
    pl.lit("").alias("claims"),
    pl.lit("").alias("description"),
)
meta_with_text = (
    meta.lazy()
    .join(patent, on="publication_number", how="left")
    .collect(streaming=True)
)
meta_with_text.write_parquet("meta_with_text.parquet")

documents = meta_with_text.to_dicts()
Path("test_index").mkdir(parents=True, exist_ok=True)
whoosh_utils.create_index("test_index", documents)


def round_figures(x, n):
    return round(x, int(n - math.ceil(math.log10(abs(x)))))


def time_string(seconds):
    s = int(round(seconds))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return "%4i:%02i:%02i" % (h, m, s)


class Annealer(object):
    __metaclass__ = abc.ABCMeta

    Tmax = 25000.0
    Tmin = 2.5
    steps = 50000
    max_time = 8
    updates = 100
    copy_strategy = "deepcopy"
    user_exit = False
    save_state_on_exit = False

    best_state = None
    best_energy = None
    start = None

    def __init__(self, initial_state=None, load_state=None):
        if initial_state is not None:
            self.state = self.copy_state(initial_state)
        elif load_state:
            self.load_state(load_state)
        else:
            raise ValueError(
                "No valid values supplied for neither \
            initial_state nor load_state"
            )

        signal.signal(signal.SIGINT, self.set_user_exit)

    def save_state(self, fname=None):
        if not fname:
            date = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
            fname = date + "_energy_" + str(self.energy()) + ".state"
        with open(fname, "wb") as fh:
            pickle.dump(self.state, fh)

    def load_state(self, fname=None):
        with open(fname, "rb") as fh:
            self.state = pickle.load(fh)

    @abc.abstractmethod
    def move(self):
        pass

    @abc.abstractmethod
    def energy(self):
        pass

    def set_user_exit(self, signum, frame):
        self.user_exit = True

    def set_schedule(self, schedule):
        self.Tmax = schedule["tmax"]
        self.Tmin = schedule["tmin"]
        self.steps = int(schedule["steps"])
        self.updates = int(schedule["updates"])

    def copy_state(self, state):
        if self.copy_strategy == "deepcopy":
            return copy.deepcopy(state)
        elif self.copy_strategy == "slice":
            return state[:]
        elif self.copy_strategy == "method":
            return state.copy()
        else:
            raise RuntimeError(
                "No implementation found for "
                + 'the self.copy_strategy "%s"' % self.copy_strategy
            )

    def update(self, *args, **kwargs):
        self.default_update(*args, **kwargs)

    def default_update(self, step, T, E, acceptance, improvement):
        elapsed = time.time() - self.start
        if step == 0:
            print(
                "\n Temperature        Energy    Accept   Improve     Elapsed   Remaining",
                file=sys.stderr,
            )
            print(
                "\r{Temp:12.5f}  {Energy:12.2f}                      {Elapsed:s}            ".format(
                    Temp=T, Energy=E, Elapsed=time_string(elapsed)
                ),
                file=sys.stderr,
                end="",
            )
            sys.stderr.flush()
        else:
            remain = (self.steps - step) * (elapsed / step)
            print(
                "\r{Temp:12.5f}  {Energy:12.2f}   {Accept:7.2%}   {Improve:7.2%}  {Elapsed:s}  {Remaining:s}".format(
                    Temp=T,
                    Energy=E,
                    Accept=acceptance,
                    Improve=improvement,
                    Elapsed=time_string(elapsed),
                    Remaining=time_string(remain),
                ),
                file=sys.stderr,
                end="",
            )
            sys.stderr.flush()

    def anneal(self):
        step = 0
        self.start = time.time()

        if self.Tmin <= 0.0:
            raise Exception(
                'Exponential cooling requires a minimum "\
                "temperature greater than zero.'
            )
        Tfactor = -math.log(self.Tmax / self.Tmin)

        T = self.Tmax
        E = self.energy()
        prevState = self.copy_state(self.state)
        prevEnergy = E
        self.best_state = self.copy_state(self.state)
        self.best_energy = E
        trials = accepts = improves = 0
        if self.updates > 0:
            updateWavelength = self.steps / self.updates
            self.update(step, T, E, None, None)

        while (
            (step < self.steps)
            and (not self.user_exit)
            and ((time.time() - self.start) <= self.max_time)
        ):
            step += 1
            T = self.Tmax * math.exp(Tfactor * step / self.steps)
            dE = self.move()
            if dE is None:
                E = self.energy()
                dE = E - prevEnergy
            else:
                E += dE
            trials += 1
            if dE > 0.0 and math.exp(-dE / T) < random.random():
                self.state = self.copy_state(prevState)
                E = prevEnergy
            else:
                accepts += 1
                if dE < 0.0:
                    improves += 1
                prevState = self.copy_state(self.state)
                prevEnergy = E
                if E < self.best_energy:
                    self.best_state = self.copy_state(self.state)
                    self.best_energy = E
            if self.updates > 1:
                if (step // updateWavelength) > ((step - 1) // updateWavelength):
                    self.update(step, T, E, accepts / trials, improves / trials)
                    trials = accepts = improves = 0

        self.state = self.copy_state(self.best_state)
        if self.save_state_on_exit:
            self.save_state()

        return self.best_state, self.best_energy


def select_top_k_columns(X: Any, k: int) -> tuple[Any, NDArray]:
    row_sums = X.sum(axis=0)
    top_k_indices = np.argsort(-row_sums.A1)[:k]
    X_top = X[:, top_k_indices]

    return X_top, top_k_indices


def ap50(preds: list[str], labels: list[str]) -> float:
    precisions = list()
    n_found = 0
    for e, i in enumerate(preds):
        if i in labels:
            n_found += 1
        precisions.append(n_found / (e + 1))
    return sum(precisions) / 50


@dataclass
class Word:
    category: str
    content: str

    def to_str(self):
        return f"{self.category}:{self.content}"


@dataclass
class State:
    words: list[Word]

    def __post_init__(self):
        self.use = np.random.binomial(1, 0.5, len(self.words))

    def to_query(self):
        words = [word.to_str() for word, use in zip(self.words, self.use) if use]

        return " OR ".join(words)

    def move_1(self):
        idx = np.random.choice(len(self.words))
        self.use[idx] = 1 - self.use[idx]
        return self


class USPTOProblem(Annealer):
    def __init__(
        self,
        qp: Any,
        searcher: Any,
        target: list[str],
        init_state: State,
        tmax: int = 30,
        tmin: int = 10,
        steps: int = 100,
        max_time: int = 8,
        copy_strategy: str = "deepcopy",
    ):
        super(USPTOProblem, self).__init__(init_state)
        self.qp = qp
        self.searcher = searcher
        self.target = target
        self.Tmax = tmax
        self.Tmin = tmin
        self.steps = steps
        self.max_time = max_time
        self.copy_strategy = copy_strategy

    def move(self):
        self.state.move_1()

    def energy(self):
        query = self.state.to_query()
        cand = whoosh_utils.execute_query(query, self.qp, self.searcher)
        ap50_score = ap50(cand, self.target)

        return -ap50_score


class GeneticAlgorithm:
    def __init__(
        self,
        qp,
        searcher,
        target,
        init_population,
        mutation_rate=0.1,
        max_generations=100,
        max_time=8,
    ):
        self.qp = qp
        self.searcher = searcher
        self.target = target
        self.population = init_population
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.max_time = max_time
        self.best_state = None
        self.best_fitness = None
        self.start_time = None

    def fitness(self, state):
        query = state.to_query()
        candidates = whoosh_utils.execute_query(query, self.qp, self.searcher)
        ap50_score = ap50(candidates, self.target)
        return -ap50_score

    def selection(self):
        selected = []
        for _ in range(len(self.population)):
            idx1, idx2 = random.randint(0, len(self.population) - 1), random.randint(
                0, len(self.population) - 1
            )
            if self.fitness(self.population[idx1]) > self.fitness(
                self.population[idx2]
            ):
                selected.append(self.population[idx1])
            else:
                selected.append(self.population[idx2])
        return selected

    def crossover(self, parent1, parent2):
        point = random.randint(1, len(parent1.words) - 1)
        child1_words = parent1.words[:point] + parent2.words[point:]
        child2_words = parent2.words[:point] + parent1.words[point:]
        return State(words=child1_words), State(words=child2_words)

    def mutation(self, state):
        mutated_words = []
        for word in state.words:
            if random.random() < self.mutation_rate:
                mutated_words.append(Word(category=word.category, content=word.content))
            else:
                mutated_words.append(word)
        return State(words=mutated_words)

    def evolve(self):
        self.start_time = time.time()
        generation = 0
        while (
            generation < self.max_generations
            and (time.time() - self.start_time) <= self.max_time
        ):
            next_population = []

            fitness_scores = [self.fitness(state) for state in self.population]
            best_idx = np.argmax(fitness_scores)
            best_state = self.population[best_idx]
            best_fitness = fitness_scores[best_idx]

            if self.best_fitness is None or best_fitness > self.best_fitness:
                self.best_state = best_state
                self.best_fitness = best_fitness

            selected = self.selection()

            for i in range(0, len(selected), 2):
                child1, child2 = self.crossover(selected[i], selected[i + 1])
                next_population.extend([child1, child2])

            next_population = [self.mutation(state) for state in next_population]

            self.population = next_population
            generation += 1

        return self.best_state, self.best_fitness


comp_data_dir = Path("/kaggle/input/uspto-explainable-ai")
tfidf_dir = Path("/kaggle/input/uspto-ti-cpc-tfidf")

test = pl.read_csv(comp_data_dir / "test.csv")
test_meta = pl.read_parquet("meta_with_text.parquet")

test_idx = whoosh_utils.load_index("./test_index")
searcher = whoosh_utils.get_searcher(test_idx)
qp = whoosh_utils.get_query_parser()


def identity(x: Any) -> Any:
    return x


with open(tfidf_dir / "tfidf.pkl", "rb") as f:
    ti_tfidf = pickle.load(f)
with open(tfidf_dir / "cpc_cv_tfidf.pkl", "rb") as f:
    cpc_cv_tfidf = pickle.load(f)

scores = []
results = []

for i in tqdm(range(len(test))):
    target = test[i].to_numpy().flatten()[1:].tolist()
    meta_i = test_meta.filter(pl.col("publication_number").is_in(target))

    if len(meta_i) == 0:
        results.append(
            {"publication_number": test[i, "publication_number"], "query": "ti:device"}
        )
        print("\t Append Dummy", i)
        continue

    ti_mat = ti_tfidf.transform(meta_i.get_column("title").fill_null(""))
    cpc_mat = cpc_cv_tfidf.transform(meta_i.get_column("cpc"))

    X_ti, idx = select_top_k_columns(ti_mat, k=10)
    X_cpc, cpc_idx = select_top_k_columns(cpc_mat, k=10)

    topk_words = ti_tfidf.get_feature_names_out()[idx].tolist()
    topk_cpc = cpc_cv_tfidf.get_feature_names_out()[cpc_idx]
    topk_words = [Word(category="ti", content=x) for x in topk_words]
    topk_cpc = [Word(category="cpc", content=x) for x in topk_cpc]
    words = topk_words + topk_cpc

    # initial_population = [State(words=words) for _ in range(20)]

    # ga = GeneticAlgorithm(
    #     qp,
    #     searcher,
    #     target,
    #     initial_population,
    #     mutation_rate=0.2,
    #     max_generations=100,
    #     max_time=5,
    # )
    # solution, score = ga.evolve()

    state = State(words=words)

    problem = USPTOProblem(
        qp, searcher, target, state, steps=1000, tmax=30, tmin=10, max_time=10
    )
    solution, score = problem.anneal()
    print(f"\t Problem Number {i} Score:", -score)
    scores.append(-score)

    results.append(
        {
            "publication_number": test[i, "publication_number"],
            "query": solution.to_query(),
        }
    )


print("Average Score:", sum(scores) / len(scores))

submission = pl.DataFrame(results)
submission.write_csv("submission.csv")
