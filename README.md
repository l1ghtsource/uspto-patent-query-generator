# USPTO - Explainable AI for Patent Professionals (Kaggle Bronze Medal)

## Overview

> The goal of this competition is to generate Boolean search queries that effectively characterize collections of patent documents. You are challenged to create a query generation model that, given an input set of related patents, outputs a Boolean query that returns the same set of patent documents. Capable solutions to this challenge will enable patent professionals to use AI-powered search capabilities with increased confidence by providing them the ability to interpret the results in a familiar language and syntax. Your work will support the effective and responsible adoption of AI technology in the IP ecosystem.

## Solution

- select top-k words with high TFIDF values as candidates
- select the word with the maximum AP@50 using the annealing method (the genetic algorithm was also tested, but the results were worse)
- combine the selected words with “OR” to form a query

## Certificate

![certificate](certificate.png)
