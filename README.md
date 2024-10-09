# Radiological Society of North America - Lumbar Spine Degenerative Classification (Kaggle Bronze Medal)

## Description

> Low back pain is the leading cause of disability worldwide, according to the World Health Organization, affecting 619 million people in 2020. Most people experience low back pain at some point in their lives, with the frequency increasing with age. Pain and restricted mobility are often symptoms of spondylosis, a set of degenerative spine conditions including degeneration of intervertebral discs and subsequent narrowing of the spinal canal (spinal stenosis), subarticular recesses, or neural foramen with associated compression or irritations of the nerves in the low back. Magnetic resonance imaging (MRI) provides a detailed view of the lumbar spine vertebra, discs and nerves, enabling radiologists to assess the presence and severity of these conditions. Proper diagnosis and grading of these conditions help guide treatment and potential surgery to help alleviate back pain and improve overall health and quality of life for patients. RSNA has teamed with the American Society of Neuroradiology (ASNR) to conduct this competition exploring whether artificial intelligence can be used to aid in the detection and classification of degenerative spine conditions using lumbar spine MR images. The challenge will focus on the classification of five lumbar spine degenerative conditions: Left Neural Foraminal Narrowing, Right Neural Foraminal Narrowing, Left Subarticular Stenosis, Right Subarticular Stenosis, and Spinal Canal Stenosis. For each imaging study in the dataset, we’ve provided severity scores (Normal/Mild, Moderate, or Severe) for each of the five conditions across the intervertebral disc levels L1/L2, L2/L3, L3/L4, L4/L5, and L5/S1. 

## Solution

- 3x YOLOv10x 2D-detector as 1st classifier
  - ([NFN train script](scripts/lsdc-train-yolo-nfn.py))
  - ([SCS train script](scripts/lsdc-train-yolo-scs.py))
  - ([SS train script](scripts/lsdc-train-yolo-ss.py))
- 1x 3D-ViT model as 2nd classifier
  - ([train script](scripts/lsdc-train-vit.py))
- ensembling and TTA on inference
  - ([inference script](scripts/lsdc-final-inference.py))

## Certificate

![certificate](certificate.png)
