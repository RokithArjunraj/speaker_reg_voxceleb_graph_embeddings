## Graph Embedding for Speaker Verification — Updated with Novel Enhancement

### Overview
This repository replicates He et al. (ICASSP 2024) and extends it with ScalarGateGNN,
a novel edge-featured graph neural network that modulates message passing using learned
acoustic edge gates (cosine similarity + L2 distance per edge).

---

## Pre-extracted Embeddings

- outputs/xvectors/          — TDNN x-vectors (512-d)
- outputs/ecapa_xvectors/    — ECAPA-TDNN embeddings (192-d)
- outputs/librispeech_tdnn/  — TDNN embeddings for LibriSpeech test-clean
- outputs/librispeech_ecapa/ — ECAPA embeddings for LibriSpeech test-clean

Audio files are not required to reproduce reported results on VoxCeleb1.
For LibriSpeech experiments, download test-clean (~350MB) from openslr.org/12
and run the extraction notebooks below.

---

## Execution Steps

### 1. Embedding Extraction (VoxCeleb1 — already done, skip if using pre-extracted)
- notebooks/01_extract_xvectors.ipynb          — TDNN x-vectors
- notebooks/01_extract_TDNN_embeddings.ipynb   — ECAPA embeddings

### 2. Embedding Extraction (LibriSpeech — run after downloading test-clean)
- notebooks/01_extract_librispeech_tdnn.ipynb  — TDNN on LibriSpeech
- notebooks/01_extract_librispeech_ecapa.ipynb — ECAPA on LibriSpeech

### 3. Baseline Cosine Scoring (VoxCeleb1)
- notebooks/02_Traditional_cosine_tdnn_xvector.ipynb
- notebooks/02_Traditional_cosine_ecapa.ipynb

### 4. Graph-based Refinement — Replication (VoxCeleb1)
- notebooks/04_GCN_KNN.ipynb         — GCN with kNN graph
- notebooks/04_GCN_Threshold.ipynb   — GCN with threshold graph
- notebooks/03_GAT_KNN.ipynb         — GAT with kNN graph

### 5. Novel Enhancement — ScalarGateGNN (VoxCeleb1, TDNN + ECAPA)
- notebooks/04_ScalarGateGNN.ipynb
  Runs cosine → GCN → ScalarGate-v1 → ScalarGate-v2 → ScalarGate-full
  Also includes ECAPA results and qualitative analysis (t-SNE + gate weights)

### 6. Cross-domain Validation (LibriSpeech)
- notebooks/05_LibriSpeech_eval.ipynb
  Runs all methods on LibriSpeech for both TDNN and ECAPA embeddings

---

## Key Results

| Method              | VoxCeleb1 TDNN EER% | VoxCeleb1 ECAPA EER% | LibriSpeech TDNN EER% |
|---------------------|---------------------|----------------------|-----------------------|
| Cosine              | 8.88                | 0.90                 | 7.08                  |
| GCN binary edges    | 4.62                | 0.62                 | 2.64                  |
| ScalarGate-v2 (ours)| 3.14                | 0.22                 | 1.72                  |

ScalarGate-v2 achieves 32% relative EER reduction over GCN on VoxCeleb1-TDNN.

---

## Repository Structure

data/
  voxceleb1/test/          — VoxCeleb1 test wav files + trial list
  librispeech/             — LibriSpeech test-clean (download separately)
outputs/
  xvectors/                — TDNN embeddings
  ecapa_xvectors/          — ECAPA embeddings
  librispeech_tdnn/        — LibriSpeech TDNN embeddings
  librispeech_ecapa/       — LibriSpeech ECAPA embeddings
  protocol/                — Trial labels
  tsne_before_after.png    — Qualitative t-SNE figure
  gate_weights.png         — Gate weight analysis figure
notebooks/
  01_*                     — Embedding extraction
  02_*                     — Cosine baselines
  03_*                     - Graph refinement using Attention 
  04_*                     — Graph refinement + ScalarGateGNN
  05_*                     — LibriSpeech cross-domain evaluation
