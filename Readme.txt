## Pre-extracted Embeddings

To simplify reproduction, we provide pre-extracted speaker embeddings:

- TDNN x-vectors (512-d). (Using 01_extract_TDNN_embeddings.ipynb)
- ECAPA-TDNN embeddings (192-d) ( Using 01_extract_xvectors.ipynb)

These were extarcted from voxceleb test dataset.
These are used directly for cosine scoring and graph neural network refinement.
Audio files are not required to reproduce the reported results.

Execution Steps: (Run following notebooks for the results)
1. Baseline cosine scoring
TDNN:

02_Traditional_cosine_tdnn_xvector.ipynb

ECAPA:

02_Traditional_cosine_ecapa.ipynb

2. Graph-based refinement

GCN with KNN:

04_GCN_KNN.ipynb

GCN with threshold graph:

04_GCN_Threshold.ipynb

GAT with KNN:

04_GAT_KNN.ipynb

