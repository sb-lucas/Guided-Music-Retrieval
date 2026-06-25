# Guided Music Retrieval

This is the official code for Guided Music Retrieval: Music Search with Multimodal Queries.

## Repository Contents

- `create_triplets/MTIQ/`: code to construct triplets for the MTIQ dataset
- `create_triplets/MTSQ/`: code to synthesize audio and construct triplets for the MTSQ dataset
- `train/`: training and evaluation scripts for all models considered in this work
- `feature_extractors/`: audio preprocessing utilities, including embedding extraction, tagging models, source separation, and fixed-window trimming (centered on the middle 10 seconds)
- `dimsim/`: code to evaluate different embedding models for music similarity

Dataset and pretrained model paths are not hardcoded and must be configured locally by the user.

## Environment

```bash
conda env create -f environment.yml
conda activate gmr
```

## Dataset

Both MTIQ and MTSQ datasets, along with instructions for download and usage, are available at:
https://huggingface.co/datasets/sb-lucas/Music-Text-Queries

## Note

This repository is released for research transparency and reproducibility. It is not actively maintained.