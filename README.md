# Make Data Count - LLaMA Pipeline

This repository provides a lightweight inference pipeline for the [Kaggle Make Data Count competition](https://www.kaggle.com/competitions/make-data-count-finding-data-references). It supports multiple open‑source LLMs and can run end‑to‑end inside a Kaggle notebook or from the command line.

## Folder Structure
```
make-data-count-llama/
├── data/               # context units, predictions and submissions
├── models/             # downloaded model weights
├── notebooks/          # example notebooks
├── scripts/            # CLI utilities
├── utils/              # helper modules
└── tests/              # unit tests
```

## Setup
Install dependencies in your Kaggle notebook or local environment:
```bash
pip install -r requirements.txt
```

## CLI Usage
Run the full pipeline with a single command:
```bash
python scripts/main_pipeline.py \
  --model deepseek \
  --model-path /kaggle/input/deepseek-coder-1.3b \
  --input data/context/context.jsonl \
  --output data/submission/submission.csv \
  --reask \
  --save-errors
```

## Supported Models
- `llama3`
- `qwen`
- `deepseek`
- `mixtral`
- `gemma`

## Kaggle Notebook
See [`notebooks/main_pipeline.ipynb`](notebooks/main_pipeline.ipynb) for a minimal example of running the CLI inside a Kaggle notebook and previewing the resulting predictions.
