# DL_Team8_kzHateSpeechDetection

Comparative study of hate speech detection models for the Kazakh language:
- TF–IDF + MLP baseline (fully reproducible code under `src/`)
- BiLSTM and Transformer-based models (implemented and analyzed in notebooks)

---
Although the repository contains a fully reproducible baseline pipeline implemented under src/ (TF–IDF + MLP, with one-command training and evaluation), the main experimental work of the project was conducted in Jupyter notebooks.
## Repository structure

- `src/`
  - `data/`: TF–IDF datamodule (`datamodule_tfidf.py`) and raw CSVs used to build the final dataset
  - `models/`: baseline MLP model (`baseline_traditional.py`)
  - `training/`: training entrypoint (`train_baseline.py`)
  - `evaluation/`: evaluation entrypoint (`eval_baseline.py`)
  - `utils/`: metrics (`metrics.py`) and seed utilities (`seed.py`)
- `configs/`: configuration file for the baseline (`baseline.yaml`)
- `notebooks/`: EDA and experimental models (CNN, BiLSTM, Transformer)
- `docs/`: annotation guidelines
- `reports/`: final project report
- `data/`: sample CSV used to illustrate the expected data format
- `results/`: checkpoints, logs, metrics and figures (created after running training)
- `scripts/`: one-command training and evaluation shell scripts
- `tests/`: minimal unit tests for key utilities

---

## Setup

```bash
git clone <YOUR_REPO_URL>
cd DL_Team8_kzHateSpeechDetection-main

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt
