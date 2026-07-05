# Predicting Review Helpfulness on Amazon

A binary classification project that predicts whether an Amazon Electronics product review will be considered helpful by other customers, using NLP-derived features and metadata.

## Problem Statement

Most Amazon reviews receive few or no helpfulness votes, making it difficult to surface useful content. This project predicts helpfulness before votes accumulate using review text and metadata.

A review is labeled **helpful** if it received 3 or more helpful votes.

## Dataset

Amazon Reviews 2023 - Electronics category
Source: [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
Size: 200,000 records sampled from ~20M, split 70/15/15 (train/val/test)

## Models

### Classical Baseline - XGBoost
Trained on 13 handcrafted NLP and metadata features:
- Text features: review length, word count, avg word length, lexical diversity
- Readability: Flesch Reading Ease, Flesch-Kincaid Grade
- Sentiment: VADER (positive, negative, neutral, compound)
- Metadata: star rating, verified purchase, title length

### Transformer - DistilBERT
- Pretrained distilbert-base-uncased with frozen weights
- Custom classification head (768 -> 128 -> 2)
- Trained on 10,000 stratified samples due to compute constraints
- Class weighting applied to handle imbalance

## Results

| Model | F1 Macro (Test) | AUC-ROC (Test) |
|-------|----------------|----------------|
| XGBoost | 0.59 | 0.83 |
| DistilBERT | 0.55 | 0.82 |

XGBoost matches transformer performance while offering interpretability through feature importance. Review length and word count are the strongest predictors of helpfulness.

## Repository Structure

    src/
        pipeline.py          - Data loading, filtering, train/val/test split
        features.py          - NLP feature extraction
        model_classical.py   - XGBoost classifier
        model_transformer.py - DistilBERT classifier
    experiments/
        classical_results.json
        transformer_results.json
    data/                    - Generated CSVs (not tracked in git)
    README.md

## Setup

    conda activate tf_env
    pip install xgboost textstat vaderSentiment --break-system-packages

## Reproduce

    curl -L -O https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/Electronics.jsonl
    python src/pipeline.py
    python src/features.py
    python src/model_classical.py
    python src/model_transformer.py

## Running in Jupyter

If you prefer not to use the terminal, notebooks/nlp_helpfulness_walkthrough.ipynb runs the entire pipeline (data loading, feature engineering, both models) inside Jupyter:

    pip install jupyterlab --break-system-packages
    jupyter lab

Open the notebook and run cells top to bottom. Update the DATA_PATH variable at the top to point to your local Electronics.jsonl file.

## GPU vs CPU Training Time

Training the DistilBERT classification head on CPU takes roughly 2-4 hours for 3 epochs on a 10,000-record sample. The same run on a CUDA-enabled GPU takes approximately 5-10 minutes, and scaling to the full training set (~140,000 records) on GPU would take roughly 30-60 minutes.

## Dependencies

- Python 3.11
- pandas, numpy, scikit-learn
- xgboost
- transformers, torch
- textstat, vaderSentiment

## Citation

McAuley, J. et al. (2023). Amazon Reviews 2023. UC San Diego. https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
