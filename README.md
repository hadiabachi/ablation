# Ablation Project

This repository contains two main pipelines for property listing analysis:

- `LLM_feature_generator.py`: meta-feature extraction from property descriptions using an Ollama LLM.
- `downstream_predictor_models.py`: downstream predictor model training, evaluation, and diagnostic plotting.

## Usage

### Meta-feature extraction

```bash
python LLM_feature_generator.py --city gold_coast --model llama3.1
```

### Valuation pipeline

```bash
python downstream_predictor_models.py --csv sample_data/gold_coast.csv --target LISTING_PRICE --test_size 0.3
```

## Outputs

- `output_data/`: enriched datasets with meta-feature one-hot encodings.
- `reports/`: CSV summaries, metrics, and predictions.
- `figures/`: diagnostic charts, residuals, PDP, SHAP, and PCA outputs.

## Dependencies

Install the Python dependencies before running the scripts:

```bash
pip install -r requirements.txt
```

## Notes

- The modeling script uses a log-1p transform for the target to stabilize variance.
- Adjusted R² is computed for both train and test splits to support robust model selection.
- SHAP is optional and only enabled when the `shap` package is installed.
