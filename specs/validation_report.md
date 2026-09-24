# Validation report

This report records the validation run for the requested Phase 3 checks.

| Criterion | Status | Evidence |
|---|---|---|
| AC1 | PASS | `pytest tests -v`: `8 passed in 0.07s`. |
| AC2 | PASS | `prep.py`: table contains `cat`, `chicken`, `cow`, `dog`, `horse`, each `16` train / `4` test. |
| AC3 | PASS | Artifact check: `models/model.pt: True`; `models/classes.json: True`. |
| AC4 | PASS | `evaluate.py`: `Overall accuracy: 0.8500` and `Confusion matrix (rows=real, columns=predicted)`; `metrics.json` exists. |
| AC5 | PASS | `metrics/metrics.json` reports `accuracy: 0.85`, above the `0.70` quality gate. |
| AC6 | PASS | `predict.py`: `Prediction: cat  (confidence 88%)`. |
| AC7 | PASS | `score.run()`: `{"animal": "cat", "confidence": 0.881878..., "all_scores": {...}}`. |
| AC8 | PASS | Tests and scripts discover folders/classes dynamically; no animal count or names are hard-coded. |
| AC9 | PASS | `README.md` documents installation, tests, prep, train, evaluate, predict, score, and app steps. |

## Commands

The key output lines from the validation run are summarized above. The
validation commands were:

```text
python -m pip install -r requirements.txt
python -m pytest tests -v
python src/prep.py --raw_data data/animals --train_out split/train --test_out split/test
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
python src/make_request.py --image demo_images/cat_demo.jpg
AZUREML_MODEL_DIR=models; import score; score.init(); score.run(...)
```