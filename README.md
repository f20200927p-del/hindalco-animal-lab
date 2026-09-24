# Animal Predictor

This project trains a simple image classifier and exposes it through an Azure
ML-compatible scoring script. Animal names are discovered from the folders in
`data/animals`; no code needs changing when a class is added.

## 1. Install the training dependencies

From the project folder, use Python 3.10+ and run:

```bash
python -m pip install -r requirements.txt
```

The first line of `requirements.txt` selects CPU PyTorch wheels. The optional
web app has its own dependencies:

```bash
python -m pip install -r app/requirements.txt
```

## 2. Check the dataset

Each animal must have a lowercase, space-free folder under `data/animals`, and
at least ten readable images. Run:

```bash
python -m pytest tests -v
```

## 3. Create the train/test split

```bash
python src/prep.py \
  --raw_data data/animals \
  --train_out split/train \
  --test_out split/test
```

This makes a deterministic 80/20 split for every discovered animal.

## 4. Train and evaluate

```bash
python src/train.py --train_data split/train --model_dir models
python src/evaluate.py --model_dir models --test_data split/test --metrics_out metrics
```

Training writes `models/model.pt` and `models/classes.json`. Evaluation writes
`metrics/metrics.json`, prints accuracy and a confusion matrix, and fails if
accuracy is below 0.70. Use `--epochs` to change the training duration.

## 5. Make a local prediction

```bash
python src/predict.py --model_dir models --image demo_images/cat_demo.jpg
```

## 6. Test the scoring contract

First create a compact base64 JSON request. The image is converted to JPEG and
its longest side is reduced to 512 pixels so Azure request-size limits are not
exceeded:

```bash
python src/make_request.py --image demo_images/cat_demo.jpg
```

Then test the same interface used by Azure ML:

```bash
export AZUREML_MODEL_DIR=models
python - <<'PY'
import score

score.init()
with open("sample-request.json", encoding="utf-8") as request:
    print(score.run(request.read()))
PY
```

## 7. Run the Streamlit app

Set the endpoint URL and key without putting secrets in source control:

```bash
export ENDPOINT_URL="https://<your-endpoint>.<region>.inference.ml.azure.com/score"
export ENDPOINT_KEY="<your-endpoint-key>"
streamlit run app/app.py
```

The app uploads a photo, shrinks it locally, sends
`{"image": "<base64>"}` with an `Authorization: Bearer <key>` header, and
displays the prediction, confidence, all-score bar chart, and a warning below
60% confidence. It uses no cloud service other than the configured endpoint.

## 8. Validation evidence

The commands and acceptance-criteria evidence are recorded in
`specs/validation_report.md`.