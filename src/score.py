import base64
import io
import json
import os

from PIL import Image

from model_utils import load_model, predict_image


model = None
classes = None


def init():
    global model, classes
    model, classes = load_model(os.environ["AZUREML_MODEL_DIR"])


def run(raw_data):
    try:
        payload = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
        image = Image.open(io.BytesIO(base64.b64decode(payload["image"])))
        result = predict_image(model, classes, image)
        print(f"Prediction: {result['animal']} confidence={result['confidence']:.6f}")
        return result
    except Exception as exc:
        return {"error": str(exc)}