import json
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def build_model(num_classes, pretrained=True, freeze=True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    if freeze:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    return model


def save_model(model, classes, model_dir):
    destination = Path(model_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), destination / "model.pt")
    with (destination / "classes.json").open("w", encoding="utf-8") as handle:
        json.dump(list(classes), handle, indent=2)


def load_model(model_dir):
    root = Path(model_dir)
    candidates = sorted(root.rglob("model.pt"))
    for model_path in candidates:
        classes_path = model_path.parent / "classes.json"
        if classes_path.exists():
            with classes_path.open(encoding="utf-8") as handle:
                classes = json.load(handle)
            model = build_model(len(classes), pretrained=False, freeze=True)
            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            model.features.eval()
            return model, classes
    raise FileNotFoundError(f"Could not find model.pt and classes.json below {root}")


def predict_image(model, classes, pil_image):
    image = pil_image.convert("RGB")
    tensor = get_transform()(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    scores = {name: float(probabilities[index]) for index, name in enumerate(classes)}
    index = int(torch.argmax(probabilities))
    return {"animal": classes[index], "confidence": float(probabilities[index]), "all_scores": scores}


def log_metric(name, value, step=None):
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri.startswith("azureml"):
        return
    import mlflow
    if step is None:
        mlflow.log_metric(name, float(value))
    else:
        mlflow.log_metric(name, float(value), step=step)