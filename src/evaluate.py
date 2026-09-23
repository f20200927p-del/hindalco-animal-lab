import argparse
import json
from pathlib import Path

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model_utils import get_transform, load_model, log_metric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--metrics_out", required=True)
    parser.add_argument("--min_accuracy", type=float, default=0.70)
    args = parser.parse_args()
    model, classes = load_model(args.model_dir)
    dataset = ImageFolder(args.test_data, transform=get_transform())
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    matrix = [[0 for _ in classes] for _ in classes]
    with torch.no_grad():
        for images, labels in loader:
            predictions = model(images).argmax(1)
            for real, predicted in zip(labels.tolist(), predictions.tolist()):
                matrix[real][predicted] += 1
    total = sum(map(sum, matrix))
    overall = sum(matrix[i][i] for i in range(len(classes))) / total if total else 0.0
    per_animal = {}
    for i, name in enumerate(classes):
        count = sum(matrix[i])
        per_animal[name] = matrix[i][i] / count if count else 0.0
    metrics = {"accuracy": overall, "per_animal_accuracy": per_animal, "confusion_matrix": matrix, "classes": classes}
    output = Path(args.metrics_out); output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(f"Overall accuracy: {overall:.4f}")
    print("Per-animal accuracy:"); [print(f"  {name}: {value:.4f}") for name, value in per_animal.items()]
    print("Confusion matrix (rows=real, columns=predicted):"); [print(row) for row in matrix]
    log_metric("test_accuracy", overall)
    if overall < args.min_accuracy:
        raise SystemExit(1)


if __name__ == "__main__":
    main()