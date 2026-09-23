import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model_utils import build_model, get_transform, log_metric, save_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--from_scratch", action="store_true")
    args = parser.parse_args()
    dataset = ImageFolder(args.train_data, transform=get_transform())
    model = build_model(len(dataset.classes), pretrained=not args.from_scratch, freeze=not args.from_scratch)
    if args.from_scratch:
        for parameter in model.parameters():
            parameter.requires_grad = True
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        model.features.eval()
        total_loss = correct = total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
        loss_value, accuracy = total_loss / total, correct / total
        print(f"Epoch {epoch + 1}/{args.epochs} - loss: {loss_value:.4f} - train accuracy: {accuracy:.4f}")
        log_metric("train_loss", loss_value, epoch + 1)
        log_metric("train_accuracy", accuracy, epoch + 1)
    save_model(model.cpu(), dataset.classes, args.model_dir)


if __name__ == "__main__":
    main()