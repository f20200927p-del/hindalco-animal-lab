import argparse
import random
import shutil
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_animal_root(raw_data):
    root = Path(raw_data)
    directories = [p for p in root.iterdir() if p.is_dir()]
    if len(directories) == 1:
        nested = [p for p in directories[0].iterdir() if p.is_dir()]
        if nested:
            return directories[0]
    return root


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", required=True)
    parser.add_argument("--train_out", required=True)
    parser.add_argument("--test_out", required=True)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--min_images", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    source = find_animal_root(args.raw_data)
    animals = sorted(p for p in source.iterdir() if p.is_dir())
    if len(animals) < 2:
        raise ValueError("At least two animal directories are required")
    rng = random.Random(args.seed)
    rows = []
    for animal in animals:
        images = sorted(p for p in animal.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        for image in images:
            with Image.open(image) as opened:
                opened.verify()
        if len(images) < args.min_images:
            raise ValueError(f"{animal.name} has {len(images)} images; need at least {args.min_images}")
        shuffled = images[:]
        rng.shuffle(shuffled)
        test_count = max(1, int(round(len(shuffled) * args.test_ratio)))
        test_images, train_images = shuffled[:test_count], shuffled[test_count:]
        for output, selected in ((Path(args.train_out), train_images), (Path(args.test_out), test_images)):
            destination = output / animal.name
            destination.mkdir(parents=True, exist_ok=True)
            for image in selected:
                shutil.copy2(image, destination / image.name)
        rows.append((animal.name, len(train_images), len(test_images)))
    print("Animal       Train  Test  Total")
    print("-----------  -----  ----  -----")
    for name, train, test in rows:
        print(f"{name:<11}  {train:>5}  {test:>4}  {train + test:>5}")


if __name__ == "__main__":
    main()