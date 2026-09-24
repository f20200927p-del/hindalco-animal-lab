"""Basic dataset checks. These tests intentionally do not import torch."""

from pathlib import Path

import pytest
from PIL import Image


DATA_ROOT = Path(__file__).parents[1] / "data" / "animals"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def animal_directories():
    return sorted(path for path in DATA_ROOT.iterdir() if path.is_dir())


def test_animals_folder_exists():
    assert DATA_ROOT.is_dir()


def test_at_least_two_animals_and_valid_names():
    animals = animal_directories()
    assert len(animals) >= 2
    for animal in animals:
        assert animal.name == animal.name.lower()
        assert " " not in animal.name


def test_no_loose_images_at_top_level():
    loose_images = [
        path for path in DATA_ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    assert loose_images == []


@pytest.mark.parametrize("animal", animal_directories(), ids=lambda path: path.name)
def test_each_animal_has_ten_readable_images(animal):
    images = [
        path for path in animal.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    assert len(images) >= 10
    for image_path in images:
        with Image.open(image_path) as image:
            image.verify()