"""Cassava image preparation and stratified partitions."""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image
from torchvision import datasets, transforms

CLASSES = (
    "cassava-bacterial-blight-cbb",
    "cassava-brown-streak-disease-cbsd",
    "cassava-green-mottle-cgm",
    "cassava-healthy",
    "cassava-mosaic-disease-cmd",
)
KAGGLE_LABELS = {
    0: CLASSES[0],
    1: CLASSES[1],
    2: CLASSES[2],
    3: CLASSES[4],
    4: CLASSES[3],
}


class ToRGB:
    def __call__(self, image: Image.Image):
        return image.convert("RGB")


def transform(training=False):
    steps = [ToRGB(), transforms.Resize((224, 224))]
    if training:
        steps += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
    return transforms.Compose(
        steps
        + [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def load_data(root, seed):
    training = datasets.ImageFolder(root, transform=transform(True))
    validation = datasets.ImageFolder(root, transform=transform(False))
    if tuple(training.classes) != CLASSES:
        raise ValueError(f"Expected these five class folders: {', '.join(CLASSES)}")
    groups = defaultdict(list)
    for index, target in enumerate(training.targets):
        groups[target].append(index)
    rng = random.Random(seed)
    train_indices, val_indices = [], []
    for indices in groups.values():
        if len(indices) < 2:
            raise ValueError("Each class requires at least two images.")
        rng.shuffle(indices)
        count = min(len(indices) - 1, max(1, round(0.2 * len(indices))))
        val_indices.extend(indices[:count])
        train_indices.extend(indices[count:])
    return training, validation, sorted(train_indices), sorted(val_indices)


def prepare(images, labels, output):
    frame = pd.read_csv(labels, dtype={"image_id": str})
    if not {"image_id", "label"}.issubset(frame.columns) or frame.empty:
        raise ValueError("CSV needs image_id and label columns.")
    if frame.image_id.isna().any() or frame.image_id.duplicated().any():
        raise ValueError("Image IDs must be unique and nonempty.")
    if not frame.label.isin(KAGGLE_LABELS).all():
        raise ValueError("Labels must be integers from 0 to 4.")
    if set(frame.label) != set(KAGGLE_LABELS):
        raise ValueError("The CSV must contain all five classes.")
    copies = []
    for row in frame.itertuples(index=False):
        source = (images / row.image_id).resolve()
        if not source.is_relative_to(images.resolve()) or not source.is_file():
            raise ValueError(f"Invalid image path: {row.image_id}")
        copies.append((source, output / KAGGLE_LABELS[row.label] / source.name))
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Prepared {len(copies)} images in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/train"))
    args = parser.parse_args()
    try:
        prepare(args.images, args.labels, args.output)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
