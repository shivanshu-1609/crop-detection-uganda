"""Classify a cassava leaf image."""

import argparse
import json
from pathlib import Path

import torch
from data import transform
from model import device, load_checkpoint
from PIL import Image


@torch.inference_mode()
def predict(checkpoint_path, image_path):
    target_device = device()
    model, checkpoint = load_checkpoint(checkpoint_path, target_device)
    with Image.open(image_path) as image:
        inputs = transform()(image).unsqueeze(0).to(target_device)
    probabilities = model(inputs).softmax(1)[0].cpu().tolist()
    best = max(range(len(probabilities)), key=probabilities.__getitem__)
    return {
        "class": checkpoint["classes"][best],
        "probabilities": dict(zip(checkpoint["classes"], probabilities, strict=True)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("outputs/resnet50/model.pt")
    )
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.checkpoint, args.image), indent=2))
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
