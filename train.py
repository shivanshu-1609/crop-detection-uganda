"""Train a cassava CNN or ResNet50 classifier."""

import argparse
import math
import random
from pathlib import Path

import pandas as pd
import torch
from data import load_data
from model import build_model, device, load_checkpoint
from torch import nn
from torch.utils.data import DataLoader, Subset


@torch.inference_mode()
def evaluate(model, loader, target_device):
    model.eval()
    loss, correct, count = 0.0, 0, 0
    predictions, targets = [], []
    for images, labels in loader:
        logits = model(images.to(target_device))
        loss += nn.functional.cross_entropy(
            logits, labels.to(target_device), reduction="sum"
        ).item()
        predicted = logits.argmax(1).cpu()
        correct += (predicted == labels).sum().item()
        count += len(labels)
        predictions.extend(predicted.tolist())
        targets.extend(labels.tolist())
    return loss / count, correct / count, predictions, targets


def train(args):
    if (
        min(args.epochs, args.batch_size, args.patience) < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError(
            "Epochs, batch size, patience, and learning rate must be positive."
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty output directory.")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    training, validation, train_ids, val_ids = load_data(args.data, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        Subset(training, train_ids),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(Subset(validation, val_ids), batch_size=args.batch_size)
    target_device = device()
    model = build_model(
        args.model, len(training.classes), pretrained=args.model == "resnet50"
    ).to(target_device)
    frozen = args.model == "resnet50" and not args.fine_tune
    if frozen:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.2)
    args.output.mkdir(parents=True, exist_ok=True)
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if frozen:
            model.eval()
            model.fc.train()
        training_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(images.to(target_device)), labels.to(target_device)
            )
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(labels)
        val_loss, accuracy, _, _ = evaluate(model, val_loader, target_device)
        training_loss /= len(train_ids)
        if not math.isfinite(val_loss) or not math.isfinite(training_loss):
            raise ValueError("Training produced a non-finite loss.")
        history.append(
            dict(
                epoch=epoch,
                training_loss=training_loss,
                validation_loss=val_loss,
                accuracy=accuracy,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )
        print(
            f"Epoch {epoch}: loss={training_loss:.4f}, "
            f"val_loss={val_loss:.4f}, accuracy={accuracy:.2%}"
        )
        if val_loss < best:
            best, stale = val_loss, 0
            checkpoint = dict(
                architecture=args.model,
                classes=training.classes,
                epoch=epoch,
                validation_loss=val_loss,
                accuracy=accuracy,
                seed=args.seed,
                validation_files=[
                    Path(validation.samples[i][0]).relative_to(args.data).as_posix()
                    for i in val_ids
                ],
                state_dict={
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
            )
            torch.save(checkpoint, args.output / "model.pt")
        else:
            stale += 1
        pd.DataFrame(history).to_csv(args.output / "history.csv", index=False)
        scheduler.step()
        if stale >= args.patience:
            break
    model, checkpoint = load_checkpoint(args.output / "model.pt", target_device)
    _, _, predicted, targets = evaluate(model, val_loader, target_device)
    count = len(training.classes)
    matrix = torch.bincount(
        torch.tensor(targets) * count + torch.tensor(predicted), minlength=count**2
    ).reshape(count, count)
    pd.DataFrame(
        matrix.numpy(), index=training.classes, columns=training.classes
    ).to_csv(args.output / "confusion_matrix.csv", index_label="true_class")
    print(f"Saved best epoch {checkpoint['epoch']} to {args.output / 'model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/resnet50"))
    parser.add_argument("--model", choices=["cnn", "resnet50"], default="resnet50")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-tune", action="store_true")
    args = parser.parse_args()
    try:
        train(args)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")
