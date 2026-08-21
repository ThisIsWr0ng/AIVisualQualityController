"""Train, evaluate and export the first AIVQC object-detection model."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large

from aivqc_training.dataset import (
    VocDetectionDataset,
    collate_detection_batch,
    discover_classes,
    resolve_split,
)
from aivqc_training.metrics import evaluate_detection_metrics


@dataclass(frozen=True)
class TrainingConfiguration:
    dataset_root: str
    output_root: str
    run_name: str
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 0.005
    workers: int = 0
    device: str = "auto"
    pretrained_backbone: bool = True
    score_threshold: float = 0.25
    seed: int = 42
    max_samples: int = 0


def emit(event_type: str, **payload: Any) -> None:
    """Write one machine-readable event for the desktop process."""

    print(json.dumps({"type": event_type, **payload}, ensure_ascii=False), flush=True)


def load_configuration(path: Path) -> TrainingConfiguration:
    with path.open("r", encoding="utf-8") as file:
        configuration = TrainingConfiguration(**json.load(file))

    if not 1 <= configuration.epochs <= 1000:
        raise ValueError("Epochs must be between 1 and 1000.")
    if not 2 <= configuration.batch_size <= 256:
        raise ValueError("Batch size must be between 2 and 256 for detector training.")
    if not 0 < configuration.learning_rate <= 1:
        raise ValueError("Learning rate must be greater than 0 and at most 1.")
    if not configuration.run_name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Run name may contain letters, numbers, hyphens and underscores.")
    if configuration.max_samples < 0:
        raise ValueError("Max samples cannot be negative.")
    return configuration


def select_device(requested: str) -> tuple[torch.device, str]:
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("Device must be auto, cpu or gpu.")

    if requested in {"auto", "gpu"} and torch.cuda.is_available():
        backend = "AMD ROCm" if torch.version.hip else "NVIDIA CUDA"
        return torch.device("cuda"), f"{backend}: {torch.cuda.get_device_name(0)}"
    if requested == "gpu":
        raise RuntimeError("GPU training was requested, but CUDA/ROCm is unavailable.")
    return torch.device("cpu"), "CPU"


def create_model(class_count: int, pretrained_backbone: bool) -> nn.Module:
    backbone_weights = (
        MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
    )
    return ssdlite320_mobilenet_v3_large(
        weights=None,
        weights_backbone=backbone_weights,
        num_classes=class_count + 1,
    )


def limit_dataset(dataset: Dataset, maximum: int) -> Dataset:
    """Limit a dataset for diagnostics while preserving normal runs by default."""

    if maximum <= 0 or len(dataset) <= maximum:
        return dataset
    return Subset(dataset, range(maximum))


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for images, targets in loader:
        image_batch = [image.to(device) for image in images]
        target_batch = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        losses = model(image_batch, target_batch)
        loss = sum(losses.values())

        if not torch.isfinite(loss):
            raise RuntimeError(f"Training produced a non-finite loss: {loss.item()}.")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
    score_threshold: float,
) -> dict[str, Any]:
    model.eval()
    predictions: list[dict[str, Any]] = []
    targets_for_metrics: list[dict[str, Any]] = []

    for images, targets in loader:
        image_batch = [image.to(device) for image in images]
        outputs = model(image_batch)

        for output, target in zip(outputs, targets):
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu().tolist(),
                    "labels": output["labels"].detach().cpu().tolist(),
                    "scores": output["scores"].detach().cpu().tolist(),
                }
            )
            targets_for_metrics.append(
                {
                    "boxes": target["boxes"].tolist(),
                    "labels": target["labels"].tolist(),
                }
            )

    return evaluate_detection_metrics(
        predictions,
        targets_for_metrics,
        class_names,
        score_threshold,
    )


class OnnxDetectionWrapper(nn.Module):
    """Expose one fixed-size image and tensor outputs to ONNX Runtime."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model([image[0]])[0]
        return output["boxes"], output["scores"], output["labels"]


def export_onnx(model: nn.Module, output_path: Path, device: torch.device) -> None:
    wrapper = OnnxDetectionWrapper(model).to(device).eval()
    sample = torch.zeros((1, 3, 320, 320), dtype=torch.float32, device=device)
    torch.onnx.export(
        wrapper,
        sample,
        output_path,
        input_names=["images"],
        output_names=["boxes", "scores", "labels"],
        dynamic_axes={
            "boxes": {0: "detections"},
            "scores": {0: "detections"},
            "labels": {0: "detections"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    configuration: TrainingConfiguration,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    torch.save(
        {
            "architecture": "ssdlite320_mobilenet_v3_large",
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "configuration": asdict(configuration),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def run(configuration: TrainingConfiguration, configuration_path: Path) -> None:
    random.seed(configuration.seed)
    torch.manual_seed(configuration.seed)

    dataset_root = Path(configuration.dataset_root).resolve()
    train_directory = resolve_split(dataset_root, ("train",))
    validation_directory = resolve_split(dataset_root, ("valid", "val", "validation"))
    test_directory = resolve_split(dataset_root, ("test",)) or validation_directory
    if train_directory is None or validation_directory is None or test_directory is None:
        raise ValueError("Dataset requires train and valid/val directories.")

    class_names = discover_classes(train_directory)
    class_to_label = {name: index for index, name in enumerate(class_names, start=1)}
    train_dataset = VocDetectionDataset(train_directory, class_to_label)
    validation_dataset = VocDetectionDataset(validation_directory, class_to_label)
    test_dataset = VocDetectionDataset(test_directory, class_to_label)
    train_dataset = limit_dataset(train_dataset, configuration.max_samples)
    validation_dataset = limit_dataset(validation_dataset, configuration.max_samples)
    test_dataset = limit_dataset(test_dataset, configuration.max_samples)

    run_directory = Path(configuration.output_root).resolve() / configuration.run_name
    run_directory.mkdir(parents=True, exist_ok=False)
    shutil.copy2(configuration_path, run_directory / "job.json")
    with (run_directory / "classes.json").open("w", encoding="utf-8") as file:
        json.dump(class_to_label, file, indent=2, ensure_ascii=False)

    device, device_name = select_device(configuration.device)
    emit(
        "started",
        device=device_name,
        classes=class_names,
        training_images=len(train_dataset),
        validation_images=len(validation_dataset),
        test_images=len(test_dataset),
        run_directory=str(run_directory),
    )

    generator = torch.Generator().manual_seed(configuration.seed)
    evaluation_loader_arguments = {
        "batch_size": configuration.batch_size,
        "num_workers": configuration.workers,
        "collate_fn": collate_detection_batch,
        "pin_memory": device.type == "cuda",
    }
    training_batch_size = min(configuration.batch_size, len(train_dataset))
    if training_batch_size < 2:
        raise ValueError("Training requires at least two annotated images.")
    train_loader = DataLoader(
        train_dataset,
        batch_size=training_batch_size,
        shuffle=True,
        generator=generator,
        drop_last=len(train_dataset) > training_batch_size
        and len(train_dataset) % training_batch_size == 1,
        num_workers=configuration.workers,
        collate_fn=collate_detection_batch,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        **evaluation_loader_arguments,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **evaluation_loader_arguments)

    model = create_model(len(class_names), configuration.pretrained_backbone).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(
        parameters,
        lr=configuration.learning_rate,
        momentum=0.9,
        weight_decay=0.0005,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_map50 = -1.0
    best_checkpoint = run_directory / "best.pt"
    training_started = time.perf_counter()
    for epoch in range(1, configuration.epochs + 1):
        epoch_started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer, device)
        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            class_names,
            configuration.score_threshold,
        )
        scheduler.step()

        save_checkpoint(
            run_directory / "last.pt",
            model,
            class_names,
            configuration,
            epoch,
            validation_metrics,
        )
        if float(validation_metrics["map50"]) >= best_map50:
            best_map50 = float(validation_metrics["map50"])
            save_checkpoint(
                best_checkpoint,
                model,
                class_names,
                configuration,
                epoch,
                validation_metrics,
            )

        emit(
            "epoch",
            epoch=epoch,
            epochs=configuration.epochs,
            train_loss=train_loss,
            map50=validation_metrics["map50"],
            map50_95=validation_metrics["map50_95"],
            precision=validation_metrics["precision"],
            recall=validation_metrics["recall"],
            elapsed_seconds=time.perf_counter() - epoch_started,
        )

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        device,
        class_names,
        configuration.score_threshold,
    )
    test_metrics.update(
        {
            "dataset_split": test_directory.name,
            "class_names": class_names,
            "device": device_name,
            "duration_seconds": time.perf_counter() - training_started,
        }
    )
    metrics_path = run_directory / "evaluation.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2, ensure_ascii=False)

    onnx_path = run_directory / "model.onnx"
    emit("exporting", format="onnx")
    export_onnx(model, onnx_path, device)
    emit(
        "completed",
        checkpoint_path=str(best_checkpoint),
        onnx_path=str(onnx_path),
        metrics_path=str(metrics_path),
        map50=test_metrics["map50"],
        map50_95=test_metrics["map50_95"],
        precision=test_metrics["precision"],
        recall=test_metrics["recall"],
        f1=test_metrics["f1"],
        duration_seconds=test_metrics["duration_seconds"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    arguments = parser.parse_args()

    try:
        run(load_configuration(arguments.config), arguments.config.resolve())
        return 0
    except Exception as exception:
        emit("failed", message=str(exception), exception_type=type(exception).__name__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
