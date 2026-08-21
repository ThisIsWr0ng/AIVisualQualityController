"""Pascal VOC dataset loading and validation."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def resolve_split(dataset_root: Path, names: tuple[str, ...]) -> Path | None:
    """Return the first existing split directory from a list of accepted names."""

    for name in names:
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    return None


def discover_classes(training_directory: Path) -> list[str]:
    """Discover a deterministic class list from the training annotations."""

    classes: set[str] = set()
    for annotation_path in sorted(training_directory.glob("*.xml")):
        root = ElementTree.parse(annotation_path).getroot()
        for item in root.findall("object"):
            name = (item.findtext("name") or "").strip()
            if name:
                classes.add(name)

    if not classes:
        raise ValueError(f"No object classes found in {training_directory}.")

    return sorted(classes, key=str.casefold)


class VocDetectionDataset(Dataset):
    """Torch dataset backed by images and matching Pascal VOC XML files."""

    def __init__(self, directory: Path, class_to_label: dict[str, int]) -> None:
        self.directory = directory
        self.class_to_label = class_to_label
        self.annotation_paths = sorted(directory.glob("*.xml"))

        if not self.annotation_paths:
            raise ValueError(f"No Pascal VOC XML annotations found in {directory}.")

    def __len__(self) -> int:
        return len(self.annotation_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        annotation_path = self.annotation_paths[index]
        root = ElementTree.parse(annotation_path).getroot()
        image_path = self._resolve_image_path(root, annotation_path)

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
            image_tensor = pil_to_tensor(image).float().div(255.0)

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []

        for item in root.findall("object"):
            class_name = (item.findtext("name") or "").strip()
            box = item.find("bndbox")
            if class_name not in self.class_to_label or box is None:
                continue

            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
            if xmax <= xmin or ymax <= ymin:
                continue

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_label[class_name])
            areas.append((xmax - xmin) * (ymax - ymin))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        return image_tensor, target

    def _resolve_image_path(self, root: ElementTree.Element, annotation_path: Path) -> Path:
        file_name = (root.findtext("filename") or "").strip()
        if file_name:
            declared_path = self.directory / file_name
            if declared_path.is_file():
                return declared_path

        for extension in IMAGE_EXTENSIONS:
            candidate = annotation_path.with_suffix(extension)
            if candidate.is_file():
                return candidate

        raise FileNotFoundError(f"No image found for annotation {annotation_path.name}.")


def collate_detection_batch(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, torch.Tensor], ...]]:
    """Keep variable-sized detection targets as lists instead of stacking them."""

    return tuple(zip(*batch))
