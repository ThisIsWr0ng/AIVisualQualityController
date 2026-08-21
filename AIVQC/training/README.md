# AIVQC training backend

The Trainer launches this Python backend as a separate process. It trains an
SSDLite320 MobileNetV3 object detector from a Pascal VOC dataset and reports
machine-readable JSON events to the desktop application.

## Dataset layout

```text
dataset/
├── train/  # images and matching Pascal VOC XML files
├── valid/  # or val/
└── test/   # optional; valid is used when test is absent
```

Every XML file must contain the image size, object class and bounding box. Class
names are discovered from the training split. Label `0` is reserved internally
for the detector background.

## Python environment

Use Python 3.10 or later. Create a virtual environment and install the backend:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

The default packages support CPU. Install the PyTorch build recommended for the
target CUDA or ROCm platform before installing the remaining requirements when
GPU training is required. The application uses the same `cuda` device API for
NVIDIA CUDA and AMD ROCm builds and falls back to CPU.

## Output

Each run gets its own directory containing:

- `job.json` — immutable input configuration,
- `classes.json` — stable class-to-label mapping,
- `best.pt` and `last.pt` — training checkpoints,
- `evaluation.json` — aggregate and per-class test metrics,
- `model.onnx` — deployment model.
