# AIVQC User Manual

> **Manual version:** 0.1
>
> **Applies to:** AIVQC `0.5.4-alpha.1` development build
>
> **Last updated:** 2026-08-22

## 1. About this manual

This manual describes the functions available in the current AIVQC development build. It covers:

- creating and opening a Trainer project,
- importing project images,
- training an object-detection model from a Pascal VOC dataset,
- importing and validating an existing ONNX model,
- exporting an AIVQC deployment package,
- loading a verified package in Production,
- running an inspection on a local image,
- interpreting results and resolving common problems.

Functions defined in `Scope.md` but not yet implemented are listed in [Current limitations](#12-current-limitations). Do not use this development build as the sole production quality or safety control.

## 2. Applications

AIVQC currently contains two desktop applications:

- **AIVQC Trainer** prepares project data, trains or imports a model, and exports a deployment package.
- **AIVQC Production** imports a deployment package and runs local ONNX inference on selected image files.

The normal workflow is:

`Images and annotations → Trainer → model.onnx → .aivqcpkg → Production → OK/NOK/Error`

## 3. Safety and security

- Validate every model and recipe with representative production data before use.
- Do not interpret an application error or missing result as OK.
- Do not use the current software output as a certified machine-safety function.
- Load packages only from a trusted source. Production verifies package structure and model integrity, but the current package format is not yet cryptographically signed.
- Do not place passwords, access tokens, confidential notes, or personal data in project names, product IDs, recipe IDs, file names, or deployment metadata.
- Keep datasets, models, packages, and inspection images in access-controlled directories with appropriate backups.
- Commercial production use requires a separate commercial license. See `LICENSE.md` and the repository README.

## 4. System requirements

### 4.1. Desktop applications

- Windows,
- a .NET SDK compatible with `Aivqc.sln` when running from source,
- sufficient storage for datasets, project copies, training outputs, and deployment packages,
- CPU inference support through ONNX Runtime.

### 4.2. Training backend

- Python 3.10 or later,
- packages from `training/requirements.txt`,
- CPU training or a correctly installed PyTorch CUDA/ROCm environment for GPU training.

GPU availability depends on the installed PyTorch build and hardware drivers. Selecting **GPU required** causes training to fail rather than silently use the CPU when no compatible accelerator is available.

## 5. Starting the applications

From the `AIVQC` directory:

```powershell
dotnet restore Aivqc.sln
dotnet build Aivqc.sln
dotnet run --project src/Aivqc.Trainer
```

To run Production:

```powershell
dotnet run --project src/Aivqc.Production
```

To prepare the Python environment:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r training\requirements.txt
```

Packaged installers are not part of the current development build.

## 6. AIVQC Trainer quick start

1. Start **AIVQC Trainer**.
2. Select **New project**.
3. Choose the parent directory in which the project folder will be created.
4. Enter or edit the project name. Trainer saves project changes automatically.
5. Select **Copy into project** or **Reference originals**.
6. Select **Import images…** and choose JPG, PNG, BMP, or WebP files.
7. Prepare or select a Pascal VOC dataset if you want to train a model.
8. Configure the training parameters and select **Start training**.
9. After training, review the reported evaluation metrics.
10. Configure Product ID, Recipe ID, author, and the default defect threshold.
11. Select **Export deployment package** and save the `.aivqcpkg` file.

Alternatively, select **Import ONNX model** to use a compatible existing model instead of training one.

## 7. Working with Trainer projects

### 7.1. Creating a project

Select **New project** and choose a parent directory. Trainer creates a dedicated project directory containing:

```text
project-name/
├── project.aivqc.json
├── images/
└── thumbnails/
```

The project manifest contains the project identity, product information, imported-image metadata, hashes, warnings, and storage modes. Do not edit `project.aivqc.json` manually while Trainer is open.

### 7.2. Opening a project

Select **Open project** and choose an existing AIVQC project. Recently opened projects also appear under **Recent projects**.

If referenced source images have been moved or deleted, Trainer reports them as missing. Restore the files at their original locations or import them again.

### 7.3. Renaming and autosave

Edit the project name in the current-project panel. Changes are saved automatically. Confirm the displayed project path before importing large datasets.

### 7.4. Importing images

Supported formats are JPG/JPEG, PNG, BMP, and WebP.

Choose one storage mode before selecting **Import images…**:

- **Copy into project** copies each accepted image into the project. This is more portable but uses additional disk space.
- **Reference originals** keeps the image in its current location. This avoids duplication, but moving or deleting the original breaks the reference.

During import, Trainer:

- decodes and validates each image,
- records dimensions and actual image format,
- calculates SHA-256,
- creates a thumbnail,
- records quality warnings,
- skips duplicate image content even when file names differ.

Review the **Images**, **Warnings**, and **Missing** counters after import.

## 8. Preparing and training a model

### 8.1. Dataset format

The current backend trains an SSDLite320 MobileNetV3 object detector from Pascal VOC XML annotations. Use this structure:

```text
dataset/
├── train/  # images and matching XML annotations
├── valid/  # "val" is also accepted
└── test/   # optional; validation data is used when absent
```

Each image must have a matching Pascal VOC XML file containing:

- the image dimensions,
- each object class name,
- a bounding box for each annotated object.

Class names are discovered from the training split. Keep spelling and capitalization consistent across all annotations. Label `0` is reserved internally for the detector background.

Do not place consecutive frames from the same video across training, validation, and test splits. This can produce misleadingly high metrics.

### 8.2. Selecting a dataset

In the training panel, select **Select dataset…** and choose the dataset root containing `train` and `valid` or `val`.

Trainer reports an error when required directories or Pascal VOC annotations are missing.

### 8.3. Training settings

- **Epochs** controls the number of complete passes over the training set.
- **Batch size** controls how many images are processed together. Reduce it if GPU or system memory is exhausted.
- **Learning rate** controls the optimizer step size. Use the default unless model evaluation indicates a reason to change it.
- **Device: Auto** uses a compatible accelerator when available and otherwise uses CPU.
- **Device: CPU** forces CPU training.
- **Device: GPU required** stops with an error if a compatible GPU environment is unavailable.
- **Use pretrained ImageNet backbone** initializes the feature extractor from pretrained weights when enabled.

### 8.4. Running training

Select **Start training**. Trainer displays:

- current epoch and total epochs,
- training loss,
- progress,
- mAP@0.50,
- mAP@0.50:0.95,
- precision,
- recall,
- F1 score.

Select **Cancel** to stop the active job. An incomplete run is retained for diagnostics.

Each completed run creates a separate output directory containing:

- `job.json` — immutable training configuration,
- `classes.json` — stable class mapping,
- `best.pt` and `last.pt` — PyTorch checkpoints,
- `evaluation.json` — aggregate and per-class metrics,
- `model.onnx` — exported deployment model.

After successful training, Trainer automatically imports the resulting ONNX model.

### 8.5. Evaluating results

Do not approve a model based only on training loss. Review at least:

- per-class precision and recall,
- false accepts and false rejects on representative data,
- mAP values,
- difficult and previously unseen production images,
- inference performance on the intended Production hardware.

Required acceptance values are deployment-specific and are not yet enforced automatically by this development build.

## 9. Importing an existing ONNX model

1. Place `classes.json` next to the ONNX model.
2. Select **Import ONNX model** or **Browse…** in the imported-model panel.
3. Choose the `.onnx` file.
4. Wait while Trainer validates the model with ONNX Runtime.
5. Review the displayed input, output, size, and SHA-256 information.

Without `classes.json`, Trainer can inspect the model but cannot export a deployment package.

The ONNX model must match the detection contract expected by the current Production runtime. A model that can be opened by ONNX Runtime is not necessarily compatible with AIVQC output processing.

## 10. Exporting a deployment package

After importing or training a compatible model:

1. Enter a stable **Product ID**.
2. Enter a stable **Recipe ID**.
3. Enter the person responsible in **Created by**.
4. Set the default defect threshold.
5. Select **Export deployment package**.
6. Save the file with the `.aivqcpkg` extension.

The package contains a strict manifest, ONNX model, class definitions, preprocessing metadata, region of interest, and per-class thresholds. Trainer verifies the newly created package before reporting success.

Keep exported packages under version control or in an access-controlled release directory. Do not overwrite a validated package with unrelated contents while retaining the same product and recipe identifiers.

## 11. AIVQC Production

### 11.1. Quick start

1. Start **AIVQC Production**.
2. Select **Load package**.
3. Choose an `.aivqcpkg` exported by Trainer.
4. Confirm that the active-deployment panel shows the intended product, recipe, model, input, and classes.
5. Select **Select image** and choose a supported image.
6. Select **Run inspection**.
7. Review the decision, detections, confidence, latency, and session counters.

### 11.2. Loading a package

Production validates:

- the package structure and schema,
- required files and metadata,
- the model SHA-256,
- model input dimensions against the manifest,
- class names and thresholds.

A verified package is imported into a controlled local cache. If validation fails, Production does not activate it and displays an error.

The **Raw ONNX** option is intended only for development. It does not provide deployment-package integrity or recipe metadata and may use numeric labels when `classes.json` is unavailable.

### 11.3. Selecting an image

Select **Select image** and choose a JPG/JPEG, PNG, BMP, or WebP image. Confirm the preview, image name, and resolution before inspection.

You cannot change the package, model, or image while an inspection is active. Cancel the inspection first.

### 11.4. Running an inspection

Select **Run inspection**. Production preprocesses the image, performs ONNX inference locally on the CPU, applies the recipe thresholds, draws accepted detections, and returns:

- **OK** — no accepted defect detection is at or above its threshold,
- **NOK** — at least one defect detection is at or above its threshold,
- **Error** — the model, image, preprocessing, inference, or result processing failed.

An Error must never be interpreted as OK. Investigate it before continuing a real inspection process.

Select **Cancel inspection** to request cancellation of an active inspection.

### 11.5. Thresholds

Verified packages provide per-class thresholds. When the recipe locks thresholds, the Production slider is disabled. Modify the recipe in Trainer and export a new package rather than bypassing the lock.

Raw ONNX development mode uses the editable global threshold shown in Production.

### 11.6. Session statistics

The bottom panel displays counts for:

- inspected images,
- OK results,
- NOK results,
- inference errors.

These counters describe only the current application session. Persistent shift reporting and Server synchronization are not implemented yet.

## 12. Current limitations

The following planned functions are not available in the current development build:

- live camera capture and continuous production streaming,
- physical trigger input and product tracking,
- PLC, digital I/O, heartbeat, reject mechanism, or line-stop integration,
- calibrated dimensional measurements,
- built-in annotation editor,
- automatic or assisted labeling,
- AIVQC Server, user accounts, Operator/Technician/Expert enforcement, and multi-line synchronization,
- transfer of Production images to Trainer,
- remote deployment of packages from Trainer to selected lines,
- cryptographic package signing,
- automatic application updates,
- persistent production statistics and audit logs,
- Polish UI translation.

Navigation buttons for future workflow areas may be visible before those workspaces are implemented.

## 13. Troubleshooting

### Trainer cannot import an image

- Confirm that the format is JPG/JPEG, PNG, BMP, or WebP.
- Confirm that the file is a valid image rather than a renamed or damaged file.
- Check read permission and available disk space.
- Review whether identical image content was already imported.

### A referenced image is missing

The project uses **Reference originals**, and the source file was moved, renamed, disconnected, or deleted. Restore it at the original path or import it again.

### Dataset selection fails

- Confirm the presence of `train` and `valid` or `val` directories.
- Confirm that images have matching Pascal VOC XML annotations.
- Confirm that annotations contain valid image dimensions, classes, and bounding boxes.

### Python environment is not found

- Install Python 3.10 or later.
- Create the virtual environment and install `training/requirements.txt`.
- Restart Trainer after changing the Python installation.

### GPU-required training fails

- Confirm that the GPU driver is installed.
- Confirm that the installed PyTorch build supports the target CUDA or ROCm environment.
- Select **Auto** or **CPU** to test without GPU acceleration.

### Package export is unavailable

- Import or train a compatible ONNX model.
- Place a valid `classes.json` beside an externally imported model.
- Complete Product ID, Recipe ID, and author metadata.
- Review the package-export status message.

### Production rejects a package

- Do not modify the archive manually.
- Export the package again from a compatible Trainer version.
- Confirm that the file transfer completed without truncation.
- Review the exact schema, checksum, model-contract, or compatibility error displayed by Production.

### Production cannot run an inspection

- Load a compatible package or development ONNX model.
- Select a supported image.
- Wait for any current load or inspection operation to finish.
- Review the inspection status message.

### The result appears incorrect

- Confirm that the correct product and recipe package is active.
- Confirm class names and thresholds.
- Check whether the test image resembles the model's validated production conditions.
- Review image scale, orientation, lighting, blur, and exposure.
- Retain the image and model/package version for expert analysis.

## 14. Data and backup locations

Trainer projects are stored in the directory selected when the project is created. Training outputs are stored in a separate directory for each run. Production imports verified packages into its local application cache.

Back up at least:

- Trainer project manifests and copied images,
- original referenced images,
- Pascal VOC annotations and dataset split definitions,
- `classes.json`, training configurations, metrics, and checkpoints,
- exported `.aivqcpkg` files that were validated or deployed.

Do not rely on the Production package cache as the only copy of a model or recipe.

## 15. Glossary

- **Defect class** — a named type of product nonconformity.
- **Detection** — a predicted defect class, confidence, and image bounding box.
- **Model** — the trained ONNX artifact used for inference.
- **Product** — the physical item being inspected.
- **Recipe** — model, classes, preprocessing, thresholds, and other inspection settings for a product.
- **Deployment package** — a versioned `.aivqcpkg` archive transferred from Trainer to Production.
- **Threshold** — minimum confidence required for a detection to affect the decision.
- **OK** — no accepted defect was detected.
- **NOK** — at least one accepted defect was detected.
- **Error** — no valid inspection decision could be produced.

## 16. Support and licensing

Author: **Dawid Oleśko**

Contact: [oleskodawid@gmail.com](mailto:oleskodawid@gmail.com)

AIVQC is source-available under the PolyForm Noncommercial License 1.0.0. Commercial use requires a separate written commercial license.
