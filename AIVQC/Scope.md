# AIVQC — Project Scope and Assumptions

> This is a living project document. Update it as requirements, decisions, and test results evolve.

## 1. Vision

AI Visual Quality Controller (AIVQC) will be a universal, well-optimized, and easy-to-deploy visual quality inspection system for small and medium-sized manufacturing operations.

The system should detect different types of defects regardless of industry or product type. It should lower the machine-learning barrier: a user should be able to collect and annotate data, train and evaluate a model, and deploy it to production without manually transferring multiple configuration files.

The project consists of two applications:

1. **AIVQC Trainer** — dataset preparation, training, testing, benchmarking, and deployment package export.
2. **AIVQC Production** — inspection execution on production equipment.

## 2. Target users

- process and quality engineers configuring inspections,
- technicians preparing cameras, datasets, and models,
- line operators running approved inspections,
- small and medium-sized manufacturers without an internal machine-learning team.

## 3. AIVQC Trainer

### 3.1. Purpose

Guide users through the complete workflow, from image collection and dataset preparation to a tested, production-ready deployment package.

### 3.2. Core scope

- create a project for a product and inspection task,
- capture images from a camera and import existing images or videos,
- display a live camera preview,
- configure supported camera properties such as resolution, exposure, gain, white balance, focus, FPS, and region of interest,
- store camera settings with the project and deployment package,
- calibrate measurements using a reference of known dimensions,
- define measured product features, units, nominal values, and lower and upper tolerance limits,
- provide efficient defect annotation for large sets of similar images,
- manage defect classes and examples of conforming products,
- split data into training, validation, and test sets without leaking related video frames between sets,
- select supported model architectures with clear speed, accuracy, and hardware trade-offs,
- offer safe training defaults and an advanced configuration mode,
- present training progress and results,
- compare models on the same test set,
- benchmark models on target hardware or a defined hardware profile,
- export a complete, versioned deployment package.

### 3.3. Assisted and automated dataset creation

Introduce automation incrementally:

1. propagate annotations across similar or consecutive video frames,
2. pre-label data with an existing model and let the user approve or correct the result,
3. use active learning to identify the most valuable samples to annotate,
4. detect duplicates, blurred images, and exposure problems,
5. optionally use foundation or segmentation models to accelerate object selection,
6. use synthetic data only to supplement real data and validate it separately.

Fully unattended annotation is not an initial goal. Incorrect automatic labels can silently reduce model quality.

### 3.4. Model evaluation

The Trainer should report at least:

- precision, recall, and F1 score for every defect class,
- mAP for detection and IoU for segmentation,
- a confusion matrix,
- false-accept and false-reject counts,
- results by batch or data series, not only global averages,
- inference time, FPS, CPU/GPU/RAM/VRAM usage, and model warm-up time,
- a gallery of difficult and incorrectly classified samples,
- suggested per-class thresholds that the user can accept or modify.

### 3.5. Deployment package

A self-contained deployment package should include:

- the model, its format, and version,
- the product and inspection-recipe identifiers,
- defect classes and their default thresholds,
- required preprocessing, input size, and normalization,
- camera configuration and region of interest,
- measurement definitions, units, calibration data, and tolerances,
- validation metrics and hardware requirements,
- the application and environment versions used to create it,
- date, author, and optional deployment notes,
- checksums for detecting damaged or replaced files.

## 4. AIVQC Production

### 4.1. Purpose

Provide stable, fast, and simple production inspection with minimal operator interaction.

### 4.2. Core scope

- select a product or inspection recipe that automatically loads the correct model and settings,
- allow authorized users to select a specific model version,
- restore camera configuration from the deployment package,
- clearly report unsupported or unapplied camera settings,
- display the camera image and detection overlays,
- perform recipe-defined measurements such as width, height, diameter, distance, or area,
- display measured values, units, nominal values, and permitted ranges,
- reject a product when any required measurement is outside its specification,
- mark an unreliable measurement as invalid; an invalid measurement must never be treated as conforming,
- start, stop, and monitor detection,
- configure a separate threshold for each defect class,
- optionally lock threshold editing for operators,
- return an unambiguous OK, NOK, or Error/Indeterminate result,
- display statistics for the current shift and selected periods,
- log events, errors, and configuration changes,
- optionally retain NOK images and sampled OK images according to a retention policy,
- operate locally without mandatory Internet access,
- safely roll back to the last known working model.

### 4.3. Statistics

- counts and percentages of OK and NOK products,
- defect counts by class,
- confirmed false rejects and false accepts when a review workflow is available,
- FPS, latency, uptime, and dropped-frame counts,
- trends by time, shift, batch, and product,
- camera and application health,
- CSV export, with plant-system integrations added later.

## 5. Domain model

- **Product** — the physical item being inspected.
- **Inspection recipe** — a versioned combination of product, model, camera settings, region of interest, thresholds, measurement specification, and decision rules.
- **Model** — a versioned ML artifact that can be used by one or more recipes.
- **Defect** — a nonconformity class detected by a model.
- **Product specification** — a versioned collection of measured features, nominal values, units, and tolerances.
- **Measurement calibration** — parameters that convert image pixels into physical units and, when required, correct lens distortion and perspective.
- **Measurement** — a feature value with validity and specification-conformance status.
- **Inspection** — one product or image evaluation ending with OK, NOK, or Error/Indeterminate.
- **User profile** — an authenticated identity with one or more assigned roles and an auditable set of permissions.
- **Shared resource** — a versioned model, deployment package, recipe, dataset artifact, localization resource, report, or other file synchronized through the AIVQC Server.

A product and a model are not the same entity. A product can use successive model versions, while a recipe can change thresholds without retraining the model.

## 6. Cross-cutting requirements

### 6.1. User profiles and authorization

Access must be controlled through role-based access control. The initial roles are:

- **Operator** — select an approved product or recipe, start and stop inspections, view live results and permitted statistics, and acknowledge operational messages. An Operator cannot modify models, calibration, specifications, or protected thresholds.
- **Technician** — all Operator permissions plus camera and station configuration, calibration, diagnostics, installation or activation of approved deployment packages, and recipe or threshold adjustment within limits defined by an Expert.
- **Expert** — all Technician permissions plus dataset management, model training and evaluation, specification definition, model/package approval and publication, unrestricted recipe configuration, user and role administration, and access to audit and system-management functions.

Authorization must be enforced by the application and server, not only by hiding UI controls. Every security-sensitive action must record the user, timestamp, station, action, and affected resource version in the audit log. The detailed permission matrix must be configurable without changing the meaning of the three base roles.

### 6.2. AIVQC Server

A shared server component will connect AIVQC Trainer and AIVQC Production and provide:

- centralized user accounts, authentication, role assignments, sessions, and account revocation,
- password storage using a current, salted, adaptive password-hashing algorithm; passwords must never be stored or logged in plaintext or with reversible encryption,
- versioned storage for models, deployment packages, recipes, calibration data, specifications, localization resources, reports, and other approved shared assets,
- controlled publication from Trainer and controlled retrieval by Production,
- immutable identifiers, version history, checksums, and integrity verification for distributed packages,
- approval states such as Draft, Validated, Approved, Published, Retired, and Revoked,
- audit logging for authentication, downloads, publication, activation, rollback, configuration changes, and administrative actions,
- encrypted network communication and authenticated API access,
- backup, restore, retention, storage-quota, and disaster-recovery policies,
- compatibility checks between server, package, Trainer, and Production versions,
- synchronization status, conflict handling, retry behavior, and clear error reporting.

Production must keep an authenticated, integrity-checked local cache of approved recipes and packages. A temporary server or network outage must not stop an already authorized production inspection. Offline actions and statistics must be queued and synchronized after connectivity returns. Revoked packages must be blocked as soon as the station receives the revocation, subject to a separately defined emergency/offline policy.

Trainer and Production must access shared resources through documented server APIs; they must not directly access the server database or shared storage directories.

### 6.3. General requirements

- simple installation and configuration on industrial PCs,
- CPU support with optional GPU and edge-device acceleration,
- modular support for cameras and model formats,
- deterministic behavior and resilience to camera loss or damaged packages,
- versioned configuration and a complete audit log,
- least-privilege authorization based on Operator, Technician, and Expert roles,
- recipe backup and migration,
- performance measured on target hardware,
- local data storage with a clear retention policy,
- an architecture that can later integrate with PLCs, signals, reject mechanisms, MES, or APIs,
- English as the default language for the UI, source code, documentation, configuration keys, logs, and technical identifiers,
- Polish as an optional UI translation,
- UI language switching without stopping an active inspection,
- all user-facing text, errors, and unit labels stored in localization resources rather than hard-coded in application code,
- language preference stored per user or station while technical data and package formats remain language-neutral,
- version tracking for the calibration and specification used in each inspection,
- a warning or lockout after changing the camera, lens, resolution, focus, or station geometry when the existing calibration may no longer be valid.

## 7. Proposed delivery stages

### Stage 1 — MVP

- one ML task type: object detection,
- one primary deployment format, preferably ONNX,
- project creation, image import/capture, and manual annotation,
- training launched and monitored from the Trainer,
- basic metrics and performance benchmarks,
- deployment package export and import,
- Production recipe selection, preview, thresholds, and basic statistics,
- stored camera settings with verification that they were applied,
- an English default UI with an optional Polish translation,
- scale calibration, at least one linear measurement, and tolerance-based OK/NOK decisions,
- local user profiles and role enforcement sufficient for a single-station MVP, with a migration path to centralized authentication,
- a versioned server API contract and deployment-package repository prototype.

### Stage 2 — Data workflow improvements

- pre-labeling, annotation propagation, active learning, and dataset quality checks,
- experiment and model-version comparison,
- advanced statistics and reports,
- centralized user management and audit logging,
- production-ready AIVQC Server authentication and shared-resource synchronization,
- package approval, publication, revocation, local caching, and offline synchronization workflows.

### Stage 3 — Industrial integration

- PLC and digital I/O integration with reject-mechanism control,
- MES/API integration,
- multiple cameras and inspection stations,
- optional centralized deployment management,
- additional ML tasks such as segmentation, classification, and anomaly detection.

## 8. Primary risks

- lighting, product position, and optics can affect results more than model architecture,
- randomly splitting consecutive video frames can inflate metrics through data leakage,
- rare defects may not have enough examples for reliable evaluation,
- threshold adjustment cannot compensate for a poor or unrepresentative dataset,
- camera settings may not transfer between different camera models,
- measurements without valid calibration, fixed working distance, and controlled perspective can appear precise while being inaccurate,
- measurement accuracy must be verified with reference artifacts and is not equivalent to image resolution,
- laboratory results do not guarantee production-line quality or throughput,
- automated annotation requires human quality control,
- an undefined failure response could allow a product to pass without a valid inspection,
- a compromised account or overly broad role could allow unauthorized model, recipe, or threshold changes,
- server downtime, network partitions, or synchronization conflicts could leave stations on outdated packages,
- loss of encryption keys, backups, or server storage could make shared resources unavailable or unrecoverable.

## 9. MVP success criteria

- a new user can create a project, annotate data, train a model, and export a package without manually editing configuration files,
- AIVQC Production restores the model, classes, preprocessing, region of interest, thresholds, camera settings, measurement specification, tolerances, and calibration version,
- the application clearly reports a missing camera, incompatible model, or configuration that could not be applied,
- benchmarks produce repeatable quality and performance results,
- recipes and threshold changes are versioned or audited,
- Production operates offline for a complete shift,
- Operator, Technician, and Expert permissions are enforced and covered by authorization tests,
- passwords are never stored in plaintext and security-sensitive actions appear in the audit log,
- a package published by Trainer can be retrieved, verified, activated, and rolled back by an authorized Production station,
- the UI defaults to English and the complete primary workflow can be switched to Polish,
- a reference part is measured within deployment-specific uncertainty and an out-of-tolerance result produces NOK.

Target quality, maximum latency, FPS, and measurement uncertainty must be defined for each deployment.

## 10. Outside the MVP

- fully unattended annotation without human approval,
- simultaneous support for every ML framework and model family,
- cloud training and centralized fleet management,
- integration with every PLC and MES platform,
- formal certification for regulated industries,
- process decisions beyond the configured inspection result.

## 11. Open decisions

- first MVP product and defect classes,
- target operating system and minimum hardware,
- first supported models and training execution method,
- first supported cameras and communication library,
- required FPS, maximum latency, and inspection trigger method,
- continuous-stream inspection, trigger-based product inspection, or both,
- NOK rules for multiple classes and detections,
- false-accept and false-reject review workflow,
- image and statistics retention periods,
- deployment package format and versioning strategy,
- UI technology for both applications,
- required measurement types and target accuracy/uncertainty,
- calibration method and verification interval,
- decision rules for boundary values and invalid measurements,
- additional UI languages beyond English and Polish,
- server deployment model: on-premises, cloud, or both,
- identity model: server-managed accounts, Active Directory/LDAP, OpenID Connect, or a staged combination,
- password policy, multi-factor authentication, session duration, lockout, and account-recovery rules,
- permissions assigned to each role and whether custom roles are required,
- resource-storage technology, capacity limits, retention, backup location, and recovery objectives,
- behavior and maximum permitted offline duration after a user, station, recipe, or package is revoked,
- ownership and synchronization rules for datasets, which may be too large or sensitive for automatic replication.

## 12. Change log

- **2026-08-19 — version 0.1:** created the document, separated AIVQC Trainer and AIVQC Production, and defined the MVP, principal requirements, risks, and open decisions.
- **2026-08-20 — version 0.2:** added UI localization and calibrated product measurements with specifications, tolerances, and OK/NOK decisions.
- **2026-08-20 — version 0.3:** established English as the default language for all project files and the UI, with Polish as an optional UI translation.
- **2026-08-20 — version 0.4:** selected a license-neutral PyTorch/TorchVision training backend and implemented Pascal VOC training, automatic validation/test evaluation, progress reporting, cancellation, checkpointing, and ONNX export.
- **2026-08-21 — version 0.5:** added Operator, Technician, and Expert profiles plus the AIVQC Server for authentication, secure password handling, shared-resource storage, publication, synchronization, and offline Production operation.
- **2026-08-21 — version 0.6:** implemented the first Production ONNX pipeline for local images, including model-contract validation, class loading, preprocessing, CPU inference, threshold-based OK/NOK decisions, detection overlays, latency reporting, cancellation, and session statistics.
