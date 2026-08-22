# AIVQC — Project Scope and Assumptions

> This is a living project document. Update it as requirements, decisions, and test results evolve.
>
> **Document version:** 0.9.1 — first-release baseline draft

## 1. Vision

AI Visual Quality Controller (AIVQC) will be a universal, well-optimized, and easy-to-deploy visual quality inspection system for small and medium-sized manufacturing operations.

The system should detect different types of defects regardless of industry or product type. It should lower the machine-learning barrier: a user should be able to collect and annotate data, train and evaluate a model, and deploy it to production without manually transferring multiple configuration files.

AIVQC must be secure by design and secure by default. Introducing the system into a company must not create an avoidable path for unauthorized access, production disruption, credential theft, or leakage of product images, models, recipes, measurements, employee data, production statistics, or other confidential information. Security controls apply to Trainer, Production, Server, deployment packages, updates, development infrastructure, and their communication channels.

The project consists of two applications:

1. **AIVQC Trainer** — dataset preparation, training, testing, benchmarking, and deployment package export.
2. **AIVQC Production** — inspection execution on production equipment.

### 1.1. Scope version and product version

The version of this document is independent from the compiled AIVQC product version. **Scope 1.0** will mean that the requirements and boundaries of the first production-oriented release have been reviewed, resolved, and baselined. It does not claim that every requirement has already been implemented.

Scope 0.9 defines the structure required for that baseline. Items marked as open decisions must either be resolved before Scope 1.0 or explicitly deferred beyond the first product release. Later changes to a baselined requirement must be recorded in the change log and linked to a design decision or change request.

## 2. Target users

- process and quality engineers configuring inspections,
- technicians preparing cameras, datasets, and models,
- line operators running approved inspections,
- small and medium-sized manufacturers without an internal machine-learning team.

## 3. AIVQC Trainer

### 3.1. Purpose

Guide users through the complete workflow, from image collection and dataset preparation to a tested, production-ready deployment package.

The normal deployment topology uses one centrally managed Trainer workstation with sufficient CPU, GPU, memory, and storage for model training. It must receive inspection data from multiple Production lines and publish a model or deployment package to one selected line, a selected group of lines, or all compatible lines.

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
- receive line images and associated inspection metadata from multiple authorized Production stations for model testing, dataset improvement, and retraining,
- keep the origin of every received sample, including station, line, product, recipe, model version, result, and capture time,
- select target Production lines when publishing a trained model or deployment package,
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
- safely roll back to the last known working model,
- collect configurable samples from the production line, including NOK images, uncertain cases, and a controlled sample of OK images,
- upload selected images and inspection metadata for retrieval by Trainer without interrupting inspection,
- receive, verify, stage, and activate deployment packages assigned specifically to that Production line,
- expose a fail-safe inspection-health output that the production line can use to stop when detection is unavailable or no longer trustworthy.

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
- persistent storage limited to essential shared assets such as account and authorization data, station registry, package metadata, audit records, and currently approved or rollback-required deployment packages,
- temporary transfer storage for images, models, packages, and related metadata currently moving between Trainer and Production,
- automatic removal of successfully delivered temporary transfer files after acknowledgement and expiry of a configurable recovery window,
- controlled publication from Trainer and controlled retrieval by Production,
- immutable identifiers, version history, checksums, and integrity verification for distributed packages,
- approval states such as Draft, Validated, Approved, Published, Retired, and Revoked,
- audit logging for authentication, downloads, publication, activation, rollback, configuration changes, and administrative actions,
- encrypted network communication and authenticated API access,
- backup, restore, retention, storage-quota, and disaster-recovery policies,
- compatibility checks between server, package, Trainer, and Production versions,
- synchronization status, conflict handling, retry behavior, and clear error reporting.

The server is a secure coordination and transfer service, not the primary archive for full datasets or all historical model artifacts. Trainer remains the authoritative long-term store for datasets, experiments, and training artifacts. Each Production station remains the local source of inspection records according to its retention policy. Server-side retention must be defined per resource type and must prevent unbounded image or model accumulation.

Every Production station must have a unique identity and line assignment. Transfers must be explicitly addressed and resumable: Production sends selected line images toward Trainer, while Trainer publishes signed, versioned deployment packages only to selected compatible Production stations. The server must track Queued, Transferring, Delivered, Acknowledged, Failed, and Expired states without treating upload completion as successful activation.

Production must keep an authenticated, integrity-checked local cache of approved recipes and packages. A temporary server or network outage must not stop an already authorized production inspection. Offline actions and statistics must be queued and synchronized after connectivity returns. Revoked packages must be blocked as soon as the station receives the revocation, subject to a separately defined emergency/offline policy.

Trainer and Production must access shared resources through documented server APIs; they must not directly access the server database or shared storage directories.

### 6.3. Inspection health output and line interlock

Production must provide an external status output suitable for integration with a PLC or line controller. The output must indicate that the complete inspection path is healthy, including the application process, camera stream, active recipe, model inference, result pipeline, and required communication with local I/O.

The interface should support a fail-safe heartbeat or watchdog rather than a permanently asserted software flag. Loss of power, an application crash, a frozen process, camera failure, invalid model, excessive inference delay, or stale results must cause the external controller to observe an unhealthy state. The production-line safety or control system, not AIVQC alone, owns the final line-stop action.

The required electrical interface, heartbeat frequency, maximum detection gap, timeout, startup behavior, and restart acknowledgement must be configurable and validated for each line. This operational interlock is not a substitute for a certified functional-safety system where one is legally or technically required.

### 6.4. Automatic updates

Trainer, Production, and Server should support controlled automatic updates with:

- cryptographically signed update packages and integrity verification before installation,
- separate release channels and compatibility rules for each application,
- staged rollout to selected lines before wider deployment,
- configurable maintenance windows and an explicit policy preventing unsafe updates during an active inspection,
- download retry and resume support,
- pre-installation checks, migration validation, health checks, and audit records,
- automatic rollback to the last working application version after a failed update,
- an authorized option to postpone, approve, or block an update,
- offline update packages for isolated production networks.

Application updates and model deployment are separate processes. Updating a model must not silently update application binaries, and updating an application must not silently activate a different recipe or model.

### 6.5. General requirements

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

### 6.6. Inspection lifecycle and decision handling

Each inspection must follow an explicit state model:

`Ready → Triggered → Capturing → Validating → Inferring → Measuring → Deciding → Reporting → Completed`

An inspection may enter `Indeterminate` or `Error` from any processing state. The recipe must define whether those results stop the line, reject the product, request a retry, or require operator acknowledgement. The system must define deterministic behavior for duplicate triggers, missing products, multiple products, invalid or stale images, camera timeouts, inference timeouts, duplicate result delivery, and restart during an active inspection.

Every external trigger and result must be correlated by an inspection identifier. Retrying communication must not create a second logical inspection or apply the same reject action twice.

### 6.7. Traceability and time

Every inspection record must include, where available:

- a globally unique inspection identifier,
- station and production-line identifiers,
- product, batch, and optional serial number,
- trigger, capture, decision, and reporting timestamps,
- active model, package, recipe, specification, calibration, and application versions,
- result, detected defects, measurements, thresholds, and confidence values,
- user identity for manual actions or overrides,
- image references and retention status,
- communication and health-output status.

Trainer, Server, and Production clocks must use a documented synchronization method. Stored timestamps must be unambiguous, include UTC or an offset, and remain interpretable across daylight-saving changes.

### 6.8. Model and recipe lifecycle

Models and deployment packages follow the controlled lifecycle:

`Draft → Trained → Validated → Approved → Published → Active → Retired/Revoked`

Only an authorized Expert may approve or publish a package. Validation criteria must be tied to a fixed dataset version and include both quality and target-hardware performance results. Activation on Production is a separate, audited event and must never be implied by upload or download completion. Rollback must restore the complete compatible recipe and package, not only the model file.

Production feedback must support human review and correction before it becomes training ground truth. The system should track model drift indicators, changes in defect distribution, uncertain detections, and confirmed false accepts or false rejects.

### 6.9. Data ownership and retention

- Trainer is the authoritative long-term store for datasets, annotations, experiments, and training artifacts.
- Production is the authoritative source for local inspection records until their successful transfer or expiry under the station retention policy.
- Server is authoritative for identities, authorization, station registry, routing state, audit records, and published-package metadata; it is not the permanent archive for full datasets.
- Transfer files on Server are temporary and must have quotas, expiry, acknowledgement, retry, and cleanup rules.

Every resource type must have a defined owner, retention period, backup requirement, maximum size, encryption requirement, and deletion process. Disk-full behavior must protect active inspection and must never silently discard required inspection results.

### 6.10. Cybersecurity, data protection, and operational resilience

Cybersecurity is a release requirement, not an optional hardening activity. AIVQC will follow a risk-based secure-development process informed by NIST SSDF and use OWASP ASVS for Server/API verification and OWASP TCASVS for the desktop applications. Security requirements and tests must be traceable to the applicable component and threat.

The first implementation priorities, in order, are:

1. **Threat model and data inventory** — document protected data, trust boundaries, entry points, actors, data flows, plausible misuse, and consequences for Trainer, Production, Server, update infrastructure, and industrial interfaces.
2. **Minimal exposure and network segmentation** — default to an on-premises deployment, bind services only to required interfaces, deny unused inbound access, expose only documented API endpoints, and support firewall rules that isolate Production networks from office and public networks.
3. **Unique identity and least privilege** — eliminate default or shared credentials; give every user, station, and service a unique identity; enforce authorization on Server/API operations; require MFA for Expert and administrative access when technically available.
4. **Protected credentials** — hash passwords with a salted, adaptive algorithm such as Argon2id, store application secrets and private keys outside source code and ordinary configuration files, rate-limit authentication attempts, and provide secure revocation and recovery.
5. **Authenticated encryption** — use current TLS for every network connection and mutually authenticate machines where practical. Reject invalid, expired, untrusted, or revoked certificates instead of silently downgrading security.
6. **Signed and validated artifacts** — cryptographically sign application updates and deployment packages; verify signer, integrity, schema, compatibility, file sizes, paths, and contents before extraction or activation. Untrusted models, archives, images, and metadata must be treated as hostile input.
7. **Data minimization and leakage prevention** — collect, transfer, and retain only required information; use explicit transfer destinations and allowlists; encrypt sensitive data at rest where exposure risk requires it; redact secrets and sensitive business data from logs; disable external telemetry and cloud upload by default unless explicitly configured and documented.
8. **Audit and detection** — record successful and failed authentication, authorization failures, administrative changes, data export, package publication/activation, certificate events, and security configuration changes. Protect logs against unauthorized deletion or modification and alert on suspicious repeated failures.
9. **Software supply-chain security** — pin and review dependencies, scan source and dependencies for known vulnerabilities and leaked secrets, produce an SBOM for releases, protect build and signing credentials, and document licenses and provenance of dependencies, models, and training assets.
10. **Recovery and response** — maintain tested backups, signed offline installers, rollback procedures, credential and certificate rotation, vulnerability intake, security-update procedures, and an incident-response process that can isolate a station without losing required production evidence.

Security must fail closed for authentication, authorization, package trust, and administrative operations. Inspection availability and line safety require an explicitly documented fail-safe policy so that a security failure cannot silently convert an invalid inspection into OK.

Before the first production release, the architecture must define:

- unique machine identity and credentials for every Trainer, Server, and Production installation,
- transport encryption, certificate validation, secret storage, and key rotation,
- update-signing and package-signing key ownership, backup, rotation, and revocation,
- password, session, lockout, recovery, and optional multi-factor authentication policies,
- rate limiting and protection from repeated authentication attempts,
- security event logging without passwords, tokens, private keys, or sensitive image data,
- backup and restore procedures with tested recovery-point and recovery-time objectives,
- log rotation, disk-space monitoring, diagnostics export, and health monitoring,
- a process for reporting, assessing, and correcting security vulnerabilities.

The following data-protection rules apply by default:

- no Internet-facing Server or remotely accessible Production station unless explicitly designed, reviewed, and approved for that deployment,
- no anonymous API access and no permanent factory-default passwords,
- no model, image, dataset, recipe, statistic, or diagnostic upload to third parties without an authorized configuration and visible destination,
- no secrets, access tokens, passwords, private keys, complete connection strings, or sensitive production images in logs or crash reports,
- no direct database or file-share access from Trainer or Production,
- no activation of unsigned, invalid, revoked, or incorrectly targeted packages,
- no unsupported silent fallback from encrypted to unencrypted communication,
- no security control that depends only on hiding a button or trusting client-provided role information.

### 6.11. Verification and release qualification

Every release intended for production must pass a documented qualification suite covering:

- the complete Trainer-to-Server-to-Production model workflow,
- a continuous run representative of at least one production shift,
- camera loss, malformed frames, stale frames, and reconnection,
- server and network loss during inspection and file transfer,
- corrupted, incompatible, incorrectly routed, and revoked packages,
- process crash, power interruption, restart, and local-cache recovery,
- authentication and authorization for Operator, Technician, and Expert,
- network exposure, firewall rules, certificate validation, machine identity, and attempted unauthorized API access,
- password hashing parameters, authentication rate limits, MFA for privileged access, session expiry, and account revocation,
- malicious or malformed images, archives, metadata, models, package paths, and oversized inputs,
- dependency, secret, static-code, and known-vulnerability scanning plus release SBOM generation,
- verification that logs, diagnostics, crash reports, and telemetry do not leak protected data or credentials,
- bounded storage, queue saturation, and disk-full behavior,
- inspection-health heartbeat and external-controller timeout,
- calibration validity and measurement checks against reference artifacts,
- throughput, latency, resource consumption, and result repeatability on target hardware,
- backup restoration and rollback to the last working application and model package.

The test evidence, configuration, hardware, datasets, expected results, and pass/fail outcome must be retained with the release record.

### 6.12. Quantitative deployment targets

Scope 1.0 must define a default first-release target or an explicit per-deployment acceptance method for each of the following:

| Area | Required target |
| --- | --- |
| Inspection | trigger-to-result latency, inference latency, minimum FPS/throughput, timeout, and dropped-frame limit |
| Reliability | permitted downtime, restart time, offline duration, and continuous-run duration |
| Model quality | per-class precision/recall or false-accept/false-reject limits and minimum validation sample size |
| Measurement | supported units, range, resolution, accuracy/uncertainty, and calibration verification interval |
| Health output | heartbeat frequency, unhealthy timeout, startup state, polarity, and acknowledgement behavior |
| Server scale | supported Production stations, concurrent transfers, queue capacity, and maximum file size |
| Storage | local quotas, minimum free-space reserve, retention periods, and cleanup thresholds |
| Recovery | backup frequency, recovery-point objective, recovery-time objective, and rollback duration |
| Hardware | minimum and recommended Trainer and Production CPU, GPU, RAM, storage, and network |

Terms such as fast, reliable, real-time, lightweight, or accurate are not acceptance criteria unless accompanied by a measurable target and test method.

## 7. Release boundaries

### 7.1. First product release

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
- an on-premises AIVQC Server with a versioned API, centralized authentication, audit records, bounded transfer storage, and package routing,
- server-managed user accounts and the package approval, publication, revocation, local caching, and offline synchronization workflow,
- one Trainer able to register multiple Production stations and route a package to a selected station,
- asynchronous upload of selected Production samples to Trainer through bounded temporary server storage,
- one supported physical or industrial communication adapter for the inspection-health heartbeat and fail-safe timeout behavior.

### 7.2. Planned after the first release

- pre-labeling, annotation propagation, active learning, and dataset quality checks,
- experiment and model-version comparison,
- advanced statistics and reports,
- optional Active Directory, LDAP, or OpenID Connect integration and advanced identity policies,
- customizable permission policies beyond the three base roles,
- signed automatic updates, staged rollout, compatibility checks, and rollback,
- multi-line dashboards, transfer queues, routing, and per-line deployment status.

### 7.3. Later industrial expansion

- additional PLC and industrial I/O adapters with reject-mechanism control,
- MES/API integration,
- multiple cameras per line and advanced station layouts,
- cloud or cross-site fleet management,
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
- loss of encryption keys, backups, or server storage could make shared resources unavailable or unrecoverable,
- uncontrolled image transfer could consume line bandwidth, server storage, or Trainer capacity,
- an incorrectly routed model could be activated on the wrong product or production line,
- a software-only status signal could remain stale after a process hang unless an external watchdog verifies a changing heartbeat,
- an interrupted or incompatible automatic update could disable inspection without transactional installation and rollback.

## 9. First-release acceptance criteria

- a new user can create a project, annotate data, train a model, and export a package without manually editing configuration files,
- AIVQC Production restores the model, classes, preprocessing, region of interest, thresholds, camera settings, measurement specification, tolerances, and calibration version,
- the application clearly reports a missing camera, incompatible model, or configuration that could not be applied,
- benchmarks produce repeatable quality and performance results,
- recipes and threshold changes are versioned or audited,
- Production operates offline for a complete shift,
- Operator, Technician, and Expert permissions are enforced and covered by authorization tests,
- passwords are never stored in plaintext and security-sensitive actions appear in the audit log,
- a package published by Trainer can be retrieved, verified, activated, and rolled back by an authorized Production station,
- one Trainer can receive traceable samples from at least two simulated Production lines and deploy a package only to the selected target line,
- acknowledged transfer files expire from server temporary storage while required metadata and audit records remain,
- loss of camera frames, inference progress, or the Production process changes the external inspection-health signal within the configured timeout,
- the UI defaults to English and the complete primary workflow can be switched to Polish,
- a reference part is measured within deployment-specific uncertainty and an out-of-tolerance result produces NOK.

Target quality, maximum latency, FPS, and measurement uncertainty must be defined for each deployment.

## 10. Outside the first product release

- fully unattended annotation without human approval,
- simultaneous support for every ML framework and model family,
- cloud-hosted training and cross-site fleet management,
- integration with every PLC and MES platform,
- formal certification for regulated industries,
- process decisions beyond the configured inspection result.

## 11. Open decisions

Every item below must have an owner and target decision date. Before Scope 1.0, each item must be either resolved in this document, recorded in an approved architecture decision, or explicitly deferred beyond the first product release.

- first MVP product and defect classes,
- target operating system and minimum hardware,
- additional model families beyond the initial PyTorch/TorchVision-to-ONNX workflow,
- first supported cameras and communication library,
- required FPS, maximum latency, and inspection trigger method,
- continuous-stream inspection, trigger-based product inspection, or both,
- NOK rules for multiple classes and detections,
- false-accept and false-reject review workflow,
- image and statistics retention periods,
- compatibility and migration policy for future `.aivqcpkg` schema versions,
- required measurement types and target accuracy/uncertainty,
- calibration method and verification interval,
- decision rules for boundary values and invalid measurements,
- additional UI languages beyond English and Polish,
- post-first-release support for cloud or cross-site server deployment,
- timing and scope of optional Active Directory/LDAP/OpenID Connect integration,
- password policy, multi-factor authentication, session duration, lockout, and account-recovery rules,
- permissions assigned to each role and whether custom roles are required,
- resource-storage technology, capacity limits, retention, backup location, and recovery objectives,
- behavior and maximum permitted offline duration after a user, station, recipe, or package is revoked,
- ownership and synchronization rules for datasets, which may be too large or sensitive for automatic replication,
- exact definition of essential server files and retention time for each temporary transfer type,
- sample-selection rules, image anonymization requirements, bandwidth limits, and transfer schedules for each line,
- authoritative ownership and conflict rules when Trainer, Server, and Production hold different versions,
- line-grouping and compatibility rules used when targeting model deployments,
- external status interface: digital I/O, PLC protocol, OPC UA, industrial Ethernet, or another adapter,
- fail-safe polarity, heartbeat frequency, detection timeout, and required line-controller acknowledgement,
- update hosting, signing-key ownership, release channels, maintenance windows, and mandatory-update policy.
- data-classification levels and which images, models, recipes, statistics, logs, and user attributes belong to each level,
- encryption-at-rest requirements and key ownership for Trainer, Production, Server, backups, and temporary transfer storage,
- approved network topology, firewall ports, remote-support method, and whether any component may access the public Internet,
- security baseline and verification level for Server/API and desktop applications,
- vulnerability-response targets based on severity and the supported lifetime of each product release.

The following decisions block a credible Scope 1.0 baseline and have the highest priority:

1. first product, defect set, and representative validation dataset,
2. inspection trigger and result/health interface,
3. supported camera and minimum Production hardware,
4. measurable throughput, latency, model-quality, and measurement targets,
5. first-release Server topology and identity model,
6. retention, offline, failure, and line-stop policies,
7. exact contents of the first product release versus deferred capabilities.

## 12. Change log

- **2026-08-19 — version 0.1:** created the document, separated AIVQC Trainer and AIVQC Production, and defined the MVP, principal requirements, risks, and open decisions.
- **2026-08-20 — version 0.2:** added UI localization and calibrated product measurements with specifications, tolerances, and OK/NOK decisions.
- **2026-08-20 — version 0.3:** established English as the default language for all project files and the UI, with Polish as an optional UI translation.
- **2026-08-20 — version 0.4:** selected a license-neutral PyTorch/TorchVision training backend and implemented Pascal VOC training, automatic validation/test evaluation, progress reporting, cancellation, checkpointing, and ONNX export.
- **2026-08-21 — version 0.5:** added Operator, Technician, and Expert profiles plus the AIVQC Server for authentication, secure password handling, shared-resource storage, publication, synchronization, and offline Production operation.
- **2026-08-21 — version 0.6:** implemented the first Production ONNX pipeline for local images, including model-contract validation, class loading, preprocessing, CPU inference, threshold-based OK/NOK decisions, detection overlays, latency reporting, cancellation, and session statistics.
- **2026-08-22 — version 0.7:** implemented versioned `.aivqcpkg` export in Trainer and integrity-checked import in Production, including strict archive contents, schema validation, SHA-256 model verification, controlled local caching, recipe metadata, class names, and locked per-class thresholds.
- **2026-08-22 — version 0.8:** implemented persistent local Trainer projects with atomic JSON autosave, recent-project history, missing-source diagnostics, copy/reference image import, actual-format validation, SHA-256 deduplication, quality warnings, thumbnail generation, and dataset browsing.
- **2026-08-22 — version 0.8:** defined centralized multi-line Trainer topology, bidirectional Production sample and model-package transfer, minimal server retention, fail-safe inspection-health output, and controlled automatic application updates.
- **2026-08-22 — version 0.9:** restructured the scope into a first-release baseline, separated document and product versions, resolved delivery-stage contradictions, and added inspection lifecycle, traceability, model lifecycle, data ownership, security, operational resilience, release qualification, and quantitative target requirements.
- **2026-08-22 — version 0.9.1:** established the first explicit cybersecurity baseline, prioritizing threat modeling, network minimization, unique identities, least privilege, protected credentials, authenticated encryption, signed artifacts, data-loss prevention, audit logging, supply-chain security, and incident recovery.
