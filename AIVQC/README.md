# AIVQC

AI Visual Quality Controller is a visual inspection system intended for small and medium-sized manufacturing operations. It detects product defects and verifies product dimensions against a versioned inspection recipe.

The modernized system consists of two desktop applications:

- **Aivqc.Trainer** — dataset creation, camera and measurement configuration, model training, evaluation, benchmarking, and deployment package export.
- **Aivqc.Production** — production inspection, defect detection, product measurement, specification checks, statistics, and OK/NOK decisions.

Both applications share platform-independent contracts and domain logic from **Aivqc.Core**. English is the default language for the software, source code, configuration, documentation, and technical identifiers. Polish is provided as an optional user-interface translation.

See [Scope.md](Scope.md) for the product scope, MVP requirements, risks, and open decisions.
See [VERSIONING.md](VERSIONING.md) for the product release and Git tagging procedure.
See [training/README.md](training/README.md) for dataset, Python and hardware setup.

## Author

Copyright © 2026 **Dawid Oleśko**

Contact: [oleskodawid@gmail.com](mailto:oleskodawid@gmail.com)

## License

AIVQC is source-available under the [PolyForm Noncommercial License 1.0.0](LICENSE.md). Noncommercial use, modification, and distribution are permitted under its terms.

Commercial use is not granted by that license. This includes using AIVQC in a commercial manufacturing operation, integrating it into a commercial product, or providing paid products or services based on it. A separate written commercial license is required. Contact [oleskodawid@gmail.com](mailto:oleskodawid@gmail.com) for commercial licensing.

## Key capabilities

- create and annotate training datasets,
- import and validate existing ONNX models,
- train and evaluate an SSDLite320 object detector from Pascal VOC annotations,
- select, train, compare, and benchmark models,
- store camera settings with an inspection recipe,
- detect multiple defect classes with per-class thresholds,
- perform calibrated product measurements and tolerance checks,
- collect inspection statistics and audit configuration changes,
- export versioned deployment packages,
- run production inspections without mandatory Internet access,
- switch the user interface between English and Polish.

## Repository layout

```text
AIVQC/
├── src/
│   ├── Aivqc.Core/
│   ├── Aivqc.Trainer/
│   └── Aivqc.Production/
├── tests/
│   └── Aivqc.Core.Tests/
├── Model/                  # Legacy models retained for migration tests
├── Scope.md                # Product scope and MVP requirements
└── Aivqc.sln
```

The recovered legacy WinForms application remains outside this directory in `../Recovery` and is used only as a behavioral reference.

## Requirements

- a .NET SDK compatible with the projects in `Aivqc.sln`,
- Windows for the desktop applications,
- a camera supported by the selected device adapter.

## Build and test

```powershell
dotnet restore Aivqc.sln
dotnet build Aivqc.sln
dotnet test Aivqc.sln
```

Run either application with:

```powershell
dotnet run --project src/Aivqc.Trainer
dotnet run --project src/Aivqc.Production
```

## Product version

Trainer, Production, Core, and the test project share one semantic version from
[`Version.props`](Version.props). The applications display this compiled version in their footer.
The current development version is `0.5.3-alpha.3`.

## Status

The project is under active development. Some capabilities described in `Scope.md` are product requirements and may not be implemented yet. Legacy scripts, models, and prototypes in the repository root are retained as migration material.
