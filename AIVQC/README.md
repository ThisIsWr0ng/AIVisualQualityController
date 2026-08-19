# AIVQC

The modernized AI Visual Quality Controller consists of two desktop applications:

- **Aivqc.Trainer** prepares datasets, trains and evaluates models, and exports deployment packages.
- **Aivqc.Production** imports deployment packages and runs inspections on production equipment.

Both applications share platform-independent contracts and domain logic from **Aivqc.Core**.

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
