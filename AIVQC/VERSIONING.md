# Versioning

AIVQC uses [Semantic Versioning](https://semver.org/) for the Trainer, Production, and Core binaries.
All projects read the product version from the single file [`Version.props`](Version.props).

The version has the form `MAJOR.MINOR.PATCH[-PRERELEASE]`:

- increment **MAJOR** for an incompatible application or public package-format change,
- increment **MINOR** for a backward-compatible feature,
- increment **PATCH** for a backward-compatible correction,
- use a suffix such as `alpha.2`, `beta.1`, or `rc.1` for a pre-release.

The current development version is `0.5.3-alpha.3`.

## Prepare a new version

1. Update `VersionPrefix` in `Version.props`.
2. Set `VersionSuffix` to the next pre-release label. Leave it empty for a stable release.
3. Update relevant release notes or the scope change log.
4. Build and test the complete solution:

   ```powershell
   dotnet build Aivqc.sln --configuration Release
   dotnet test Aivqc.sln --configuration Release --no-build
   ```

5. Commit the release changes.
6. Create an annotated Git tag matching the compiled version:

   ```powershell
   git tag -a v0.1.0-alpha.1 -m "AIVQC 0.1.0-alpha.1"
   git push origin v0.1.0-alpha.1
   ```

Do not reuse or move a published version tag. Prepare a new version instead.

## Version types are independent

- **Product version** identifies the AIVQC application binaries and is controlled by `Version.props`.
- **Deployment schema version** identifies the JSON/package structure and is controlled by `DeploymentPackageManifest.CurrentSchemaVersion`.
- **Model version** identifies a trained ML artifact.
- **Recipe version** identifies production settings, thresholds, camera configuration, calibration, and decision rules.

Changing a model or recipe does not automatically require a new application version. Changing the deployment schema may require a new application version and a migration path.
