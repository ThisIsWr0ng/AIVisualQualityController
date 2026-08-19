namespace Aivqc.Core.Deployment;

/// <summary>
/// Describes a versioned model package exported by Trainer and consumed by Production.
/// </summary>
public sealed record DeploymentPackageManifest(
    string SchemaVersion,
    Guid PackageId,
    string ProductId,
    string RecipeId,
    DateTimeOffset CreatedAtUtc,
    string CreatedBy,
    ModelManifest Model,
    PreprocessingManifest Preprocessing,
    RegionOfInterestManifest RegionOfInterest,
    IReadOnlyList<DefectClassManifest> DefectClasses)
{
    public const string CurrentSchemaVersion = "1.0";
}

public sealed record ModelManifest(
    string FileName,
    string Sha256,
    ModelTask Task,
    string Runtime,
    int InputWidth,
    int InputHeight);

public sealed record PreprocessingManifest(
    string ColorSpace,
    string Normalization,
    bool PreserveAspectRatio);

public sealed record RegionOfInterestManifest(
    int X,
    int Y,
    int Width,
    int Height);

public sealed record DefectClassManifest(
    int Id,
    string Name,
    float Threshold);

public enum ModelTask
{
    ObjectDetection
}
