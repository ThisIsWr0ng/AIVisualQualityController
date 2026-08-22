namespace Aivqc.Core.Projects;

/// <summary>
/// Describes a local Trainer workspace and its imported image assets.
/// </summary>
public sealed record TrainerProjectManifest(
    string SchemaVersion,
    Guid ProjectId,
    string Name,
    string ProductId,
    DateTimeOffset CreatedAtUtc,
    DateTimeOffset UpdatedAtUtc,
    IReadOnlyList<string> DefectClasses,
    IReadOnlyList<ProjectImageAsset> Images)
{
    public const string CurrentSchemaVersion = "1.0";
}

public sealed record ProjectImageAsset(
    Guid ImageId,
    string SourceFileName,
    ImageStorageMode StorageMode,
    string Location,
    string ThumbnailPath,
    string Sha256,
    int Width,
    int Height,
    string Format,
    DateTimeOffset ImportedAtUtc,
    IReadOnlyList<string> Warnings);

public enum ImageStorageMode
{
    Copy,
    Reference,
}
