namespace Aivqc.Server.Storage;

public sealed record ServerState(
    string SchemaVersion,
    IReadOnlyList<RegisteredStation> Stations,
    IReadOnlyList<StoredPackage> Packages,
    IReadOnlyList<PackageAssignment> Assignments)
{
    public const string CurrentSchemaVersion = "1.0";

    public static ServerState Empty { get; } = new(CurrentSchemaVersion, [], [], []);
}

public sealed record RegisteredStation(
    string StationId,
    string Name,
    DateTimeOffset RegisteredAtUtc,
    string RegisteredBy);

public sealed record StoredPackage(
    Guid PackageId,
    string ProductId,
    string RecipeId,
    DateTimeOffset CreatedAtUtc,
    DateTimeOffset PublishedAtUtc,
    string PublishedBy,
    string RelativePath,
    long SizeBytes,
    string Sha256,
    bool Revoked,
    DateTimeOffset? RevokedAtUtc,
    string? RevokedBy);

public sealed record PackageAssignment(
    Guid PackageId,
    string StationId,
    DateTimeOffset AssignedAtUtc,
    string AssignedBy,
    DateTimeOffset? DownloadedAtUtc,
    DateTimeOffset? ActivatedAtUtc,
    DateTimeOffset? FailedAtUtc,
    string? LastMessage);

public sealed record PublishPackageResult(StoredPackage Package, PackageAssignment Assignment);

public sealed record PackageAcknowledgement(
    Aivqc.Core.Connectivity.PackageAcknowledgementStatus Status,
    string? Message);
