using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Aivqc.Core.Connectivity;
using Aivqc.Core.Deployment;
using Microsoft.Extensions.Options;

namespace Aivqc.Server.Storage;

/// <summary>
/// Persists package-routing metadata atomically and keeps package files in bounded local storage.
/// </summary>
public sealed class ServerPackageStore
{
    private const string StateFileName = "server-state.json";
    private const string AuditFileName = "audit.jsonl";
    private readonly ServerOptions _options;
    private readonly string _root;
    private readonly string _packagesRoot;
    private readonly string _verificationRoot;
    private readonly SemaphoreSlim _writeLock = new(1, 1);
    private readonly JsonSerializerOptions _jsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public ServerPackageStore(IOptions<ServerOptions> options)
    {
        _options = options.Value;
        _options.Validate();
        _root = Path.GetFullPath(_options.DataDirectory);
        _packagesRoot = Path.Combine(_root, "packages");
        _verificationRoot = Path.Combine(_root, "verification-cache");
        Directory.CreateDirectory(_root);
        Directory.CreateDirectory(_packagesRoot);
        Directory.CreateDirectory(_verificationRoot);

        var statePath = Path.Combine(_root, StateFileName);
        if (!File.Exists(statePath))
        {
            SaveState(ServerState.Empty);
        }

        _ = LoadState();
    }

    public ServerState GetState() => LoadState();

    public async Task<RegisteredStation> RegisterStationAsync(
        string stationId,
        string name,
        string actor,
        CancellationToken cancellationToken)
    {
        ValidateIdentifier(stationId, "station ID");
        ValidateDisplayText(name, "station name");

        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            var state = LoadState();
            if (state.Stations.Any(station => string.Equals(
                station.StationId,
                stationId,
                StringComparison.OrdinalIgnoreCase)))
            {
                throw new InvalidOperationException($"Station '{stationId}' is already registered.");
            }

            var station = new RegisteredStation(
                stationId.Trim(),
                name.Trim(),
                DateTimeOffset.UtcNow,
                actor);
            SaveState(state with { Stations = state.Stations.Append(station).ToArray() });
            AppendAudit(actor, "station.registered", station.StationId, null);
            return station;
        }
        finally
        {
            _writeLock.Release();
        }
    }

    public async Task<PublishPackageResult> PublishAsync(
        string uploadedPath,
        string targetStationId,
        string actor,
        CancellationToken cancellationToken)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(uploadedPath);
        ValidateIdentifier(targetStationId, "station ID");

        var file = new FileInfo(uploadedPath);
        if (!file.Exists || file.Length == 0 || file.Length > _options.MaximumPackageBytes)
        {
            throw new InvalidDataException(
                $"The package is empty or exceeds the {_options.MaximumPackageBytes} byte limit.");
        }

        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            var state = LoadState();
            if (!state.Stations.Any(station => string.Equals(
                station.StationId,
                targetStationId,
                StringComparison.OrdinalIgnoreCase)))
            {
                throw new KeyNotFoundException($"Station '{targetStationId}' is not registered.");
            }

            var occupiedBytes = Directory.EnumerateFiles(_root, "*", SearchOption.AllDirectories)
                .Sum(path => new FileInfo(path).Length);
            var requiredWorkingBytes = checked(file.Length * 2);
            if (occupiedBytes + requiredWorkingBytes > _options.StorageQuotaBytes)
            {
                throw new IOException("AIVQC Server package storage quota would be exceeded.");
            }

            var imported = DeploymentPackageArchive.Import(
                uploadedPath,
                _verificationRoot,
                _options.MaximumPackageBytes);
            if (state.Packages.Any(package => package.PackageId == imported.Manifest.PackageId))
            {
                throw new InvalidOperationException(
                    $"Package '{imported.Manifest.PackageId}' has already been published.");
            }

            var packageDirectory = Path.Combine(_packagesRoot, imported.Manifest.PackageId.ToString("N"));
            Directory.CreateDirectory(packageDirectory);
            var packagePath = Path.Combine(packageDirectory, "package.aivqcpkg");
            File.Copy(uploadedPath, packagePath, overwrite: false);
            var relativePath = Path.GetRelativePath(_root, packagePath).Replace('\\', '/');
            var publishedAt = DateTimeOffset.UtcNow;
            var storedPackage = new StoredPackage(
                imported.Manifest.PackageId,
                imported.Manifest.ProductId,
                imported.Manifest.RecipeId,
                imported.Manifest.CreatedAtUtc,
                publishedAt,
                actor,
                relativePath,
                file.Length,
                CalculateSha256(packagePath),
                false,
                null,
                null);
            var assignment = new PackageAssignment(
                storedPackage.PackageId,
                targetStationId.Trim(),
                publishedAt,
                actor,
                null,
                null,
                null,
                null);

            try
            {
                SaveState(state with
                {
                    Packages = state.Packages.Append(storedPackage).ToArray(),
                    Assignments = state.Assignments.Append(assignment).ToArray(),
                });
            }
            catch
            {
                File.Delete(packagePath);
                Directory.Delete(packageDirectory);
                throw;
            }

            AppendAudit(actor, "package.published", storedPackage.PackageId.ToString("D"), new
            {
                targetStationId,
                storedPackage.ProductId,
                storedPackage.RecipeId,
            });
            return new PublishPackageResult(storedPackage, assignment);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    public (StoredPackage Package, PackageAssignment Assignment)? GetLatest(string stationId)
    {
        ValidateIdentifier(stationId, "station ID");
        var state = LoadState();
        var assignment = state.Assignments
            .Where(item => string.Equals(item.StationId, stationId, StringComparison.OrdinalIgnoreCase))
            .OrderByDescending(item => item.AssignedAtUtc)
            .FirstOrDefault(item => state.Packages.Any(package =>
                package.PackageId == item.PackageId && !package.Revoked));
        if (assignment is null)
        {
            return null;
        }

        var package = state.Packages.Single(item => item.PackageId == assignment.PackageId);
        return (package, assignment);
    }

    public string GetPackageContentPath(string stationId, Guid packageId)
    {
        var state = LoadState();
        var package = state.Packages.SingleOrDefault(item => item.PackageId == packageId)
            ?? throw new KeyNotFoundException($"Package '{packageId}' does not exist.");
        if (package.Revoked)
        {
            throw new InvalidOperationException("The requested package has been revoked.");
        }

        if (!state.Assignments.Any(assignment =>
            assignment.PackageId == packageId
            && string.Equals(assignment.StationId, stationId, StringComparison.OrdinalIgnoreCase)))
        {
            throw new UnauthorizedAccessException("The package is not assigned to this station.");
        }

        var path = ResolvePackagePath(package);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException("The package transfer file has expired.", path);
        }

        return path;
    }

    public async Task AcknowledgeAsync(
        string stationId,
        Guid packageId,
        PackageAcknowledgement acknowledgement,
        string actor,
        CancellationToken cancellationToken)
    {
        if (!Enum.IsDefined(acknowledgement.Status)
            || acknowledgement.Message?.Length > 512
            || acknowledgement.Message?.Any(char.IsControl) == true)
        {
            throw new InvalidDataException("The package acknowledgement is invalid.");
        }

        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            var state = LoadState();
            var index = state.Assignments.ToList().FindIndex(assignment =>
                assignment.PackageId == packageId
                && string.Equals(assignment.StationId, stationId, StringComparison.OrdinalIgnoreCase));
            if (index < 0)
            {
                throw new KeyNotFoundException("The package assignment does not exist.");
            }

            var now = DateTimeOffset.UtcNow;
            var existing = state.Assignments[index];
            var updated = acknowledgement.Status switch
            {
                PackageAcknowledgementStatus.Downloaded => existing with { DownloadedAtUtc = now },
                PackageAcknowledgementStatus.Activated => existing with
                {
                    DownloadedAtUtc = existing.DownloadedAtUtc ?? now,
                    ActivatedAtUtc = now,
                },
                PackageAcknowledgementStatus.Failed => existing with { FailedAtUtc = now },
                _ => throw new InvalidDataException("Unsupported acknowledgement status."),
            };
            updated = updated with { LastMessage = acknowledgement.Message?.Trim() };
            var assignments = state.Assignments.ToArray();
            assignments[index] = updated;
            SaveState(state with { Assignments = assignments });
            AppendAudit(actor, $"package.{acknowledgement.Status.ToString().ToLowerInvariant()}", packageId.ToString("D"), new
            {
                stationId,
                acknowledgement.Message,
            });
        }
        finally
        {
            _writeLock.Release();
        }
    }

    public async Task RevokeAsync(Guid packageId, string actor, CancellationToken cancellationToken)
    {
        await _writeLock.WaitAsync(cancellationToken);
        try
        {
            var state = LoadState();
            var index = state.Packages.ToList().FindIndex(package => package.PackageId == packageId);
            if (index < 0)
            {
                throw new KeyNotFoundException($"Package '{packageId}' does not exist.");
            }

            var packages = state.Packages.ToArray();
            packages[index] = packages[index] with
            {
                Revoked = true,
                RevokedAtUtc = DateTimeOffset.UtcNow,
                RevokedBy = actor,
            };
            SaveState(state with { Packages = packages });
            AppendAudit(actor, "package.revoked", packageId.ToString("D"), null);
        }
        finally
        {
            _writeLock.Release();
        }
    }

    private ServerState LoadState()
    {
        var path = Path.Combine(_root, StateFileName);
        ServerState state;
        try
        {
            state = JsonSerializer.Deserialize<ServerState>(File.ReadAllText(path), _jsonOptions)
                ?? throw new InvalidDataException("The AIVQC Server state is empty.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The AIVQC Server state contains invalid JSON.", exception);
        }

        if (state.SchemaVersion != ServerState.CurrentSchemaVersion
            || state.Stations is null
            || state.Packages is null
            || state.Assignments is null)
        {
            throw new InvalidDataException("The AIVQC Server state schema is invalid.");
        }

        return state;
    }

    private void SaveState(ServerState state)
    {
        var destination = Path.Combine(_root, StateFileName);
        var temporary = Path.Combine(_root, $".{StateFileName}.{Guid.NewGuid():N}.tmp");
        try
        {
            File.WriteAllText(
                temporary,
                JsonSerializer.Serialize(state, _jsonOptions),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.Move(temporary, destination, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporary))
            {
                File.Delete(temporary);
            }
        }
    }

    private string ResolvePackagePath(StoredPackage package)
    {
        if (Path.IsPathFullyQualified(package.RelativePath))
        {
            throw new InvalidDataException("Stored package paths must be relative.");
        }

        var path = Path.GetFullPath(Path.Combine(_root, package.RelativePath));
        if (!path.StartsWith(_root + Path.DirectorySeparatorChar, StringComparison.Ordinal))
        {
            throw new InvalidDataException("A stored package path leaves the server data directory.");
        }

        return path;
    }

    private void AppendAudit(string actor, string action, string resource, object? details)
    {
        var entry = JsonSerializer.Serialize(new
        {
            timestampUtc = DateTimeOffset.UtcNow,
            actor,
            action,
            resource,
            details,
        });
        File.AppendAllText(Path.Combine(_root, AuditFileName), entry + Environment.NewLine, Encoding.UTF8);
    }

    private static string CalculateSha256(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }

    private static void ValidateIdentifier(string value, string description)
    {
        if (string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.Any(character =>
                !(char.IsAsciiLetterOrDigit(character) || character is '-' or '_' or '.')))
        {
            throw new InvalidDataException($"The {description} is invalid.");
        }
    }

    private static void ValidateDisplayText(string value, string description)
    {
        if (string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.Any(char.IsControl))
        {
            throw new InvalidDataException($"The {description} is invalid.");
        }
    }
}
