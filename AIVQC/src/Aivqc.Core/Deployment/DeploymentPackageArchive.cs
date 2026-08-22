using System.IO.Compression;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace Aivqc.Core.Deployment;

/// <summary>
/// Creates and verifies self-contained AIVQC deployment-package archives.
/// </summary>
public static class DeploymentPackageArchive
{
    public const string FileExtension = ".aivqcpkg";
    public const string ManifestEntryName = "manifest.json";
    public const string ModelEntryName = "model.onnx";

    private const long MaximumManifestBytes = 1024 * 1024;

    public static DeploymentPackageExportResult Export(DeploymentPackageExportRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        ValidateExportRequest(request);

        var outputPath = Path.GetFullPath(request.OutputPath);
        var outputDirectory = Path.GetDirectoryName(outputPath)
            ?? throw new InvalidOperationException("The package output directory could not be resolved.");
        Directory.CreateDirectory(outputDirectory);

        if (File.Exists(outputPath))
        {
            throw new IOException($"A file already exists at {outputPath}.");
        }

        var manifest = new DeploymentPackageManifest(
            DeploymentPackageManifest.CurrentSchemaVersion,
            Guid.NewGuid(),
            request.ProductId.Trim(),
            request.RecipeId.Trim(),
            DateTimeOffset.UtcNow,
            request.CreatedBy.Trim(),
            new ModelManifest(
                ModelEntryName,
                CalculateSha256(request.ModelPath),
                ModelTask.ObjectDetection,
                "onnxruntime",
                request.InputWidth,
                request.InputHeight),
            request.Preprocessing,
            request.RegionOfInterest,
            request.DefectClasses.ToArray());
        ValidateManifest(manifest);

        var temporaryPath = Path.Combine(
            outputDirectory,
            $".{Path.GetFileName(outputPath)}.{Guid.NewGuid():N}.tmp");

        try
        {
            using (var file = new FileStream(temporaryPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            using (var archive = new ZipArchive(file, ZipArchiveMode.Create, leaveOpen: false))
            {
                WriteTextEntry(
                    archive,
                    ManifestEntryName,
                    DeploymentPackageJson.Serialize(manifest));

                var modelEntry = archive.CreateEntry(ModelEntryName, CompressionLevel.NoCompression);
                using var source = File.OpenRead(request.ModelPath);
                using var target = modelEntry.Open();
                source.CopyTo(target);
            }

            VerifyCreatedArchive(temporaryPath, manifest);
            File.Move(temporaryPath, outputPath);
            return new DeploymentPackageExportResult(outputPath, manifest);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }

    public static DeploymentPackageImportResult Import(
        string packagePath,
        string packageCacheRoot,
        long maximumModelBytes = 2L * 1024 * 1024 * 1024)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packagePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(packageCacheRoot);

        if (maximumModelBytes <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maximumModelBytes));
        }

        if (!File.Exists(packagePath))
        {
            throw new FileNotFoundException("The selected deployment package does not exist.", packagePath);
        }

        if (!string.Equals(Path.GetExtension(packagePath), FileExtension, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"Select an AIVQC {FileExtension} deployment package.");
        }

        using var archive = ZipFile.OpenRead(packagePath);
        var manifestEntry = GetRequiredEntry(archive, ManifestEntryName);
        var modelEntry = GetRequiredEntry(archive, ModelEntryName);

        if (archive.Entries.Count != 2)
        {
            throw new InvalidDataException("The deployment package contains unexpected files.");
        }

        if (manifestEntry.Length > MaximumManifestBytes)
        {
            throw new InvalidDataException("The deployment-package manifest is too large.");
        }

        if (modelEntry.Length <= 0 || modelEntry.Length > maximumModelBytes)
        {
            throw new InvalidDataException("The deployment-package model exceeds the extraction safety limit.");
        }

        DeploymentPackageManifest manifest;
        using (var reader = new StreamReader(
            manifestEntry.Open(),
            Encoding.UTF8,
            detectEncodingFromByteOrderMarks: true,
            leaveOpen: false))
        {
            manifest = DeploymentPackageJson.Deserialize(reader.ReadToEnd());
        }

        ValidateManifest(manifest);

        var cacheRoot = Path.GetFullPath(packageCacheRoot);
        Directory.CreateDirectory(cacheRoot);
        var destinationDirectory = Path.Combine(cacheRoot, manifest.PackageId.ToString("N"));
        var existing = TryUseExistingPackage(destinationDirectory, manifest);
        if (existing is not null)
        {
            return existing;
        }

        var temporaryDirectory = Path.Combine(cacheRoot, $".{manifest.PackageId:N}.{Guid.NewGuid():N}.tmp");
        Directory.CreateDirectory(temporaryDirectory);

        try
        {
            var modelPath = Path.Combine(temporaryDirectory, ModelEntryName);
            string extractedSha256;
            using (var source = modelEntry.Open())
            using (var target = new FileStream(modelPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                extractedSha256 = CopyAndCalculateSha256(source, target, maximumModelBytes);
            }

            if (!string.Equals(extractedSha256, manifest.Model.Sha256, StringComparison.OrdinalIgnoreCase))
            {
                throw new InvalidDataException(
                    "The model checksum does not match the deployment-package manifest.");
            }

            File.WriteAllText(
                Path.Combine(temporaryDirectory, ManifestEntryName),
                DeploymentPackageJson.Serialize(manifest),
                Encoding.UTF8);
            File.WriteAllText(
                Path.Combine(temporaryDirectory, "classes.json"),
                JsonSerializer.Serialize(
                    manifest.DefectClasses.ToDictionary(item => item.Name, item => item.Id),
                    new JsonSerializerOptions { WriteIndented = true }),
                Encoding.UTF8);

            Directory.Move(temporaryDirectory, destinationDirectory);
            return CreateImportResult(destinationDirectory, manifest);
        }
        finally
        {
            if (Directory.Exists(temporaryDirectory))
            {
                Directory.Delete(temporaryDirectory, recursive: true);
            }
        }
    }

    public static void ValidateManifest(DeploymentPackageManifest manifest)
    {
        ArgumentNullException.ThrowIfNull(manifest);

        if (manifest.SchemaVersion != DeploymentPackageManifest.CurrentSchemaVersion)
        {
            throw new InvalidDataException(
                $"Unsupported deployment-package schema '{manifest.SchemaVersion}'. "
                + $"Expected '{DeploymentPackageManifest.CurrentSchemaVersion}'.");
        }

        if (manifest.PackageId == Guid.Empty)
        {
            throw new InvalidDataException("The deployment package ID cannot be empty.");
        }

        ValidateText(manifest.ProductId, "product ID");
        ValidateText(manifest.RecipeId, "recipe ID");
        ValidateText(manifest.CreatedBy, "creator");

        if (manifest.Model is null
            || manifest.Preprocessing is null
            || manifest.RegionOfInterest is null
            || manifest.DefectClasses is null)
        {
            throw new InvalidDataException("The deployment-package manifest is incomplete.");
        }

        if (manifest.Model.FileName != ModelEntryName
            || manifest.Model.Task != ModelTask.ObjectDetection
            || !string.Equals(manifest.Model.Runtime, "onnxruntime", StringComparison.OrdinalIgnoreCase)
            || manifest.Model.InputWidth <= 0
            || manifest.Model.InputHeight <= 0)
        {
            throw new InvalidDataException("The deployment package contains unsupported model metadata.");
        }

        if (manifest.Model.Sha256.Length != 64
            || !manifest.Model.Sha256.All(Uri.IsHexDigit))
        {
            throw new InvalidDataException("The model SHA-256 checksum is invalid.");
        }

        if (!string.Equals(manifest.Preprocessing.ColorSpace, "RGB", StringComparison.OrdinalIgnoreCase)
            || !string.Equals(manifest.Preprocessing.Normalization, "zeroToOne", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The deployment package uses unsupported preprocessing.");
        }

        var roi = manifest.RegionOfInterest;
        if (roi.X < 0 || roi.Y < 0 || roi.Width < 0 || roi.Height < 0
            || (roi.Width == 0) != (roi.Height == 0))
        {
            throw new InvalidDataException("The deployment-package region of interest is invalid.");
        }

        if (manifest.DefectClasses.Count == 0
            || manifest.DefectClasses.Any(item =>
                item.Id <= 0
                || string.IsNullOrWhiteSpace(item.Name)
                || item.Name.Length > 128
                || !float.IsFinite(item.Threshold)
                || item.Threshold is < 0 or > 1)
            || manifest.DefectClasses.Select(item => item.Id).Distinct().Count()
                != manifest.DefectClasses.Count
            || manifest.DefectClasses.Select(item => item.Name).Distinct(StringComparer.OrdinalIgnoreCase).Count()
                != manifest.DefectClasses.Count)
        {
            throw new InvalidDataException("The deployment-package defect classes are invalid.");
        }
    }

    private static void ValidateExportRequest(DeploymentPackageExportRequest request)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(request.OutputPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(request.ModelPath);

        if (!File.Exists(request.ModelPath))
        {
            throw new FileNotFoundException("The ONNX model selected for export does not exist.", request.ModelPath);
        }

        if (!string.Equals(Path.GetExtension(request.ModelPath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Only ONNX models can be exported.");
        }

        if (!string.Equals(Path.GetExtension(request.OutputPath), FileExtension, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"The deployment package must use the {FileExtension} extension.");
        }

        ValidateText(request.ProductId, "product ID");
        ValidateText(request.RecipeId, "recipe ID");
        ValidateText(request.CreatedBy, "creator");

        if (request.InputWidth <= 0 || request.InputHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(request), "Model dimensions must be positive.");
        }

        if (request.DefectClasses is null || request.DefectClasses.Count == 0)
        {
            throw new InvalidDataException("Add at least one defect class before exporting a package.");
        }
    }

    private static DeploymentPackageImportResult? TryUseExistingPackage(
        string destinationDirectory,
        DeploymentPackageManifest expectedManifest)
    {
        if (!Directory.Exists(destinationDirectory))
        {
            return null;
        }

        var manifestPath = Path.Combine(destinationDirectory, ManifestEntryName);
        var modelPath = Path.Combine(destinationDirectory, ModelEntryName);
        if (!File.Exists(manifestPath) || !File.Exists(modelPath))
        {
            throw new InvalidDataException("The existing cached deployment package is incomplete.");
        }

        var existingManifest = DeploymentPackageJson.Deserialize(File.ReadAllText(manifestPath));
        ValidateManifest(existingManifest);
        if (existingManifest.PackageId != expectedManifest.PackageId
            || !string.Equals(
                DeploymentPackageJson.Serialize(existingManifest),
                DeploymentPackageJson.Serialize(expectedManifest),
                StringComparison.Ordinal)
            || !string.Equals(CalculateSha256(modelPath), expectedManifest.Model.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The cached deployment package failed its integrity check.");
        }

        return CreateImportResult(destinationDirectory, existingManifest);
    }

    private static DeploymentPackageImportResult CreateImportResult(
        string destinationDirectory,
        DeploymentPackageManifest manifest) =>
        new(
            destinationDirectory,
            Path.Combine(destinationDirectory, ModelEntryName),
            Path.Combine(destinationDirectory, ManifestEntryName),
            manifest);

    private static ZipArchiveEntry GetRequiredEntry(ZipArchive archive, string entryName)
    {
        var matches = archive.Entries
            .Where(entry => string.Equals(entry.FullName, entryName, StringComparison.Ordinal))
            .ToArray();
        return matches.Length == 1
            ? matches[0]
            : throw new InvalidDataException(
                $"The deployment package must contain exactly one '{entryName}' entry.");
    }

    private static void WriteTextEntry(ZipArchive archive, string name, string content)
    {
        var entry = archive.CreateEntry(name, CompressionLevel.Optimal);
        using var writer = new StreamWriter(entry.Open(), new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
        writer.Write(content);
    }

    private static void VerifyCreatedArchive(
        string packagePath,
        DeploymentPackageManifest expectedManifest)
    {
        using var archive = ZipFile.OpenRead(packagePath);
        if (archive.Entries.Count != 2)
        {
            throw new InvalidDataException("The generated deployment package contains unexpected files.");
        }

        var manifestEntry = GetRequiredEntry(archive, ManifestEntryName);
        var modelEntry = GetRequiredEntry(archive, ModelEntryName);
        using var reader = new StreamReader(
            manifestEntry.Open(),
            Encoding.UTF8,
            detectEncodingFromByteOrderMarks: true,
            leaveOpen: false);
        var restoredManifest = DeploymentPackageJson.Deserialize(reader.ReadToEnd());
        ValidateManifest(restoredManifest);

        if (!string.Equals(
            DeploymentPackageJson.Serialize(restoredManifest),
            DeploymentPackageJson.Serialize(expectedManifest),
            StringComparison.Ordinal))
        {
            throw new InvalidDataException("The generated deployment-package manifest changed during export.");
        }

        using var modelStream = modelEntry.Open();
        var archivedSha256 = Convert.ToHexString(SHA256.HashData(modelStream));
        if (!string.Equals(archivedSha256, expectedManifest.Model.Sha256, StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("The generated deployment-package model failed verification.");
        }
    }

    private static string CalculateSha256(string filePath)
    {
        using var source = File.OpenRead(filePath);
        return Convert.ToHexString(SHA256.HashData(source));
    }

    private static string CopyAndCalculateSha256(Stream source, Stream target, long maximumBytes)
    {
        using var algorithm = SHA256.Create();
        var buffer = new byte[1024 * 1024];
        long totalBytes = 0;
        int bytesRead;
        while ((bytesRead = source.Read(buffer, 0, buffer.Length)) > 0)
        {
            totalBytes += bytesRead;
            if (totalBytes > maximumBytes)
            {
                throw new InvalidDataException("The deployment-package model exceeds the extraction safety limit.");
            }

            target.Write(buffer, 0, bytesRead);
            algorithm.TransformBlock(buffer, 0, bytesRead, null, 0);
        }

        algorithm.TransformFinalBlock([], 0, 0);
        return Convert.ToHexString(algorithm.Hash!);
    }

    private static void ValidateText(string? value, string fieldName)
    {
        if (string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.Any(char.IsControl))
        {
            throw new InvalidDataException($"The deployment-package {fieldName} is invalid.");
        }
    }
}

public sealed record DeploymentPackageExportRequest(
    string OutputPath,
    string ModelPath,
    string ProductId,
    string RecipeId,
    string CreatedBy,
    int InputWidth,
    int InputHeight,
    PreprocessingManifest Preprocessing,
    RegionOfInterestManifest RegionOfInterest,
    IReadOnlyList<DefectClassManifest> DefectClasses);

public sealed record DeploymentPackageExportResult(
    string PackagePath,
    DeploymentPackageManifest Manifest);

public sealed record DeploymentPackageImportResult(
    string PackageDirectory,
    string ModelPath,
    string ManifestPath,
    DeploymentPackageManifest Manifest);
