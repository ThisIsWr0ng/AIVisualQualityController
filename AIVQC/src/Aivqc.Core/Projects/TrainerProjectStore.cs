using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aivqc.Core.Projects;

/// <summary>
/// Creates and atomically persists local Trainer projects.
/// </summary>
public static class TrainerProjectStore
{
    public const string ManifestFileName = "project.aivqc.json";
    public const string ImagesDirectoryName = "images";
    public const string ThumbnailsDirectoryName = "thumbnails";

    private static readonly JsonSerializerOptions SerializerOptions = CreateSerializerOptions();

    public static TrainerProjectManifest Create(
        string projectDirectory,
        string name,
        string productId)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);
        ValidateText(name, "project name");
        ValidateText(productId, "product ID");

        var root = Path.GetFullPath(projectDirectory);
        var manifestPath = Path.Combine(root, ManifestFileName);
        if (File.Exists(manifestPath))
        {
            throw new IOException($"A Trainer project already exists in {root}.");
        }

        Directory.CreateDirectory(root);
        Directory.CreateDirectory(Path.Combine(root, ImagesDirectoryName));
        Directory.CreateDirectory(Path.Combine(root, ThumbnailsDirectoryName));

        var now = DateTimeOffset.UtcNow;
        var manifest = new TrainerProjectManifest(
            TrainerProjectManifest.CurrentSchemaVersion,
            Guid.NewGuid(),
            name.Trim(),
            productId.Trim(),
            now,
            now,
            [],
            []);
        Save(root, manifest);
        return manifest;
    }

    public static TrainerProjectManifest Load(string projectPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectPath);

        var manifestPath = ResolveManifestPath(projectPath);
        if (!File.Exists(manifestPath))
        {
            throw new FileNotFoundException(
                $"The selected directory does not contain {ManifestFileName}.",
                manifestPath);
        }

        TrainerProjectManifest manifest;
        try
        {
            manifest = JsonSerializer.Deserialize<TrainerProjectManifest>(
                File.ReadAllText(manifestPath, Encoding.UTF8),
                SerializerOptions)
                ?? throw new InvalidDataException("The Trainer project manifest is empty.");
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The Trainer project manifest contains invalid JSON.", exception);
        }

        Validate(manifest, Path.GetDirectoryName(manifestPath)!);
        return manifest;
    }

    public static void Save(string projectDirectory, TrainerProjectManifest manifest)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);
        ArgumentNullException.ThrowIfNull(manifest);

        var root = Path.GetFullPath(projectDirectory);
        Validate(manifest, root);
        Directory.CreateDirectory(root);
        Directory.CreateDirectory(Path.Combine(root, ImagesDirectoryName));
        Directory.CreateDirectory(Path.Combine(root, ThumbnailsDirectoryName));

        var manifestPath = Path.Combine(root, ManifestFileName);
        var temporaryPath = Path.Combine(root, $".{ManifestFileName}.{Guid.NewGuid():N}.tmp");
        try
        {
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(manifest, SerializerOptions),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.Move(temporaryPath, manifestPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }

    public static string ResolveImagePath(string projectDirectory, ProjectImageAsset image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return image.StorageMode == ImageStorageMode.Reference
            ? Path.GetFullPath(image.Location)
            : ResolveContainedPath(projectDirectory, image.Location, "image");
    }

    public static string ResolveThumbnailPath(string projectDirectory, ProjectImageAsset image)
    {
        ArgumentNullException.ThrowIfNull(image);
        return ResolveContainedPath(projectDirectory, image.ThumbnailPath, "thumbnail");
    }

    public static void Validate(TrainerProjectManifest manifest, string projectDirectory)
    {
        ArgumentNullException.ThrowIfNull(manifest);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);

        if (manifest.SchemaVersion != TrainerProjectManifest.CurrentSchemaVersion)
        {
            throw new InvalidDataException(
                $"Unsupported Trainer project schema '{manifest.SchemaVersion}'. "
                + $"Expected '{TrainerProjectManifest.CurrentSchemaVersion}'.");
        }

        if (manifest.ProjectId == Guid.Empty)
        {
            throw new InvalidDataException("The Trainer project ID cannot be empty.");
        }

        ValidateText(manifest.Name, "project name");
        ValidateText(manifest.ProductId, "product ID");

        if (manifest.CreatedAtUtc == default
            || manifest.UpdatedAtUtc < manifest.CreatedAtUtc
            || manifest.DefectClasses is null
            || manifest.Images is null)
        {
            throw new InvalidDataException("The Trainer project metadata is incomplete.");
        }

        if (manifest.DefectClasses.Any(string.IsNullOrWhiteSpace)
            || manifest.DefectClasses.Distinct(StringComparer.OrdinalIgnoreCase).Count()
                != manifest.DefectClasses.Count)
        {
            throw new InvalidDataException("The Trainer project defect classes are invalid.");
        }

        if (manifest.Images.Select(image => image.ImageId).Distinct().Count() != manifest.Images.Count
            || manifest.Images.Select(image => image.Sha256).Distinct(StringComparer.OrdinalIgnoreCase).Count()
                != manifest.Images.Count)
        {
            throw new InvalidDataException("The Trainer project contains duplicate image IDs or hashes.");
        }

        foreach (var image in manifest.Images)
        {
            ValidateImage(image, projectDirectory, manifest.DefectClasses);
        }
    }

    public static string ResolveManifestPath(string projectPath)
    {
        var fullPath = Path.GetFullPath(projectPath);
        return Directory.Exists(fullPath)
            ? Path.Combine(fullPath, ManifestFileName)
            : fullPath;
    }

    private static void ValidateImage(
        ProjectImageAsset image,
        string projectDirectory,
        IReadOnlyList<string> defectClasses)
    {
        if (image.ImageId == Guid.Empty
            || string.IsNullOrWhiteSpace(image.SourceFileName)
            || image.Width <= 0
            || image.Height <= 0
            || string.IsNullOrWhiteSpace(image.Format)
            || image.ImportedAtUtc == default
            || image.Warnings is null
            || image.Sha256.Length != 64
            || !image.Sha256.All(Uri.IsHexDigit))
        {
            throw new InvalidDataException("The Trainer project contains invalid image metadata.");
        }

        if (image.StorageMode == ImageStorageMode.Reference)
        {
            if (!Path.IsPathFullyQualified(image.Location))
            {
                throw new InvalidDataException("Referenced images must use an absolute path.");
            }
        }
        else
        {
            ResolveContainedPath(projectDirectory, image.Location, "image");
        }

        ResolveContainedPath(projectDirectory, image.ThumbnailPath, "thumbnail");

        var annotations = image.Annotations ?? [];
        if (annotations.Select(annotation => annotation.AnnotationId).Distinct().Count() != annotations.Count)
        {
            throw new InvalidDataException("The image contains duplicate annotation IDs.");
        }

        foreach (var annotation in annotations)
        {
            ValidateAnnotation(annotation, defectClasses);
        }
    }

    private static void ValidateAnnotation(
        ProjectObjectAnnotation annotation,
        IReadOnlyList<string> defectClasses)
    {
        const double coordinateTolerance = 0.000001;
        if (annotation.AnnotationId == Guid.Empty
            || annotation.UpdatedAtUtc == default
            || !defectClasses.Contains(annotation.ClassName, StringComparer.OrdinalIgnoreCase)
            || !double.IsFinite(annotation.X)
            || !double.IsFinite(annotation.Y)
            || !double.IsFinite(annotation.Width)
            || !double.IsFinite(annotation.Height)
            || annotation.X < 0
            || annotation.Y < 0
            || annotation.Width <= 0
            || annotation.Height <= 0
            || annotation.X + annotation.Width > 1 + coordinateTolerance
            || annotation.Y + annotation.Height > 1 + coordinateTolerance)
        {
            throw new InvalidDataException("The Trainer project contains an invalid object annotation.");
        }
    }

    private static string ResolveContainedPath(string projectDirectory, string relativePath, string description)
    {
        if (string.IsNullOrWhiteSpace(relativePath) || Path.IsPathFullyQualified(relativePath))
        {
            throw new InvalidDataException($"The project {description} path must be relative.");
        }

        var root = Path.GetFullPath(projectDirectory)
            .TrimEnd(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
        var resolved = Path.GetFullPath(Path.Combine(root, relativePath));
        if (!resolved.StartsWith(
            root + Path.DirectorySeparatorChar,
            StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException($"The project {description} path leaves the project directory.");
        }

        return resolved;
    }

    private static void ValidateText(string? value, string fieldName)
    {
        if (string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.Any(char.IsControl))
        {
            throw new InvalidDataException($"The Trainer project {fieldName} is invalid.");
        }
    }

    private static JsonSerializerOptions CreateSerializerOptions()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true,
        };
        options.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        return options;
    }
}
