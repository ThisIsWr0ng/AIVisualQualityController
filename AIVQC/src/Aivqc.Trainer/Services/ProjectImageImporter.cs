using System.Security.Cryptography;
using Aivqc.Core.Projects;
using SkiaSharp;

namespace Aivqc.Trainer.Services;

public static class ProjectImageImporter
{
    private const long MaximumFileBytes = 250L * 1024 * 1024;
    private const long MaximumDecodedPixels = 100_000_000;
    private const int ThumbnailWidth = 240;
    private const int ThumbnailHeight = 160;

    private static readonly HashSet<string> SupportedExtensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    };

    public static ImageImportResult Import(
        TrainerProjectManifest project,
        string projectDirectory,
        IReadOnlyList<string> sourcePaths,
        ImageStorageMode storageMode,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);
        ArgumentNullException.ThrowIfNull(sourcePaths);

        var root = Path.GetFullPath(projectDirectory);
        Directory.CreateDirectory(Path.Combine(root, TrainerProjectStore.ImagesDirectoryName));
        Directory.CreateDirectory(Path.Combine(root, TrainerProjectStore.ThumbnailsDirectoryName));

        var knownHashes = new HashSet<string>(
            project.Images.Select(image => image.Sha256),
            StringComparer.OrdinalIgnoreCase);
        var imported = new List<ProjectImageAsset>();
        var issues = new List<ImageImportIssue>();
        var duplicates = 0;

        foreach (var requestedPath in sourcePaths.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            cancellationToken.ThrowIfCancellationRequested();

            try
            {
                var sourcePath = Path.GetFullPath(requestedPath);
                var fileInfo = new FileInfo(sourcePath);
                if (!fileInfo.Exists)
                {
                    throw new FileNotFoundException("The source image does not exist.", sourcePath);
                }

                if (!SupportedExtensions.Contains(fileInfo.Extension))
                {
                    throw new InvalidDataException("Supported formats are JPG, PNG, BMP, and WebP.");
                }

                if (fileInfo.Length == 0 || fileInfo.Length > MaximumFileBytes)
                {
                    throw new InvalidDataException("The image is empty or exceeds the 250 MB safety limit.");
                }

                var sha256 = CalculateSha256(sourcePath, cancellationToken);
                if (!knownHashes.Add(sha256))
                {
                    duplicates++;
                    issues.Add(new ImageImportIssue(sourcePath, "Skipped duplicate image (identical SHA-256)."));
                    continue;
                }

                try
                {
                    var image = ImportOne(sourcePath, root, storageMode, sha256, cancellationToken);
                    imported.Add(image);
                }
                catch
                {
                    knownHashes.Remove(sha256);
                    throw;
                }
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception exception)
            {
                issues.Add(new ImageImportIssue(requestedPath, exception.Message));
            }
        }

        return new ImageImportResult(imported, issues, duplicates);
    }

    private static ProjectImageAsset ImportOne(
        string sourcePath,
        string projectDirectory,
        ImageStorageMode storageMode,
        string sha256,
        CancellationToken cancellationToken)
    {
        using var codec = SKCodec.Create(sourcePath)
            ?? throw new InvalidDataException("The file is damaged or is not a supported image.");
        var (extension, formatName) = GetEncodedFormat(codec.EncodedFormat);
        using var bitmap = SKBitmap.Decode(codec)
            ?? throw new InvalidDataException("The file is damaged or is not a supported image.");

        if (bitmap.Width <= 0 || bitmap.Height <= 0
            || (long)bitmap.Width * bitmap.Height > MaximumDecodedPixels)
        {
            throw new InvalidDataException("The decoded image dimensions exceed the safety limit.");
        }

        var warnings = GetWarnings(bitmap.Width, bitmap.Height);
        var imageId = Guid.NewGuid();
        var relativeImagePath = $"{TrainerProjectStore.ImagesDirectoryName}/{imageId:N}{extension}";
        var relativeThumbnailPath = $"{TrainerProjectStore.ThumbnailsDirectoryName}/{imageId:N}.jpg";
        var copiedImagePath = Path.Combine(projectDirectory, relativeImagePath.Replace('/', Path.DirectorySeparatorChar));
        var thumbnailPath = Path.Combine(
            projectDirectory,
            relativeThumbnailPath.Replace('/', Path.DirectorySeparatorChar));

        var copied = false;
        try
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (storageMode == ImageStorageMode.Copy)
            {
                File.Copy(sourcePath, copiedImagePath, overwrite: false);
                copied = true;
            }

            WriteThumbnail(bitmap, thumbnailPath);

            return new ProjectImageAsset(
                imageId,
                Path.GetFileName(sourcePath),
                storageMode,
                storageMode == ImageStorageMode.Copy ? relativeImagePath : Path.GetFullPath(sourcePath),
                relativeThumbnailPath,
                sha256,
                bitmap.Width,
                bitmap.Height,
                formatName,
                DateTimeOffset.UtcNow,
                warnings);
        }
        catch
        {
            if (copied && File.Exists(copiedImagePath))
            {
                File.Delete(copiedImagePath);
            }

            if (File.Exists(thumbnailPath))
            {
                File.Delete(thumbnailPath);
            }

            throw;
        }
    }

    private static void WriteThumbnail(SKBitmap source, string outputPath)
    {
        var scale = Math.Min(
            ThumbnailWidth / (double)source.Width,
            ThumbnailHeight / (double)source.Height);
        scale = Math.Min(scale, 1d);
        var width = Math.Max(1, (int)Math.Round(source.Width * scale));
        var height = Math.Max(1, (int)Math.Round(source.Height * scale));

        using var resized = source.Resize(
            new SKImageInfo(width, height, SKColorType.Rgba8888),
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
            ?? throw new InvalidOperationException("The image thumbnail could not be generated.");
        using var image = SKImage.FromBitmap(resized);
        using var data = image.Encode(SKEncodedImageFormat.Jpeg, 85);
        using var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        data.SaveTo(output);
    }

    private static IReadOnlyList<string> GetWarnings(int width, int height)
    {
        var warnings = new List<string>();
        if (width < 128 || height < 128)
        {
            warnings.Add($"Low resolution ({width} × {height}).");
        }

        var aspectRatio = Math.Max(width / (double)height, height / (double)width);
        if (aspectRatio > 10)
        {
            warnings.Add("Unusual image aspect ratio.");
        }

        return warnings;
    }

    private static string CalculateSha256(string path, CancellationToken cancellationToken)
    {
        using var stream = File.OpenRead(path);
        using var algorithm = SHA256.Create();
        var buffer = new byte[1024 * 1024];
        int bytesRead;
        while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            algorithm.TransformBlock(buffer, 0, bytesRead, null, 0);
        }

        algorithm.TransformFinalBlock([], 0, 0);
        return Convert.ToHexString(algorithm.Hash!);
    }

    private static (string Extension, string Name) GetEncodedFormat(SKEncodedImageFormat format) =>
        format switch
        {
            SKEncodedImageFormat.Jpeg => (".jpg", "jpeg"),
            SKEncodedImageFormat.Png => (".png", "png"),
            SKEncodedImageFormat.Bmp => (".bmp", "bmp"),
            SKEncodedImageFormat.Webp => (".webp", "webp"),
            _ => throw new InvalidDataException("The decoded image format is not supported."),
        };
}

public sealed record ImageImportResult(
    IReadOnlyList<ProjectImageAsset> ImportedImages,
    IReadOnlyList<ImageImportIssue> Issues,
    int DuplicateCount);

public sealed record ImageImportIssue(string FilePath, string Message);
