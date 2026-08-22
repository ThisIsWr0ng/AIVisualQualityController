using System.Text.Json;
using System.Xml;
using System.Xml.Linq;
using Aivqc.Core.Projects;

namespace Aivqc.Trainer.Services;

/// <summary>
/// Creates immutable Pascal VOC snapshots from manually annotated Trainer projects.
/// </summary>
public static class ProjectDatasetExporter
{
    public const string DatasetsDirectoryName = "datasets";

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    public static ProjectDatasetExportResult Export(
        TrainerProjectManifest project,
        string projectDirectory,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(project);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);

        var annotatedImages = project.Images
            .Where(image => (image.Annotations?.Count ?? 0) > 0)
            .OrderBy(image => image.Sha256, StringComparer.Ordinal)
            .ThenBy(image => image.ImageId)
            .ToArray();
        ValidateProject(project, annotatedImages);

        var split = CreateSplit(project, annotatedImages);
        var root = Path.GetFullPath(projectDirectory);
        var datasetsRoot = Path.Combine(root, DatasetsDirectoryName);
        Directory.CreateDirectory(datasetsRoot);
        var snapshotName = $"voc-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}-{Guid.NewGuid():N}"[..36];
        var outputDirectory = Path.Combine(datasetsRoot, snapshotName);
        var stagingDirectory = Path.Combine(datasetsRoot, $".{snapshotName}.tmp");

        try
        {
            Directory.CreateDirectory(stagingDirectory);
            ExportSplit("train", split.Training, stagingDirectory, root, cancellationToken);
            ExportSplit("valid", split.Validation, stagingDirectory, root, cancellationToken);
            if (split.Test.Count > 0)
            {
                ExportSplit("test", split.Test, stagingDirectory, root, cancellationToken);
            }

            var manifest = new ProjectDatasetSnapshot(
                "1.0",
                project.ProjectId,
                project.UpdatedAtUtc,
                DateTimeOffset.UtcNow,
                project.DefectClasses.ToArray(),
                split.Training.Select(image => image.ImageId).ToArray(),
                split.Validation.Select(image => image.ImageId).ToArray(),
                split.Test.Select(image => image.ImageId).ToArray());
            File.WriteAllText(
                Path.Combine(stagingDirectory, "dataset.json"),
                JsonSerializer.Serialize(manifest, JsonOptions));
            Directory.Move(stagingDirectory, outputDirectory);

            return new ProjectDatasetExportResult(
                outputDirectory,
                split.Training.Count,
                split.Validation.Count,
                split.Test.Count,
                annotatedImages.Sum(image => image.Annotations!.Count));
        }
        catch
        {
            if (Directory.Exists(stagingDirectory))
            {
                Directory.Delete(stagingDirectory, recursive: true);
            }

            throw;
        }
    }

    private static DatasetSplit CreateSplit(
        TrainerProjectManifest project,
        IReadOnlyList<ProjectImageAsset> images)
    {
        var trainingTarget = Math.Clamp(
            (int)Math.Round(images.Count * 0.7, MidpointRounding.AwayFromZero),
            2,
            images.Count - 1);
        var training = new List<ProjectImageAsset>();
        var trainingIds = new HashSet<Guid>();

        foreach (var className in project.DefectClasses)
        {
            var classSample = images.First(image =>
                image.Annotations!.Any(annotation => string.Equals(
                    annotation.ClassName,
                    className,
                    StringComparison.OrdinalIgnoreCase)));
            if (trainingIds.Add(classSample.ImageId))
            {
                training.Add(classSample);
            }
        }

        if (training.Count >= images.Count)
        {
            throw new InvalidOperationException(
                "At least one additional annotated image is required for validation after covering all classes in training.");
        }

        trainingTarget = Math.Max(trainingTarget, training.Count);
        foreach (var image in images)
        {
            if (training.Count >= trainingTarget)
            {
                break;
            }

            if (trainingIds.Add(image.ImageId))
            {
                training.Add(image);
            }
        }

        var remaining = images.Where(image => !trainingIds.Contains(image.ImageId)).ToArray();
        var validationTarget = Math.Clamp(
            (int)Math.Round(images.Count * 0.2, MidpointRounding.AwayFromZero),
            1,
            remaining.Length);
        var validation = remaining.Take(validationTarget).ToArray();
        var test = remaining.Skip(validationTarget).ToArray();
        return new DatasetSplit(training, validation, test);
    }

    private static void ExportSplit(
        string splitName,
        IReadOnlyList<ProjectImageAsset> images,
        string outputRoot,
        string projectDirectory,
        CancellationToken cancellationToken)
    {
        var outputDirectory = Path.Combine(outputRoot, splitName);
        Directory.CreateDirectory(outputDirectory);

        foreach (var image in images)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var sourcePath = TrainerProjectStore.ResolveImagePath(projectDirectory, image);
            if (!File.Exists(sourcePath))
            {
                throw new FileNotFoundException(
                    $"Annotated source image '{image.SourceFileName}' is missing.",
                    sourcePath);
            }

            var extension = Path.GetExtension(sourcePath).ToLowerInvariant();
            var baseName = image.ImageId.ToString("N");
            var fileName = baseName + extension;
            File.Copy(sourcePath, Path.Combine(outputDirectory, fileName), overwrite: false);
            WritePascalVocAnnotation(
                Path.Combine(outputDirectory, baseName + ".xml"),
                splitName,
                fileName,
                image);
        }
    }

    private static void WritePascalVocAnnotation(
        string outputPath,
        string splitName,
        string fileName,
        ProjectImageAsset image)
    {
        var document = new XDocument(
            new XElement("annotation",
                new XElement("folder", splitName),
                new XElement("filename", fileName),
                new XElement("source", new XElement("database", "AIVQC Trainer")),
                new XElement("size",
                    new XElement("width", image.Width),
                    new XElement("height", image.Height),
                    new XElement("depth", 3)),
                new XElement("segmented", 0),
                image.Annotations!.Select(annotation =>
                    new XElement("object",
                        new XElement("name", annotation.ClassName),
                        new XElement("pose", "Unspecified"),
                        new XElement("truncated", 0),
                        new XElement("difficult", 0),
                        new XElement("bndbox",
                            new XElement("xmin", ToMinimumPixel(annotation.X, image.Width)),
                            new XElement("ymin", ToMinimumPixel(annotation.Y, image.Height)),
                            new XElement("xmax", ToMaximumPixel(annotation.X + annotation.Width, image.Width)),
                            new XElement("ymax", ToMaximumPixel(annotation.Y + annotation.Height, image.Height)))))));

        var settings = new XmlWriterSettings
        {
            Encoding = new System.Text.UTF8Encoding(encoderShouldEmitUTF8Identifier: false),
            Indent = true,
        };
        using var writer = XmlWriter.Create(outputPath, settings);
        document.Save(writer);
    }

    private static int ToMinimumPixel(double coordinate, int dimension) =>
        Math.Clamp((int)Math.Floor(coordinate * dimension), 0, dimension - 1);

    private static int ToMaximumPixel(double coordinate, int dimension) =>
        Math.Clamp((int)Math.Ceiling(coordinate * dimension - 1e-9), 1, dimension);

    private static void ValidateProject(
        TrainerProjectManifest project,
        IReadOnlyList<ProjectImageAsset> annotatedImages)
    {
        if (project.DefectClasses.Count == 0)
        {
            throw new InvalidOperationException("Add at least one defect class before preparing training data.");
        }

        if (annotatedImages.Count < 3)
        {
            throw new InvalidOperationException(
                "At least three annotated images are required: two for training and one for validation.");
        }

        var usedClasses = annotatedImages
            .SelectMany(image => image.Annotations!)
            .Select(annotation => annotation.ClassName)
            .ToHashSet(StringComparer.OrdinalIgnoreCase);
        var missingClasses = project.DefectClasses
            .Where(className => !usedClasses.Contains(className))
            .ToArray();
        if (missingClasses.Length > 0)
        {
            throw new InvalidOperationException(
                $"Every defect class needs an annotation. Missing: {string.Join(", ", missingClasses)}.");
        }
    }

    private sealed record DatasetSplit(
        IReadOnlyList<ProjectImageAsset> Training,
        IReadOnlyList<ProjectImageAsset> Validation,
        IReadOnlyList<ProjectImageAsset> Test);
}

public sealed record ProjectDatasetExportResult(
    string DatasetDirectory,
    int TrainingImageCount,
    int ValidationImageCount,
    int TestImageCount,
    int AnnotationCount);

public sealed record ProjectDatasetSnapshot(
    string SchemaVersion,
    Guid ProjectId,
    DateTimeOffset ProjectUpdatedAtUtc,
    DateTimeOffset ExportedAtUtc,
    IReadOnlyList<string> DefectClasses,
    IReadOnlyList<Guid> TrainingImageIds,
    IReadOnlyList<Guid> ValidationImageIds,
    IReadOnlyList<Guid> TestImageIds);
