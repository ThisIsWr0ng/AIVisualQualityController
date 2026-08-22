using System.Xml.Linq;
using Aivqc.Core.Projects;
using Aivqc.Trainer.Services;
using SkiaSharp;

namespace Aivqc.Trainer.Tests.Services;

public sealed class ProjectDatasetExporterTests
{
    [Fact]
    public void Export_CreatesDeterministicVocSplitsAndPixelCoordinates()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var projectDirectory = Path.Combine(root, "project");
            var project = TrainerProjectStore.Create(projectDirectory, "Inspection", "product") with
            {
                DefectClasses = ["scratch", "cut"],
            };
            var images = Enumerable.Range(0, 10)
                .Select(index => CreateAnnotatedImage(
                    root,
                    index,
                    index == 1 ? "cut" : "scratch"))
                .ToArray();
            project = project with
            {
                Images = images,
                UpdatedAtUtc = DateTimeOffset.UtcNow,
            };

            var result = ProjectDatasetExporter.Export(project, projectDirectory);

            Assert.Equal(7, result.TrainingImageCount);
            Assert.Equal(2, result.ValidationImageCount);
            Assert.Equal(1, result.TestImageCount);
            Assert.Equal(10, result.AnnotationCount);
            Assert.Equal(7, Directory.EnumerateFiles(
                Path.Combine(result.DatasetDirectory, "train"), "*.xml").Count());
            Assert.Equal(2, Directory.EnumerateFiles(
                Path.Combine(result.DatasetDirectory, "valid"), "*.xml").Count());
            Assert.Single(Directory.EnumerateFiles(
                Path.Combine(result.DatasetDirectory, "test"), "*.xml"));
            Assert.True(File.Exists(Path.Combine(result.DatasetDirectory, "dataset.json")));
            using var exportedImage = SKBitmap.Decode(Directory.EnumerateFiles(
                Path.Combine(result.DatasetDirectory, "train"), "*.jpg").First());
            Assert.Equal(100, exportedImage.Width);
            Assert.Equal(200, exportedImage.Height);

            var trainingDocuments = Directory
                .EnumerateFiles(Path.Combine(result.DatasetDirectory, "train"), "*.xml")
                .Select(XDocument.Load)
                .ToArray();
            var trainingClasses = trainingDocuments
                .SelectMany(document => document.Descendants("name"))
                .Select(element => element.Value)
                .ToHashSet(StringComparer.OrdinalIgnoreCase);
            Assert.Contains("scratch", trainingClasses);
            Assert.Contains("cut", trainingClasses);

            var box = trainingDocuments[0].Descendants("bndbox").Single();
            Assert.Equal("10", box.Element("xmin")!.Value);
            Assert.Equal("40", box.Element("ymin")!.Value);
            Assert.Equal("40", box.Element("xmax")!.Value);
            Assert.Equal("120", box.Element("ymax")!.Value);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Export_RejectsDatasetWithoutEnoughAnnotatedImages()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var projectDirectory = Path.Combine(root, "project");
            var project = TrainerProjectStore.Create(projectDirectory, "Inspection", "product") with
            {
                DefectClasses = ["scratch"],
                Images =
                [
                    CreateAnnotatedImage(root, 0, "scratch"),
                    CreateAnnotatedImage(root, 1, "scratch"),
                ],
                UpdatedAtUtc = DateTimeOffset.UtcNow,
            };

            var exception = Assert.Throws<InvalidOperationException>(() =>
                ProjectDatasetExporter.Export(project, projectDirectory));

            Assert.Contains("three annotated images", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static ProjectImageAsset CreateAnnotatedImage(string root, int index, string className)
    {
        var path = Path.Combine(root, $"source-{index}.jpg");
        using var bitmap = new SKBitmap(100, 200);
        bitmap.Erase(new SKColor((byte)(20 + index), 120, 180));
        using var encodedImage = SKImage.FromBitmap(bitmap);
        using var data = encodedImage.Encode(SKEncodedImageFormat.Jpeg, 90);
        using (var output = File.Create(path))
        {
            data.SaveTo(output);
        }

        return new ProjectImageAsset(
            Guid.NewGuid(),
            Path.GetFileName(path),
            ImageStorageMode.Reference,
            path,
            $"thumbnails/{Guid.NewGuid():N}.jpg",
            index.ToString("X64"),
            100,
            200,
            "jpeg",
            DateTimeOffset.UtcNow,
            [],
            [
                new ProjectObjectAnnotation(
                    Guid.NewGuid(),
                    className,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    DateTimeOffset.UtcNow),
            ]);
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-dataset-export-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
