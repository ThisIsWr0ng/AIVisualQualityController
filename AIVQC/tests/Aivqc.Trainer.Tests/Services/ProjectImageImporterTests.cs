using Aivqc.Core.Projects;
using Aivqc.Trainer.Services;
using SkiaSharp;

namespace Aivqc.Trainer.Tests.Services;

public sealed class ProjectImageImporterTests
{
    [Fact]
    public void Import_CopyModeCreatesProjectAssetThumbnailAndSkipsDuplicate()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var projectDirectory = Path.Combine(root, "project");
            var project = TrainerProjectStore.Create(projectDirectory, "Inspection", "product");
            var sourcePath = Path.Combine(root, "sample.png");
            WriteImage(sourcePath, 640, 480);

            var first = ProjectImageImporter.Import(
                project,
                projectDirectory,
                [sourcePath],
                ImageStorageMode.Copy);
            var image = Assert.Single(first.ImportedImages);
            var updated = project with
            {
                UpdatedAtUtc = DateTimeOffset.UtcNow,
                Images = first.ImportedImages,
            };
            TrainerProjectStore.Save(projectDirectory, updated);

            Assert.True(File.Exists(TrainerProjectStore.ResolveImagePath(projectDirectory, image)));
            Assert.True(File.Exists(TrainerProjectStore.ResolveThumbnailPath(projectDirectory, image)));
            Assert.Equal(640, image.Width);
            Assert.Equal(480, image.Height);
            Assert.Empty(image.Warnings);

            var duplicate = ProjectImageImporter.Import(
                updated,
                projectDirectory,
                [sourcePath],
                ImageStorageMode.Copy);
            Assert.Empty(duplicate.ImportedImages);
            Assert.Equal(1, duplicate.DuplicateCount);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Import_ReferenceModeKeepsAbsoluteOriginalPathAndAddsQualityWarning()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var projectDirectory = Path.Combine(root, "project");
            var project = TrainerProjectStore.Create(projectDirectory, "Inspection", "product");
            var sourcePath = Path.Combine(root, "small.webp");
            WriteImage(sourcePath, 64, 64, SKEncodedImageFormat.Webp);

            var result = ProjectImageImporter.Import(
                project,
                projectDirectory,
                [sourcePath],
                ImageStorageMode.Reference);
            var image = Assert.Single(result.ImportedImages);

            Assert.Equal(ImageStorageMode.Reference, image.StorageMode);
            Assert.True(Path.IsPathFullyQualified(image.Location));
            Assert.Equal(Path.GetFullPath(sourcePath), image.Location);
            Assert.Single(image.Warnings);
            Assert.True(File.Exists(TrainerProjectStore.ResolveThumbnailPath(projectDirectory, image)));
            Assert.Empty(Directory.EnumerateFiles(Path.Combine(projectDirectory, "images")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Import_DamagedImageIsReportedWithoutChangingProject()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var projectDirectory = Path.Combine(root, "project");
            var project = TrainerProjectStore.Create(projectDirectory, "Inspection", "product");
            var sourcePath = Path.Combine(root, "damaged.jpg");
            File.WriteAllText(sourcePath, "not an image");

            var result = ProjectImageImporter.Import(
                project,
                projectDirectory,
                [sourcePath],
                ImageStorageMode.Copy);

            Assert.Empty(result.ImportedImages);
            Assert.Single(result.Issues);
            Assert.Empty(Directory.EnumerateFiles(Path.Combine(projectDirectory, "images")));
            Assert.Empty(Directory.EnumerateFiles(Path.Combine(projectDirectory, "thumbnails")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static void WriteImage(
        string path,
        int width,
        int height,
        SKEncodedImageFormat format = SKEncodedImageFormat.Png)
    {
        using var bitmap = new SKBitmap(width, height);
        bitmap.Erase(new SKColor(54, 214, 168));
        using var image = SKImage.FromBitmap(bitmap);
        using var data = image.Encode(format, 90);
        using var output = File.Create(path);
        data.SaveTo(output);
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-image-import-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
