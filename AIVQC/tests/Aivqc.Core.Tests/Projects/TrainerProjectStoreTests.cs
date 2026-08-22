using Aivqc.Core.Projects;
using System.Text.Json.Nodes;

namespace Aivqc.Core.Tests.Projects;

public sealed class TrainerProjectStoreTests
{
    [Fact]
    public void CreateSaveAndLoad_PreserveProjectMetadata()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Dressing inspection", "dressing");
            var image = new ProjectImageAsset(
                Guid.NewGuid(),
                "sample.jpg",
                ImageStorageMode.Copy,
                "images/sample.jpg",
                "thumbnails/sample.jpg",
                new string('A', 64),
                640,
                480,
                "jpeg",
                DateTimeOffset.UtcNow,
                []);
            var changed = created with
            {
                Name = "Updated inspection",
                UpdatedAtUtc = DateTimeOffset.UtcNow,
                Images = [image],
            };

            TrainerProjectStore.Save(root, changed);
            var loaded = TrainerProjectStore.Load(root);

            Assert.Equal(changed.ProjectId, loaded.ProjectId);
            Assert.Equal("Updated inspection", loaded.Name);
            var restoredImage = Assert.Single(loaded.Images);
            Assert.Equal(image.ImageId, restoredImage.ImageId);
            Assert.Equal(image.SourceFileName, restoredImage.SourceFileName);
            Assert.Equal(image.Location, restoredImage.Location);
            Assert.Equal(image.Sha256, restoredImage.Sha256);
            Assert.Empty(restoredImage.Warnings);
            Assert.Null(restoredImage.Annotations);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void SaveAndLoad_PreserveValidatedObjectAnnotations()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Inspection", "product");
            var image = CreateImage("sample.jpg", new string('C', 64));
            var project = created with { Images = [image] };
            project = TrainerProjectAnnotations.AddClass(project, "scratch");
            project = TrainerProjectAnnotations.Add(
                project,
                image.ImageId,
                "SCRATCH",
                new NormalizedBoundingBox(0.1, 0.2, 0.3, 0.4));

            TrainerProjectStore.Save(root, project);
            var loaded = TrainerProjectStore.Load(root);

            var annotation = Assert.Single(Assert.Single(loaded.Images).Annotations!);
            Assert.Equal("scratch", annotation.ClassName);
            Assert.Equal(0.1, annotation.X);
            Assert.Equal(0.2, annotation.Y);
            Assert.Equal(0.3, annotation.Width);
            Assert.Equal(0.4, annotation.Height);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Load_AcceptsLegacyImageWithoutAnnotationsProperty()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Legacy inspection", "product");
            var project = created with
            {
                Images = [CreateImage("legacy.jpg", new string('F', 64))],
                UpdatedAtUtc = DateTimeOffset.UtcNow,
            };
            TrainerProjectStore.Save(root, project);

            var manifestPath = Path.Combine(root, TrainerProjectStore.ManifestFileName);
            var document = JsonNode.Parse(File.ReadAllText(manifestPath))!.AsObject();
            document["images"]![0]!.AsObject().Remove("annotations");
            File.WriteAllText(manifestPath, document.ToJsonString());

            var loaded = TrainerProjectStore.Load(root);

            Assert.Null(Assert.Single(loaded.Images).Annotations);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Save_RejectsAnnotationOutsideImage()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Inspection", "product");
            var image = CreateImage("sample.jpg", new string('D', 64)) with
            {
                Annotations =
                [
                    new ProjectObjectAnnotation(
                        Guid.NewGuid(),
                        "scratch",
                        0.8,
                        0.2,
                        0.3,
                        0.4,
                        DateTimeOffset.UtcNow),
                ],
            };
            var invalid = created with
            {
                DefectClasses = ["scratch"],
                Images = [image],
                UpdatedAtUtc = DateTimeOffset.UtcNow,
            };

            Assert.Throws<InvalidDataException>(() => TrainerProjectStore.Save(root, invalid));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void AnnotationEditor_RejectsUnknownClassAndRemovesExistingAnnotation()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Inspection", "product");
            var image = CreateImage("sample.jpg", new string('E', 64));
            var project = created with { Images = [image] };

            Assert.Throws<InvalidOperationException>(() => TrainerProjectAnnotations.Add(
                project,
                image.ImageId,
                "scratch",
                new NormalizedBoundingBox(0.1, 0.1, 0.2, 0.2)));

            project = TrainerProjectAnnotations.AddClass(project, "scratch");
            project = TrainerProjectAnnotations.Add(
                project,
                image.ImageId,
                "scratch",
                new NormalizedBoundingBox(0.1, 0.1, 0.2, 0.2));
            var annotationId = Assert.Single(project.Images[0].Annotations!).AnnotationId;

            project = TrainerProjectAnnotations.Remove(project, image.ImageId, annotationId);

            Assert.Empty(project.Images[0].Annotations!);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Load_RejectsCopiedImagePathOutsideProject()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Inspection", "product");
            var invalid = created with
            {
                UpdatedAtUtc = DateTimeOffset.UtcNow,
                Images =
                [
                    new ProjectImageAsset(
                        Guid.NewGuid(),
                        "sample.jpg",
                        ImageStorageMode.Copy,
                        "../sample.jpg",
                        "thumbnails/sample.jpg",
                        new string('A', 64),
                        640,
                        480,
                        "jpeg",
                        DateTimeOffset.UtcNow,
                        []),
                ],
            };

            var exception = Assert.Throws<InvalidDataException>(() =>
                TrainerProjectStore.Save(root, invalid));
            Assert.Contains("leaves", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Save_RejectsDuplicateImageHashes()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var created = TrainerProjectStore.Create(root, "Inspection", "product");
            var hash = new string('B', 64);
            var images = new[]
            {
                CreateImage("first.jpg", hash),
                CreateImage("second.jpg", hash),
            };
            var invalid = created with
            {
                UpdatedAtUtc = DateTimeOffset.UtcNow,
                Images = images,
            };

            Assert.Throws<InvalidDataException>(() => TrainerProjectStore.Save(root, invalid));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static ProjectImageAsset CreateImage(string name, string hash) =>
        new(
            Guid.NewGuid(),
            name,
            ImageStorageMode.Reference,
            Path.GetFullPath(name),
            $"thumbnails/{Guid.NewGuid():N}.jpg",
            hash,
            100,
            100,
            "jpeg",
            DateTimeOffset.UtcNow,
            []);

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-project-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
