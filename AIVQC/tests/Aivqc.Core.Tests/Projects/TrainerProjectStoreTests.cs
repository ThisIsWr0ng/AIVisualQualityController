using Aivqc.Core.Projects;

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
