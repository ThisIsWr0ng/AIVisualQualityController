using System.IO.Compression;
using Aivqc.Core.Deployment;

namespace Aivqc.Core.Tests.Deployment;

public sealed class DeploymentPackageArchiveTests
{
    [Fact]
    public void ExportAndImport_PreserveManifestAndVerifyModel()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var modelPath = Path.Combine(root, "source.onnx");
            File.WriteAllBytes(modelPath, [1, 2, 3, 4, 5]);
            var packagePath = Path.Combine(root, $"dressing{DeploymentPackageArchive.FileExtension}");

            var exported = DeploymentPackageArchive.Export(CreateRequest(packagePath, modelPath));
            var imported = DeploymentPackageArchive.Import(packagePath, Path.Combine(root, "cache"));

            Assert.Equal(exported.Manifest.PackageId, imported.Manifest.PackageId);
            Assert.Equal("medical-dressing", imported.Manifest.ProductId);
            Assert.Equal("dressing-standard", imported.Manifest.RecipeId);
            Assert.Equal(File.ReadAllBytes(modelPath), File.ReadAllBytes(imported.ModelPath));
            Assert.True(File.Exists(imported.ManifestPath));
            Assert.True(File.Exists(Path.Combine(imported.PackageDirectory, "classes.json")));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Import_RejectsModelWhoseChecksumDoesNotMatchManifest()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var modelPath = Path.Combine(root, "source.onnx");
            File.WriteAllBytes(modelPath, [1, 2, 3, 4, 5]);
            var packagePath = Path.Combine(root, $"dressing{DeploymentPackageArchive.FileExtension}");
            DeploymentPackageArchive.Export(CreateRequest(packagePath, modelPath));

            using (var archive = ZipFile.Open(packagePath, ZipArchiveMode.Update))
            {
                archive.GetEntry(DeploymentPackageArchive.ModelEntryName)!.Delete();
                var replacement = archive.CreateEntry(
                    DeploymentPackageArchive.ModelEntryName,
                    CompressionLevel.NoCompression);
                using var stream = replacement.Open();
                stream.Write([9, 9, 9]);
            }

            var exception = Assert.Throws<InvalidDataException>(() =>
                DeploymentPackageArchive.Import(packagePath, Path.Combine(root, "cache")));
            Assert.Contains("checksum", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Import_RejectsUnexpectedArchiveEntries()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var modelPath = Path.Combine(root, "source.onnx");
            File.WriteAllBytes(modelPath, [1, 2, 3, 4, 5]);
            var packagePath = Path.Combine(root, $"dressing{DeploymentPackageArchive.FileExtension}");
            DeploymentPackageArchive.Export(CreateRequest(packagePath, modelPath));

            using (var archive = ZipFile.Open(packagePath, ZipArchiveMode.Update))
            {
                archive.CreateEntry("../unexpected.txt");
            }

            var exception = Assert.Throws<InvalidDataException>(() =>
                DeploymentPackageArchive.Import(packagePath, Path.Combine(root, "cache")));
            Assert.Contains("unexpected", exception.Message, StringComparison.OrdinalIgnoreCase);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void Import_RejectsModelAboveConfiguredExtractionLimit()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var modelPath = Path.Combine(root, "source.onnx");
            File.WriteAllBytes(modelPath, new byte[4096]);
            var packagePath = Path.Combine(root, $"dressing{DeploymentPackageArchive.FileExtension}");
            DeploymentPackageArchive.Export(CreateRequest(packagePath, modelPath));

            var exception = Assert.Throws<InvalidDataException>(() =>
                DeploymentPackageArchive.Import(
                    packagePath,
                    Path.Combine(root, "cache"),
                    maximumModelBytes: 1024));

            Assert.Contains("safety limit", exception.Message, StringComparison.OrdinalIgnoreCase);
            Assert.False(Directory.Exists(Path.Combine(root, "cache"))
                && Directory.EnumerateFiles(Path.Combine(root, "cache"), "*", SearchOption.AllDirectories).Any());
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static DeploymentPackageExportRequest CreateRequest(string outputPath, string modelPath) =>
        new(
            outputPath,
            modelPath,
            "medical-dressing",
            "dressing-standard",
            "Test Engineer",
            320,
            320,
            new PreprocessingManifest("RGB", "zeroToOne", PreserveAspectRatio: false),
            new RegionOfInterestManifest(0, 0, 0, 0),
            [
                new DefectClassManifest(1, "Cut", 0.5f),
                new DefectClassManifest(2, "Foreign body", 0.6f),
            ]);

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-package-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
