using Aivqc.Core.Deployment;

namespace Aivqc.Core.Tests.Deployment;

public sealed class DeploymentPackageJsonTests
{
    [Fact]
    public void ManifestRoundTripPreservesDeploymentContract()
    {
        var manifest = new DeploymentPackageManifest(
            DeploymentPackageManifest.CurrentSchemaVersion,
            Guid.Parse("f151197c-4276-4cef-a2e9-f10d98ce0181"),
            "medical-dressing",
            "dressing-line-1",
            new DateTimeOffset(2026, 8, 19, 20, 0, 0, TimeSpan.Zero),
            "Dawid Olesko",
            new ModelManifest(
                "model.onnx",
                "0123456789abcdef",
                ModelTask.ObjectDetection,
                "onnxruntime",
                416,
                416),
            new PreprocessingManifest("RGB", "zeroToOne", true),
            new RegionOfInterestManifest(0, 0, 1920, 1080),
            [
                new DefectClassManifest(0, "Cut", 0.80f),
                new DefectClassManifest(1, "ForeignBody", 0.80f)
            ]);

        var json = DeploymentPackageJson.Serialize(manifest);
        var restored = DeploymentPackageJson.Deserialize(json);

        Assert.Equal(manifest.SchemaVersion, restored.SchemaVersion);
        Assert.Equal(manifest.PackageId, restored.PackageId);
        Assert.Equal(manifest.ProductId, restored.ProductId);
        Assert.Equal(manifest.RecipeId, restored.RecipeId);
        Assert.Equal(manifest.CreatedAtUtc, restored.CreatedAtUtc);
        Assert.Equal(manifest.CreatedBy, restored.CreatedBy);
        Assert.Equal(manifest.Model, restored.Model);
        Assert.Equal(manifest.Preprocessing, restored.Preprocessing);
        Assert.Equal(manifest.RegionOfInterest, restored.RegionOfInterest);
        Assert.Equal(manifest.DefectClasses, restored.DefectClasses);
        Assert.Contains("\"task\": \"objectDetection\"", json);
        Assert.Equal(2, restored.DefectClasses.Count);
    }
}
