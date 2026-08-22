using Aivqc.Core.Connectivity;
using Aivqc.Core.Deployment;
using Aivqc.Server;
using Aivqc.Server.Storage;
using Microsoft.Extensions.Options;

namespace Aivqc.Server.Tests.Storage;

public sealed class ServerPackageStoreTests
{
    [Fact]
    public async Task PublishRouteAcknowledgeAndRevoke_PreserveAuditedState()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var dataDirectory = Path.Combine(root, "data");
            var store = new ServerPackageStore(Options.Create(new ServerOptions
            {
                DataDirectory = dataDirectory,
                ApiKeysFile = Path.Combine(root, "unused.json"),
                MaximumPackageBytes = 10 * 1024 * 1024,
                StorageQuotaBytes = 20 * 1024 * 1024,
            }));
            await store.RegisterStationAsync("line-1", "Packaging line 1", "trainer-main", default);
            var packagePath = CreatePackage(root);

            var published = await store.PublishAsync(
                packagePath,
                "line-1",
                "trainer-main",
                default);
            var latest = store.GetLatest("line-1");

            Assert.NotNull(latest);
            Assert.Equal(published.Package.PackageId, latest.Value.Package.PackageId);
            Assert.True(File.Exists(store.GetPackageContentPath("line-1", published.Package.PackageId)));

            await store.AcknowledgeAsync(
                "line-1",
                published.Package.PackageId,
                new PackageAcknowledgement(PackageAcknowledgementStatus.Activated, "Validated locally"),
                "production-line-1",
                default);
            await store.RevokeAsync(published.Package.PackageId, "server-admin", default);

            var state = store.GetState();
            Assert.True(Assert.Single(state.Packages).Revoked);
            Assert.NotNull(Assert.Single(state.Assignments).ActivatedAtUtc);
            Assert.Null(store.GetLatest("line-1"));
            var audit = File.ReadAllText(Path.Combine(dataDirectory, "audit.jsonl"));
            Assert.Contains("package.published", audit, StringComparison.Ordinal);
            Assert.Contains("package.activated", audit, StringComparison.Ordinal);
            Assert.Contains("package.revoked", audit, StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public async Task Publish_RejectsUnknownTargetStation()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            var store = new ServerPackageStore(Options.Create(new ServerOptions
            {
                DataDirectory = Path.Combine(root, "data"),
                ApiKeysFile = Path.Combine(root, "unused.json"),
                MaximumPackageBytes = 10 * 1024 * 1024,
                StorageQuotaBytes = 20 * 1024 * 1024,
            }));

            await Assert.ThrowsAsync<KeyNotFoundException>(() => store.PublishAsync(
                CreatePackage(root),
                "unknown-line",
                "trainer-main",
                default));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string CreatePackage(string root)
    {
        var modelPath = Path.Combine(root, $"model-{Guid.NewGuid():N}.onnx");
        File.WriteAllBytes(modelPath, Enumerable.Range(0, 1024).Select(index => (byte)index).ToArray());
        var packagePath = Path.Combine(root, $"package-{Guid.NewGuid():N}.aivqcpkg");
        DeploymentPackageArchive.Export(new DeploymentPackageExportRequest(
            packagePath,
            modelPath,
            "product-a",
            "recipe-1",
            "test-author",
            320,
            320,
            new PreprocessingManifest("RGB", "zeroToOne", false),
            new RegionOfInterestManifest(0, 0, 0, 0),
            [new DefectClassManifest(1, "scratch", 0.5f)]));
        return packagePath;
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-server-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
