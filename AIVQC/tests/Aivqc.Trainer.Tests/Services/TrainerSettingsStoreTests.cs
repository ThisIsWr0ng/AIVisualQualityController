using Aivqc.Trainer.Services;

namespace Aivqc.Trainer.Tests.Services;

public sealed class TrainerSettingsStoreTests
{
    [Fact]
    public void LoadReturnsUsableDefaultWhenSettingsDoNotExist()
    {
        var root = CreateTemporaryDirectory();
        try
        {
            var store = new TrainerSettingsStore(Path.Combine(root, "settings.json"));

            var settings = store.Load();

            Assert.Equal(TrainerSettingsStore.CurrentSchemaVersion, settings.SchemaVersion);
            Assert.Single(settings.Lines);
            Assert.Equal(settings.SelectedLineId, settings.Lines[0].Id);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void SaveAndLoadPreserveLinesProductsAndStatistics()
    {
        var root = CreateTemporaryDirectory();
        try
        {
            var path = Path.Combine(root, "settings.json");
            var store = new TrainerSettingsStore(path);
            var lastDeployment = new DateTimeOffset(2026, 8, 22, 18, 30, 0, TimeSpan.Zero);
            var expected = new TrainerApplicationSettings(
                TrainerSettingsStore.CurrentSchemaVersion,
                "line-b",
                true,
                [
                    new DeploymentLineSettings("line-a", "Assembly A", ["product-a"], 3, lastDeployment),
                    new DeploymentLineSettings("line-b", "Assembly B", ["product-b", "product-c"], 7, lastDeployment),
                ]);

            store.Save(expected);
            var restored = store.Load();

            Assert.Equal("line-b", restored.SelectedLineId);
            Assert.True(restored.ExpertModeEnabled);
            Assert.Equal(2, restored.Lines.Count);
            Assert.Equal(["product-b", "product-c"], restored.Lines[1].ProductIds);
            Assert.Equal(7, restored.Lines[1].DeploymentCount);
            Assert.Equal(lastDeployment, restored.Lines[1].LastDeploymentUtc);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void SaveRejectsMissingSelectedLine()
    {
        var root = CreateTemporaryDirectory();
        try
        {
            var store = new TrainerSettingsStore(Path.Combine(root, "settings.json"));
            var settings = new TrainerApplicationSettings(
                TrainerSettingsStore.CurrentSchemaVersion,
                "missing",
                false,
                [new DeploymentLineSettings("line-1", "Line 1", [], 0, null)]);

            Assert.Throws<InvalidDataException>(() => store.Save(settings));
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-settings-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
