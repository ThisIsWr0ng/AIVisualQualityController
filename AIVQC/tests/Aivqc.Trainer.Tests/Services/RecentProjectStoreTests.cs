using Aivqc.Trainer.Services;

namespace Aivqc.Trainer.Tests.Services;

public sealed class RecentProjectStoreTests
{
    [Fact]
    public void Add_MovesExistingProjectToTopWithoutDuplicatingIt()
    {
        var root = Path.Combine(Path.GetTempPath(), $"aivqc-recent-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);

        try
        {
            var store = new RecentProjectStore(Path.Combine(root, "recent.json"));
            var firstPath = Path.Combine(root, "first");
            var secondPath = Path.Combine(root, "second");
            store.Add("First", firstPath);
            store.Add("Second", secondPath);
            var entries = store.Add("First renamed", firstPath);

            Assert.Equal(2, entries.Count);
            Assert.Equal("First renamed", entries[0].Name);
            Assert.Equal(Path.GetFullPath(firstPath), entries[0].ProjectDirectory);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }
}
