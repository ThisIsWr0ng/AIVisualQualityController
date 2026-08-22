using Aivqc.Trainer.Models;

namespace Aivqc.Trainer.Tests.Models;

public sealed class DeploymentLineViewModelTests
{
    [Fact]
    public void ProductAssignmentsRemainUniqueAndCanBeRemoved()
    {
        var line = new DeploymentLineViewModel("line-1", "Line 1");

        line.AssignProduct("dressing");
        line.AssignProduct("DRESSING");
        line.AssignProduct("gauze");
        line.RemoveProduct("Dressing");

        Assert.Equal(["gauze"], line.ProductIds);
    }

    [Fact]
    public void RecordDeploymentUpdatesStatistics()
    {
        var line = new DeploymentLineViewModel("line-1", "Line 1", deploymentCount: 4);

        line.RecordDeployment();

        Assert.Equal(5, line.DeploymentCount);
        Assert.NotNull(line.LastDeploymentUtc);
    }
}
