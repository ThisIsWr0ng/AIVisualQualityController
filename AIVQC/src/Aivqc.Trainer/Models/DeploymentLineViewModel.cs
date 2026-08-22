using CommunityToolkit.Mvvm.ComponentModel;

namespace Aivqc.Trainer.Models;

public partial class DeploymentLineViewModel : ObservableObject
{
    public DeploymentLineViewModel(
        string id,
        string name,
        IEnumerable<string>? productIds = null,
        int deploymentCount = 0,
        DateTimeOffset? lastDeploymentUtc = null)
    {
        Id = id;
        _name = name;
        _productIds = productIds?.Distinct(StringComparer.OrdinalIgnoreCase).ToArray() ?? [];
        _deploymentCount = deploymentCount;
        _lastDeploymentUtc = lastDeploymentUtc;
    }

    public string Id { get; }

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DisplayName))]
    private string _name;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(AvailableProductsDisplay))]
    private IReadOnlyList<string> _productIds;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(DeploymentCountDisplay))]
    private int _deploymentCount;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(LastDeploymentDisplay))]
    private DateTimeOffset? _lastDeploymentUtc;

    public string DisplayName => $"{Name}  ·  {Id}";

    public string AvailableProductsDisplay => ProductIds.Count == 0
        ? "No projects assigned"
        : string.Join(", ", ProductIds);

    public string DeploymentCountDisplay => DeploymentCount.ToString("N0");

    public string LastDeploymentDisplay => LastDeploymentUtc is null
        ? "Never"
        : LastDeploymentUtc.Value.ToLocalTime().ToString("g");

    public DeploymentLineViewModel Clone() => new(
        Id,
        Name,
        ProductIds,
        DeploymentCount,
        LastDeploymentUtc);

    public void AssignProduct(string productId)
    {
        if (string.IsNullOrWhiteSpace(productId)
            || ProductIds.Contains(productId, StringComparer.OrdinalIgnoreCase))
        {
            return;
        }

        ProductIds = [.. ProductIds, productId.Trim()];
    }

    public void RemoveProduct(string productId)
    {
        ProductIds = ProductIds
            .Where(item => !string.Equals(item, productId, StringComparison.OrdinalIgnoreCase))
            .ToArray();
    }

    public void RecordDeployment()
    {
        DeploymentCount++;
        LastDeploymentUtc = DateTimeOffset.UtcNow;
    }
}
