using System.Collections.ObjectModel;
using Aivqc.Core.Connectivity;
using Aivqc.Trainer.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Trainer.ViewModels;

public partial class SettingsWindowViewModel : ViewModelBase
{
    private readonly MainWindowViewModel _mainWindow;

    public SettingsWindowViewModel(MainWindowViewModel mainWindow)
    {
        _mainWindow = mainWindow;
        ConnectionModeIndex = mainWindow.ConnectionModeIndex;
        ConnectionEndpoint = mainWindow.ConnectionEndpoint;
        ConnectionClientId = mainWindow.ConnectionClientId;
        ConnectionApiKey = mainWindow.ConnectionApiKey;
        AllowInsecureConnection = mainWindow.AllowInsecureConnection;
        ExpertModeEnabled = mainWindow.ExpertModeEnabled;
        Lines = new ObservableCollection<DeploymentLineViewModel>(
            mainWindow.DeploymentLines.Select(line => line.Clone()));
        SelectedLine = Lines.FirstOrDefault(line => string.Equals(
            line.Id,
            mainWindow.SelectedDeploymentLine?.Id,
            StringComparison.OrdinalIgnoreCase))
            ?? Lines.FirstOrDefault();
    }

    public ObservableCollection<DeploymentLineViewModel> Lines { get; }

    public string CurrentProjectId => _mainWindow.IsProjectLoaded
        ? _mainWindow.DeploymentProductId
        : "No project open";

    [ObservableProperty]
    private int _selectedSectionIndex;

    [ObservableProperty]
    private int _connectionModeIndex;

    [ObservableProperty]
    private string _connectionEndpoint = string.Empty;

    [ObservableProperty]
    private string _connectionClientId = string.Empty;

    [ObservableProperty]
    private string _connectionApiKey = string.Empty;

    [ObservableProperty]
    private bool _allowInsecureConnection;

    [ObservableProperty]
    private bool _expertModeEnabled;

    [ObservableProperty]
    private DeploymentLineViewModel? _selectedLine;

    [ObservableProperty]
    private string _newLineName = string.Empty;

    [ObservableProperty]
    private string _settingsStatus = "Changes are applied after saving settings.";

    [ObservableProperty]
    private bool _isBusy;

    public event EventHandler? SaveRequested;

    public event EventHandler? CancelRequested;

    [RelayCommand]
    private async Task TestConnectionAsync()
    {
        if (IsBusy)
        {
            return;
        }

        IsBusy = true;
        SettingsStatus = "Testing connection…";
        try
        {
            var settings = CreateConnectionSettings();
            using var client = new AivqcServerApiClient(settings, ConnectionApiKey);
            var info = await client.GetInfoAsync();
            SettingsStatus = $"Connected to {info.Name} · API {info.ApiVersion} · server {info.ServerVersion}.";
        }
        catch (Exception exception)
        {
            SettingsStatus = $"Connection failed: {exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private async Task RegisterSelectedLineAsync()
    {
        if (SelectedLine is null || IsBusy)
        {
            return;
        }

        IsBusy = true;
        SettingsStatus = $"Registering {SelectedLine.Name}…";
        try
        {
            var settings = CreateConnectionSettings(SelectedLine.Id);
            using var client = new AivqcServerApiClient(settings, ConnectionApiKey);
            await client.RegisterStationAsync(SelectedLine.Id, SelectedLine.Name);
            SettingsStatus = $"Line '{SelectedLine.Name}' registered on the endpoint.";
        }
        catch (Exception exception)
        {
            SettingsStatus = $"Line registration failed: {exception.Message}";
        }
        finally
        {
            IsBusy = false;
        }
    }

    [RelayCommand]
    private void AddLine()
    {
        var name = NewLineName.Trim();
        if (string.IsNullOrWhiteSpace(name))
        {
            SettingsStatus = "Enter a line name first.";
            return;
        }

        var baseId = CreateIdentifier(name);
        var id = baseId;
        var suffix = 2;
        while (Lines.Any(line => string.Equals(line.Id, id, StringComparison.OrdinalIgnoreCase)))
        {
            id = $"{baseId}-{suffix++}";
        }

        var line = new DeploymentLineViewModel(id, name);
        Lines.Add(line);
        SelectedLine = line;
        NewLineName = string.Empty;
        SettingsStatus = $"Added {name}. Save settings to keep it.";
    }

    [RelayCommand]
    private void RemoveSelectedLine()
    {
        if (SelectedLine is null || Lines.Count == 1)
        {
            SettingsStatus = "At least one production line must remain available.";
            return;
        }

        var index = Lines.IndexOf(SelectedLine);
        Lines.Remove(SelectedLine);
        SelectedLine = Lines[Math.Clamp(index, 0, Lines.Count - 1)];
        SettingsStatus = "Line removed from this draft. Save settings to apply.";
    }

    [RelayCommand]
    private void AssignCurrentProject()
    {
        if (SelectedLine is null || !_mainWindow.IsProjectLoaded)
        {
            SettingsStatus = "Open a project before assigning it to a production line.";
            return;
        }

        SelectedLine.AssignProduct(_mainWindow.DeploymentProductId);
        SettingsStatus = $"Assigned {_mainWindow.DeploymentProductId} to {SelectedLine.Name}.";
    }

    [RelayCommand]
    private void RemoveCurrentProject()
    {
        if (SelectedLine is null || !_mainWindow.IsProjectLoaded)
        {
            return;
        }

        SelectedLine.RemoveProduct(_mainWindow.DeploymentProductId);
        SettingsStatus = $"Removed {_mainWindow.DeploymentProductId} from {SelectedLine.Name}.";
    }

    [RelayCommand]
    private void Save()
    {
        try
        {
            _mainWindow.ApplySettings(this);
            SaveRequested?.Invoke(this, EventArgs.Empty);
        }
        catch (Exception exception)
        {
            SettingsStatus = $"Settings could not be saved: {exception.Message}";
        }
    }

    [RelayCommand]
    private void Cancel() => CancelRequested?.Invoke(this, EventArgs.Empty);

    public AivqcConnectionSettings CreateConnectionSettings(string? stationId = null)
    {
        if (!Uri.TryCreate(ConnectionEndpoint, UriKind.Absolute, out var endpoint))
        {
            throw new InvalidOperationException("Enter a valid absolute connection URL.");
        }

        var selectedStationId = stationId ?? SelectedLine?.Id;
        var settings = new AivqcConnectionSettings(
            ConnectionModeIndex == 1 ? AivqcConnectionMode.Direct : AivqcConnectionMode.Server,
            endpoint,
            ConnectionClientId.Trim(),
            selectedStationId,
            AllowInsecureConnection);
        settings.Validate();
        return settings;
    }

    private static string CreateIdentifier(string value)
    {
        var characters = value.Trim().ToLowerInvariant()
            .Select(character => char.IsAsciiLetterOrDigit(character) ? character : '-')
            .ToArray();
        var identifier = string.Join('-', new string(characters)
            .Split('-', StringSplitOptions.RemoveEmptyEntries));
        return string.IsNullOrWhiteSpace(identifier) ? "line" : identifier;
    }
}
