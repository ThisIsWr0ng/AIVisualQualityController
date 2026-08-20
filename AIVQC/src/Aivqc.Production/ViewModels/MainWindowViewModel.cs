using Aivqc.Core.Diagnostics;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Production.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(InspectionState))]
    [NotifyPropertyChangedFor(nameof(InspectionAction))]
    [NotifyPropertyChangedFor(nameof(InspectionStatusMessage))]
    private bool _isInspectionRunning;

    public string InspectionAction => IsInspectionRunning ? "Stop inspection" : "Start inspection";

    public string InspectionState => IsInspectionRunning ? "RUNNING" : "READY";

    public string InspectionStatusMessage => IsInspectionRunning
        ? "Inspection is active. Waiting for the next product."
        : "System checks passed. Start when the line is ready.";

    public string VersionDisplay { get; } =
        $"v{ApplicationVersion.DisplayFromAssembly(typeof(MainWindowViewModel).Assembly)}";

    [RelayCommand]
    private void ToggleInspection()
    {
        IsInspectionRunning = !IsInspectionRunning;
    }
}
