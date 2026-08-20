using Aivqc.Core.Diagnostics;
using Aivqc.Trainer.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Trainer.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    [ObservableProperty]
    private string _activityMessage = "Create a project or open an existing workspace to begin.";

    [ObservableProperty]
    private string _projectName = "Medical dressing inspection";

    [ObservableProperty]
    private bool _hasImportedModel;

    [ObservableProperty]
    private bool _isModelImporting;

    [ObservableProperty]
    private string _modelImportStatus = "No external model imported.";

    [ObservableProperty]
    private string _modelName = "—";

    [ObservableProperty]
    private string _modelSize = "—";

    [ObservableProperty]
    private string _modelInput = "—";

    [ObservableProperty]
    private string _modelOutput = "—";

    [ObservableProperty]
    private string _modelSha256 = "—";

    public string VersionDisplay { get; } =
        $"v{ApplicationVersion.DisplayFromAssembly(typeof(MainWindowViewModel).Assembly)}";

    [RelayCommand]
    private void CreateProject()
    {
        ProjectName = "Untitled inspection project";
        ActivityMessage = "New project created. Configure the image source to continue.";
    }

    [RelayCommand]
    private void OpenProject()
    {
        ActivityMessage = "Project picker integration is the next implementation step.";
    }

    [RelayCommand]
    private void ContinueWorkflow()
    {
        ActivityMessage = "Image-source setup selected. Camera and file import will be connected next.";
    }

    public async Task ImportOnnxAsync(string filePath)
    {
        IsModelImporting = true;
        ModelImportStatus = "Validating model with ONNX Runtime…";

        try
        {
            var summary = await OnnxModelInspector.InspectAsync(filePath);

            ModelName = summary.FileName;
            ModelSize = FormatFileSize(summary.FileSizeBytes);
            ModelInput = string.Join(Environment.NewLine, summary.Inputs);
            ModelOutput = string.Join(Environment.NewLine, summary.Outputs);
            ModelSha256 = summary.Sha256;
            ModelImportStatus = "Valid ONNX model loaded into the current workspace.";
            ActivityMessage = $"Imported and validated {summary.FileName}.";
            HasImportedModel = true;
        }
        catch (Exception exception)
        {
            HasImportedModel = false;
            ModelImportStatus = $"Import failed: {exception.Message}";
            ActivityMessage = "The selected ONNX file could not be imported.";
        }
        finally
        {
            IsModelImporting = false;
        }
    }

    private static string FormatFileSize(long bytes)
    {
        const double bytesPerMegabyte = 1024d * 1024d;
        return $"{bytes / bytesPerMegabyte:0.00} MB";
    }
}
