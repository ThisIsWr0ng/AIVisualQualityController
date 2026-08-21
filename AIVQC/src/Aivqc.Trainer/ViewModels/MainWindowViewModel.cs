using Aivqc.Core.Diagnostics;
using Aivqc.Core.Training;
using Aivqc.Trainer.Models;
using Aivqc.Trainer.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Trainer.ViewModels;

public partial class MainWindowViewModel : ViewModelBase
{
    private readonly TrainingProcessRunner _trainingProcessRunner = new();
    private CancellationTokenSource? _trainingCancellation;

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

    [ObservableProperty]
    private string _trainingDatasetPath = string.Empty;

    [ObservableProperty]
    private string _pythonExecutable = PythonEnvironmentLocator.FindDefault();

    [ObservableProperty]
    private int _trainingEpochs = 10;

    [ObservableProperty]
    private int _trainingBatchSize = 4;

    [ObservableProperty]
    private double _trainingLearningRate = 0.005;

    [ObservableProperty]
    private int _trainingDeviceIndex;

    [ObservableProperty]
    private bool _usePretrainedBackbone = true;

    [ObservableProperty]
    private bool _isTraining;

    [ObservableProperty]
    private bool _hasTrainingMetrics;

    [ObservableProperty]
    private double _trainingProgress;

    [ObservableProperty]
    private string _trainingStatus = "Select a Pascal VOC dataset to configure training.";

    [ObservableProperty]
    private string _trainingEpochDisplay = "—";

    [ObservableProperty]
    private string _trainingLossDisplay = "—";

    [ObservableProperty]
    private string _trainingMap50Display = "—";

    [ObservableProperty]
    private string _trainingMapDisplay = "—";

    [ObservableProperty]
    private string _trainingPrecisionDisplay = "—";

    [ObservableProperty]
    private string _trainingRecallDisplay = "—";

    [ObservableProperty]
    private string _trainingF1Display = "—";

    [ObservableProperty]
    private string _trainingOutputPath = "—";

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
            ModelImportStatus = "Valid ONNX model loaded for this session.";
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

    public void SelectTrainingDataset(string directoryPath)
    {
        TrainingDatasetPath = directoryPath;
        var trainingDirectory = Path.Combine(directoryPath, "train");
        var validationDirectory = Directory.Exists(Path.Combine(directoryPath, "valid"))
            ? Path.Combine(directoryPath, "valid")
            : Path.Combine(directoryPath, "val");

        if (!Directory.Exists(trainingDirectory) || !Directory.Exists(validationDirectory))
        {
            TrainingStatus = "Dataset must contain train and valid (or val) directories.";
            return;
        }

        var trainingImages = Directory.EnumerateFiles(trainingDirectory, "*.xml").Count();
        var validationImages = Directory.EnumerateFiles(validationDirectory, "*.xml").Count();
        TrainingStatus = trainingImages == 0 || validationImages == 0
            ? "No Pascal VOC XML annotations were found in one of the required splits."
            : $"Dataset ready: {trainingImages} training and {validationImages} validation images.";
    }

    [RelayCommand]
    private async Task StartTrainingAsync()
    {
        if (IsTraining)
        {
            return;
        }

        IsTraining = true;
        HasTrainingMetrics = false;
        TrainingProgress = 0;
        TrainingStatus = "Preparing training job…";
        _trainingCancellation = new CancellationTokenSource();

        try
        {
            var runName = $"training-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}";
            var outputRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "AIVQC",
                "Trainer",
                "Runs");
            var configuration = new TrainingConfiguration(
                PythonExecutable,
                Path.Combine(AppContext.BaseDirectory, "TrainingBackend", "train.py"),
                TrainingDatasetPath,
                outputRoot,
                runName,
                TrainingEpochs,
                TrainingBatchSize,
                TrainingLearningRate,
                Workers: 0,
                Device: GetTrainingDevice(),
                PretrainedBackbone: UsePretrainedBackbone,
                ScoreThreshold: 0.25);
            var progress = new Progress<TrainingJobEvent>(HandleTrainingEvent);
            var result = await _trainingProcessRunner.RunAsync(
                configuration,
                progress,
                _trainingCancellation.Token);

            TrainingProgress = 100;
            TrainingStatus = "Training, evaluation and ONNX export completed.";
            TrainingOutputPath = result.RunDirectory;
            SetMetricDisplays(result.Map50, result.Map50To95, result.Precision, result.Recall, result.F1);
            HasTrainingMetrics = true;
            ActivityMessage = $"Training completed. ONNX model saved to {result.OnnxPath}.";

            if (File.Exists(result.OnnxPath))
            {
                await ImportOnnxAsync(result.OnnxPath);
            }
        }
        catch (OperationCanceledException)
        {
            TrainingStatus = "Training cancelled. The incomplete run was retained for diagnostics.";
            ActivityMessage = "Training was cancelled.";
        }
        catch (Exception exception)
        {
            TrainingStatus = $"Training failed: {exception.Message}";
            ActivityMessage = "Training could not be completed. Review the training status.";
        }
        finally
        {
            _trainingCancellation.Dispose();
            _trainingCancellation = null;
            IsTraining = false;
        }
    }

    [RelayCommand]
    private void CancelTraining()
    {
        if (IsTraining)
        {
            TrainingStatus = "Stopping the training process…";
            _trainingCancellation?.Cancel();
        }
    }

    private void HandleTrainingEvent(TrainingJobEvent trainingEvent)
    {
        switch (trainingEvent.Type)
        {
            case "started":
                TrainingStatus = $"Training started on {trainingEvent.Device}.";
                break;
            case "epoch" when trainingEvent.Epoch is not null && trainingEvent.Epochs is not null:
                var precision = trainingEvent.Precision ?? 0;
                var recall = trainingEvent.Recall ?? 0;
                var f1 = precision + recall == 0
                    ? 0
                    : 2 * precision * recall / (precision + recall);
                TrainingProgress = 100d * trainingEvent.Epoch.Value / trainingEvent.Epochs.Value;
                TrainingEpochDisplay = $"{trainingEvent.Epoch}/{trainingEvent.Epochs}";
                TrainingLossDisplay = trainingEvent.TrainLoss?.ToString("0.0000") ?? "—";
                SetMetricDisplays(
                    trainingEvent.Map50 ?? 0,
                    trainingEvent.Map50To95 ?? 0,
                    precision,
                    recall,
                    f1);
                TrainingStatus = $"Epoch {TrainingEpochDisplay} completed.";
                break;
            case "exporting":
                TrainingStatus = "Evaluation completed. Exporting the best checkpoint to ONNX…";
                break;
            case "failed":
                TrainingStatus = $"Training failed: {trainingEvent.Message}";
                break;
        }
    }

    private string GetTrainingDevice()
    {
        return TrainingDeviceIndex switch
        {
            1 => "cpu",
            2 => "gpu",
            _ => "auto",
        };
    }

    private void SetMetricDisplays(double map50, double map50To95, double precision, double recall, double f1)
    {
        TrainingMap50Display = map50.ToString("P1");
        TrainingMapDisplay = map50To95.ToString("P1");
        TrainingPrecisionDisplay = precision.ToString("P1");
        TrainingRecallDisplay = recall.ToString("P1");
        TrainingF1Display = f1.ToString("P1");
    }

    private static string FormatFileSize(long bytes)
    {
        const double bytesPerMegabyte = 1024d * 1024d;
        return $"{bytes / bytesPerMegabyte:0.00} MB";
    }
}
