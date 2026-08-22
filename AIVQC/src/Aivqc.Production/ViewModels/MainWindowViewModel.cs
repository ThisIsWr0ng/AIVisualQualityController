using Aivqc.Core.Diagnostics;
using Aivqc.Core.Deployment;
using Aivqc.Core.Inspection;
using Aivqc.Production.Services;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Production.ViewModels;

public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
    private OnnxInspectionSession? _inspectionSession;
    private CancellationTokenSource? _inspectionCancellation;
    private IReadOnlyDictionary<int, float>? _packageClassThresholds;
    private bool _disposed;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(InspectionState))]
    [NotifyPropertyChangedFor(nameof(InspectionAction))]
    private bool _isInspectionRunning;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(InspectionState))]
    private bool _hasModel;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(InspectionState))]
    private bool _hasImage;

    [ObservableProperty]
    private string _inspectionStatusMessage = "Load an AIVQC ONNX model and select an image to begin.";

    [ObservableProperty]
    private string _modelName = "No model loaded";

    [ObservableProperty]
    private string _modelInput = "—";

    [ObservableProperty]
    private string _classSummary = "classes.json will be loaded next to the model";

    [ObservableProperty]
    private string _packageSummary = "Development model · no deployment package";

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(CanEditThreshold))]
    private bool _areThresholdsLocked;

    [ObservableProperty]
    private string _imageName = "No inspection image selected";

    [ObservableProperty]
    private string _imageResolution = "—";

    [ObservableProperty]
    private Bitmap? _previewImage;

    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(ThresholdDisplay))]
    private double _confidenceThreshold = 50;

    [ObservableProperty]
    private string _lastResult = "—";

    [ObservableProperty]
    private string _lastConfidence = "—";

    [ObservableProperty]
    private string _lastLatency = "— ms";

    [ObservableProperty]
    private IReadOnlyList<string> _detectedDefects = [];

    [ObservableProperty]
    private int _inspectedCount;

    [ObservableProperty]
    private int _okCount;

    [ObservableProperty]
    private int _nokCount;

    [ObservableProperty]
    private int _errorCount;

    public string InspectionAction => IsInspectionRunning ? "Cancel inspection" : "Run inspection";

    public string InspectionState => IsInspectionRunning
        ? "RUNNING"
        : !HasModel
            ? "MODEL REQUIRED"
            : !HasImage
                ? "IMAGE REQUIRED"
                : "READY";

    public string ThresholdDisplay => $"{ConfidenceThreshold:0}%";

    public bool CanEditThreshold => !AreThresholdsLocked;

    public bool HasNoDetectedDefects => DetectedDefects.Count == 0;

    public string OkPercentage => InspectedCount == 0
        ? "0.0%"
        : $"{100d * OkCount / InspectedCount:0.0}%";

    public string NokPercentage => InspectedCount == 0
        ? "0.0%"
        : $"{100d * NokCount / InspectedCount:0.0}%";

    public string VersionDisplay { get; } =
        $"v{ApplicationVersion.DisplayFromAssembly(typeof(MainWindowViewModel).Assembly)}";

    public async Task LoadModelAsync(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (IsInspectionRunning)
        {
            InspectionStatusMessage = "Cancel the active inspection before changing the model.";
            return;
        }

        InspectionStatusMessage = "Validating the ONNX model contract…";

        try
        {
            var newSession = await Task.Run(() => new OnnxInspectionSession(filePath));
            var previousSession = _inspectionSession;
            _inspectionSession = newSession;
            previousSession?.Dispose();
            _packageClassThresholds = null;

            var information = newSession.Information;
            ModelName = information.FileName;
            ModelInput = $"Float · 1 × 3 × {information.InputHeight} × {information.InputWidth}";
            ClassSummary = information.ClassNames.Count == 0
                ? "No classes.json found · numeric labels will be used"
                : string.Join(", ", information.ClassNames.OrderBy(item => item.Key).Select(item => item.Value));
            PackageSummary = "Development model · integrity is not package-verified";
            AreThresholdsLocked = false;
            HasModel = true;
            LastResult = "—";
            DetectedDefects = [];
            InspectionStatusMessage = HasImage
                ? "Model and image are ready for inspection."
                : "Model loaded. Select an image to inspect.";
        }
        catch (Exception exception)
        {
            InspectionStatusMessage = $"Model load failed: {exception.Message}";
        }
    }

    public async Task LoadDeploymentPackageAsync(string packagePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (IsInspectionRunning)
        {
            InspectionStatusMessage = "Cancel the active inspection before changing the package.";
            return;
        }

        InspectionStatusMessage = "Verifying and importing the deployment package…";

        try
        {
            var cacheRoot = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "AIVQC",
                "Production",
                "Packages");
            var loaded = await Task.Run(() =>
            {
                var imported = DeploymentPackageArchive.Import(packagePath, cacheRoot);
                var classNames = imported.Manifest.DefectClasses.ToDictionary(item => item.Id, item => item.Name);
                var session = new OnnxInspectionSession(imported.ModelPath, classNames);

                if (session.Information.InputWidth != imported.Manifest.Model.InputWidth
                    || session.Information.InputHeight != imported.Manifest.Model.InputHeight)
                {
                    session.Dispose();
                    throw new InvalidDataException(
                        "The ONNX input dimensions do not match the deployment-package manifest.");
                }

                return (Imported: imported, Session: session);
            });

            var previousSession = _inspectionSession;
            _inspectionSession = loaded.Session;
            previousSession?.Dispose();

            var manifest = loaded.Imported.Manifest;
            _packageClassThresholds = manifest.DefectClasses.ToDictionary(item => item.Id, item => item.Threshold);
            ConfidenceThreshold = 100d * manifest.DefectClasses.Min(item => item.Threshold);
            ModelName = loaded.Session.Information.FileName;
            ModelInput = $"Float · 1 × 3 × {manifest.Model.InputHeight} × {manifest.Model.InputWidth}";
            ClassSummary = string.Join(
                ", ",
                manifest.DefectClasses
                    .OrderBy(item => item.Id)
                    .Select(item => $"{item.Name} {item.Threshold:P0}"));
            PackageSummary = $"{manifest.ProductId} · {manifest.RecipeId} · {manifest.PackageId:N}";
            AreThresholdsLocked = true;
            HasModel = true;
            LastResult = "—";
            DetectedDefects = [];
            InspectionStatusMessage = HasImage
                ? "Verified package and image are ready for inspection."
                : "Verified package loaded. Select an image to inspect.";
        }
        catch (Exception exception)
        {
            InspectionStatusMessage = $"Package import failed: {exception.Message}";
        }
    }

    public void LoadImage(string filePath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (IsInspectionRunning)
        {
            InspectionStatusMessage = "Cancel the active inspection before changing the image.";
            return;
        }

        try
        {
            var bitmap = new Bitmap(filePath);
            PreviewImage = bitmap;
            ImageName = Path.GetFileName(filePath);
            ImageResolution = $"{bitmap.PixelSize.Width} × {bitmap.PixelSize.Height}";
            SelectedImagePath = Path.GetFullPath(filePath);
            HasImage = true;
            LastResult = "—";
            LastConfidence = "—";
            LastLatency = "— ms";
            DetectedDefects = [];
            InspectionStatusMessage = HasModel
                ? "Model and image are ready for inspection."
                : "Image loaded. Select an AIVQC ONNX model.";
        }
        catch (Exception exception)
        {
            InspectionStatusMessage = $"Image load failed: {exception.Message}";
        }
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _inspectionCancellation?.Cancel();
        _inspectionCancellation?.Dispose();
        _inspectionSession?.Dispose();
        PreviewImage?.Dispose();
        _disposed = true;
    }

    [RelayCommand]
    private async Task ToggleInspectionAsync()
    {
        if (IsInspectionRunning)
        {
            InspectionStatusMessage = "Cancelling inspection…";
            _inspectionCancellation?.Cancel();
            return;
        }

        if (_inspectionSession is null || !HasModel)
        {
            InspectionStatusMessage = "Load a compatible AIVQC ONNX model first.";
            return;
        }

        if (!HasImage || string.IsNullOrWhiteSpace(SelectedImagePath))
        {
            InspectionStatusMessage = "Select an image to inspect first.";
            return;
        }

        IsInspectionRunning = true;
        InspectionStatusMessage = "Preparing the image and running ONNX inference…";
        _inspectionCancellation = new CancellationTokenSource();

        try
        {
            var result = _packageClassThresholds is null
                ? await _inspectionSession.InspectAsync(
                    SelectedImagePath,
                    (float)(ConfidenceThreshold / 100d),
                    _inspectionCancellation.Token)
                : await _inspectionSession.InspectAsync(
                    SelectedImagePath,
                    _packageClassThresholds,
                    (float)(ConfidenceThreshold / 100d),
                    _inspectionCancellation.Token);

            using var stream = new MemoryStream(result.AnnotatedImagePng, writable: false);
            PreviewImage = new Bitmap(stream);
            ImageResolution = $"{result.SourceWidth} × {result.SourceHeight}";
            LastLatency = $"{result.InferenceDuration.TotalMilliseconds:0.0} ms";
            LastResult = result.Decision.State.ToString().ToUpperInvariant();
            LastConfidence = result.Decision.RejectedDetections.Count == 0
                ? "—"
                : result.Decision.RejectedDetections.Max(item => item.Confidence).ToString("P1");
            DetectedDefects = result.Decision.RejectedDetections
                .Select(item => $"{item.ClassName} · {item.Confidence:P1}")
                .ToArray();
            InspectionStatusMessage = result.Decision.Reason;

            InspectedCount++;
            switch (result.Decision.State)
            {
                case InspectionDecisionState.Ok:
                    OkCount++;
                    break;
                case InspectionDecisionState.Nok:
                    NokCount++;
                    break;
                default:
                    ErrorCount++;
                    break;
            }

            OnPropertyChanged(nameof(OkPercentage));
            OnPropertyChanged(nameof(NokPercentage));
        }
        catch (OperationCanceledException)
        {
            InspectionStatusMessage = "Inspection cancelled.";
        }
        catch (Exception exception)
        {
            LastResult = "ERROR";
            LastConfidence = "—";
            InspectedCount++;
            ErrorCount++;
            InspectionStatusMessage = $"Inspection failed: {exception.Message}";
            OnPropertyChanged(nameof(OkPercentage));
            OnPropertyChanged(nameof(NokPercentage));
        }
        finally
        {
            _inspectionCancellation.Dispose();
            _inspectionCancellation = null;
            IsInspectionRunning = false;
        }
    }

    private string SelectedImagePath { get; set; } = string.Empty;

    partial void OnPreviewImageChanging(Bitmap? value)
    {
        if (!ReferenceEquals(PreviewImage, value))
        {
            PreviewImage?.Dispose();
        }
    }

    partial void OnDetectedDefectsChanged(IReadOnlyList<string> value)
    {
        OnPropertyChanged(nameof(HasNoDetectedDefects));
    }
}
