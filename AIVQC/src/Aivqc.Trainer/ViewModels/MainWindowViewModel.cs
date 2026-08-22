using Aivqc.Core.Diagnostics;
using Aivqc.Core.Deployment;
using Aivqc.Core.Connectivity;
using Aivqc.Core.Projects;
using Aivqc.Core.Training;
using Aivqc.Trainer.Models;
using Aivqc.Trainer.Services;
using Avalonia.Media.Imaging;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace Aivqc.Trainer.ViewModels;

public partial class MainWindowViewModel : ViewModelBase, IDisposable
{
    private readonly TrainingProcessRunner _trainingProcessRunner = new();
    private readonly RecentProjectStore _recentProjectStore = new();
    private readonly SemaphoreSlim _projectWriteLock = new(1, 1);
    private CancellationTokenSource? _trainingCancellation;
    private CancellationTokenSource? _autosaveCancellation;
    private TrainerProjectManifest? _project;
    private string _projectDirectory = string.Empty;
    private bool _isApplyingProject;
    private bool _disposed;
    private string _importedModelPath = string.Empty;
    private int _importedModelInputWidth;
    private int _importedModelInputHeight;
    private IReadOnlyDictionary<string, int> _importedClassNames = new Dictionary<string, int>();
    private string _lastExportedPackagePath = string.Empty;

    [ObservableProperty]
    private string _activityMessage = "Create a project or open an existing workspace to begin.";

    [ObservableProperty]
    private string _projectName = "Medical dressing inspection";

    [ObservableProperty]
    private bool _isProjectLoaded;

    [ObservableProperty]
    private string _projectPath = "No project workspace is open.";

    [ObservableProperty]
    private string _projectHealth = "Create or open a project to start importing images.";

    [ObservableProperty]
    private IReadOnlyList<RecentProjectViewModel> _recentProjects = [];

    [ObservableProperty]
    private IReadOnlyList<ProjectImageViewModel> _projectImages = [];

    [ObservableProperty]
    private int _projectImageCount;

    [ObservableProperty]
    private int _missingImageCount;

    [ObservableProperty]
    private int _imageWarningCount;

    [ObservableProperty]
    private int _imageImportModeIndex;

    [ObservableProperty]
    private bool _isImportingImages;

    [ObservableProperty]
    private string _imageImportStatus = "Open a project before importing images.";

    [ObservableProperty]
    private ProjectImageViewModel? _selectedProjectImage;

    [ObservableProperty]
    private string? _selectedImagePath;

    [ObservableProperty]
    private IReadOnlyList<ProjectObjectAnnotation> _selectedImageAnnotations = [];

    [ObservableProperty]
    private IReadOnlyList<AnnotationListItemViewModel> _annotationListItems = [];

    [ObservableProperty]
    private Guid _selectedAnnotationId;

    [ObservableProperty]
    private IReadOnlyList<string> _defectClasses = [];

    [ObservableProperty]
    private string? _selectedDefectClass;

    [ObservableProperty]
    private string _newDefectClassName = string.Empty;

    [ObservableProperty]
    private string _annotationStatus = "Import images and add a defect class to begin annotation.";

    [ObservableProperty]
    private int _defectClassCount;

    [ObservableProperty]
    private int _annotationCount;

    [ObservableProperty]
    private int _annotatedImageCount;

    [ObservableProperty]
    private string _annotationProgressDisplay = "0%";

    [ObservableProperty]
    private string _datasetReadinessMessage = "Import the first images to unlock annotation and training.";

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
    private bool _isPreparingProjectDataset;

    [ObservableProperty]
    private string _projectDatasetStatus = "Prepare a dataset snapshot from the current project annotations.";

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

    [ObservableProperty]
    private string _deploymentProductId = "medical-dressing";

    [ObservableProperty]
    private string _deploymentRecipeId = "standard";

    [ObservableProperty]
    private string _deploymentAuthor = Environment.UserName;

    [ObservableProperty]
    private double _deploymentThreshold = 50;

    [ObservableProperty]
    private bool _isExportingPackage;

    [ObservableProperty]
    private string _packageExportStatus = "Configure deployment metadata after loading a model.";

    [ObservableProperty]
    private int _connectionModeIndex;

    [ObservableProperty]
    private string _connectionEndpoint = "https://aivqc.local/";

    [ObservableProperty]
    private string _connectionClientId = "trainer-main";

    [ObservableProperty]
    private string _connectionStationId = "line-1";

    [ObservableProperty]
    private string _connectionStationName = "Production line 1";

    [ObservableProperty]
    private string _connectionApiKey = string.Empty;

    [ObservableProperty]
    private bool _allowInsecureConnection;

    [ObservableProperty]
    private bool _isConnectionBusy;

    [ObservableProperty]
    private string _connectionStatus = "Configure AIVQC Server or a compatible direct Production endpoint.";

    public string VersionDisplay { get; } =
        $"v{ApplicationVersion.DisplayFromAssembly(typeof(MainWindowViewModel).Assembly)}";

    public MainWindowViewModel()
    {
        RefreshRecentProjects();
        TryLoadConnectionProfile();
    }

    public async Task CreateProjectAsync(string parentDirectory)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(parentDirectory);

        try
        {
            var name = string.IsNullOrWhiteSpace(ProjectName)
                ? "Untitled inspection project"
                : ProjectName.Trim();
            var directory = CreateUniqueProjectDirectory(parentDirectory, CreateSlug(name));
            var productId = CreateSlug(name);
            var project = await Task.Run(() => TrainerProjectStore.Create(directory, name, productId));
            ApplyProject(directory, project);
            ActivityMessage = $"Created project {project.Name}.";
            ImageImportStatus = "Project ready. Import JPG, PNG, BMP, or WebP images.";
        }
        catch (Exception exception)
        {
            ActivityMessage = $"Project creation failed: {exception.Message}";
        }
    }

    public async Task OpenProjectAsync(string projectPath)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectPath);

        try
        {
            var manifestPath = TrainerProjectStore.ResolveManifestPath(projectPath);
            var directory = Path.GetDirectoryName(manifestPath)!;
            var project = await Task.Run(() => TrainerProjectStore.Load(manifestPath));
            ApplyProject(directory, project);
            ActivityMessage = $"Opened project {project.Name}.";
        }
        catch (Exception exception)
        {
            ActivityMessage = $"Project open failed: {exception.Message}";
        }
    }

    [RelayCommand]
    private Task OpenRecentProjectAsync(string projectDirectory) => OpenProjectAsync(projectDirectory);

    [RelayCommand]
    private void ContinueWorkflow()
    {
        ActivityMessage = IsProjectLoaded
            ? "Use Import images to build the project dataset."
            : "Create or open a project before configuring its dataset.";
    }

    public async Task ImportImagesAsync(IReadOnlyList<string> filePaths)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_project is null || !IsProjectLoaded)
        {
            ImageImportStatus = "Create or open a project before importing images.";
            return;
        }

        if (filePaths.Count == 0 || IsImportingImages)
        {
            return;
        }

        IsImportingImages = true;
        ImageImportStatus = $"Validating {filePaths.Count} selected image(s)…";
        _autosaveCancellation?.Cancel();

        ImageImportResult? result = null;
        try
        {
            var storageMode = ImageImportModeIndex == 1
                ? ImageStorageMode.Reference
                : ImageStorageMode.Copy;
            var projectSnapshot = _project;
            result = await Task.Run(() => ProjectImageImporter.Import(
                projectSnapshot,
                _projectDirectory,
                filePaths,
                storageMode));

            var updatedProject = projectSnapshot with
            {
                UpdatedAtUtc = DateTimeOffset.UtcNow,
                Images = projectSnapshot.Images.Concat(result.ImportedImages).ToArray(),
            };
            await SaveProjectAsync(updatedProject);
            RefreshProjectImages();
            InvalidatePreparedProjectDataset();

            var failed = result.Issues.Count - result.DuplicateCount;
            ImageImportStatus = $"Imported {result.ImportedImages.Count}; "
                + $"duplicates skipped: {result.DuplicateCount}; failed: {Math.Max(0, failed)}.";
            ActivityMessage = result.Issues.Count == 0
                ? "Image import completed successfully."
                : result.Issues.First().Message;
        }
        catch (Exception exception)
        {
            if (result is not null)
            {
                RollBackImportedFiles(result.ImportedImages);
            }

            ImageImportStatus = $"Image import failed: {exception.Message}";
            ActivityMessage = "No imported image metadata was committed after the failure.";
        }
        finally
        {
            IsImportingImages = false;
        }
    }

    [RelayCommand]
    private async Task AddDefectClassAsync()
    {
        if (_project is null)
        {
            AnnotationStatus = "Open a project before adding defect classes.";
            return;
        }

        try
        {
            var updated = TrainerProjectAnnotations.AddClass(_project, NewDefectClassName);
            await SaveProjectAsync(updated);
            NewDefectClassName = string.Empty;
            RefreshAnnotationWorkspace();
            InvalidatePreparedProjectDataset();
            SelectedDefectClass = updated.DefectClasses[^1];
            AnnotationStatus = $"Defect class '{SelectedDefectClass}' added and selected.";
        }
        catch (Exception exception)
        {
            AnnotationStatus = $"Class could not be added: {exception.Message}";
        }
    }

    public async Task CreateAnnotationAsync(NormalizedBoundingBox bounds)
    {
        if (_project is null || SelectedProjectImage is null)
        {
            AnnotationStatus = "Select a project image before drawing an annotation.";
            return;
        }

        if (string.IsNullOrWhiteSpace(SelectedDefectClass))
        {
            AnnotationStatus = "Add and select a defect class before drawing a box.";
            return;
        }

        try
        {
            var imageId = SelectedProjectImage.ImageId;
            var updated = TrainerProjectAnnotations.Add(
                _project,
                imageId,
                SelectedDefectClass,
                bounds);
            await SaveProjectAsync(updated);
            RefreshProjectImages(imageId);
            InvalidatePreparedProjectDataset();
            var annotation = updated.Images
                .Single(image => image.ImageId == imageId)
                .Annotations![^1];
            SelectedAnnotationId = annotation.AnnotationId;
            AnnotationStatus = $"Saved {annotation.ClassName} annotation.";
        }
        catch (Exception exception)
        {
            AnnotationStatus = $"Annotation could not be saved: {exception.Message}";
        }
    }

    [RelayCommand]
    private async Task DeleteSelectedAnnotationAsync()
    {
        if (_project is null
            || SelectedProjectImage is null
            || SelectedAnnotationId == Guid.Empty)
        {
            AnnotationStatus = "Select an annotation to delete.";
            return;
        }

        try
        {
            var imageId = SelectedProjectImage.ImageId;
            var updated = TrainerProjectAnnotations.Remove(
                _project,
                imageId,
                SelectedAnnotationId);
            await SaveProjectAsync(updated);
            SelectedAnnotationId = Guid.Empty;
            RefreshProjectImages(imageId);
            InvalidatePreparedProjectDataset();
            AnnotationStatus = "Annotation deleted.";
        }
        catch (Exception exception)
        {
            AnnotationStatus = $"Annotation could not be deleted: {exception.Message}";
        }
    }

    [RelayCommand]
    private void SelectPreviousImage() => SelectRelativeImage(-1);

    [RelayCommand]
    private void SelectNextImage() => SelectRelativeImage(1);

    public void SelectAnnotation(Guid annotationId)
    {
        if (SelectedImageAnnotations.Any(annotation => annotation.AnnotationId == annotationId))
        {
            SelectedAnnotationId = annotationId;
            var annotation = SelectedImageAnnotations.Single(item => item.AnnotationId == annotationId);
            AnnotationStatus = $"Selected {annotation.ClassName} annotation.";
        }
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
            _importedModelPath = summary.SourcePath;
            _importedModelInputWidth = summary.InputWidth;
            _importedModelInputHeight = summary.InputHeight;
            _importedClassNames = summary.ClassNames;
            ModelImportStatus = summary.ClassNames.Count == 0
                ? "Model is valid, but classes.json is missing. Package export is unavailable."
                : $"Valid ONNX model with {summary.ClassNames.Count} defect classes loaded.";
            PackageExportStatus = summary.ClassNames.Count == 0
                ? "Place classes.json next to the ONNX model before exporting."
                : "Model is ready for deployment-package export.";
            ActivityMessage = $"Imported and validated {summary.FileName}.";
            HasImportedModel = true;
        }
        catch (Exception exception)
        {
            HasImportedModel = false;
            _importedModelPath = string.Empty;
            _importedClassNames = new Dictionary<string, int>();
            ModelImportStatus = $"Import failed: {exception.Message}";
            ActivityMessage = "The selected ONNX file could not be imported.";
        }
        finally
        {
            IsModelImporting = false;
        }
    }

    public async Task ExportDeploymentPackageAsync(string outputPath)
    {
        if (IsExportingPackage)
        {
            return;
        }

        IsExportingPackage = true;
        PackageExportStatus = "Creating and verifying deployment package…";

        try
        {
            if (!HasImportedModel || string.IsNullOrWhiteSpace(_importedModelPath))
            {
                throw new InvalidOperationException("Import or train an ONNX model before exporting.");
            }

            if (_importedClassNames.Count == 0)
            {
                throw new InvalidOperationException("The model requires a classes.json file for package export.");
            }

            var threshold = (float)(DeploymentThreshold / 100d);
            var request = new DeploymentPackageExportRequest(
                outputPath,
                _importedModelPath,
                DeploymentProductId,
                DeploymentRecipeId,
                DeploymentAuthor,
                _importedModelInputWidth,
                _importedModelInputHeight,
                new PreprocessingManifest("RGB", "zeroToOne", PreserveAspectRatio: false),
                new RegionOfInterestManifest(0, 0, 0, 0),
                _importedClassNames
                    .OrderBy(item => item.Value)
                    .Select(item => new DefectClassManifest(item.Value, item.Key, threshold))
                    .ToArray());

            var result = await Task.Run(() => DeploymentPackageArchive.Export(request));
            _lastExportedPackagePath = result.PackagePath;
            PackageExportStatus = $"Package exported: {result.PackagePath}";
            ActivityMessage = $"Deployment package {result.Manifest.PackageId:N} is ready for Production.";
        }
        catch (Exception exception)
        {
            PackageExportStatus = $"Package export failed: {exception.Message}";
            ActivityMessage = "The deployment package could not be exported.";
        }
        finally
        {
            IsExportingPackage = false;
        }
    }

    [RelayCommand]
    private async Task TestConnectionAsync()
    {
        await RunConnectionActionAsync(async client =>
        {
            var info = await client.GetInfoAsync();
            SaveConnectionProfile(client.Settings);
            ConnectionStatus = $"Connected to {info.Name} · API {info.ApiVersion} · server {info.ServerVersion}.";
        });
    }

    [RelayCommand]
    private async Task RegisterConnectionStationAsync()
    {
        await RunConnectionActionAsync(async client =>
        {
            await client.RegisterStationAsync(ConnectionStationId, ConnectionStationName);
            SaveConnectionProfile(client.Settings);
            ConnectionStatus = $"Station '{ConnectionStationId}' registered.";
        });
    }

    [RelayCommand]
    private async Task PublishPackageToConnectionAsync()
    {
        if (string.IsNullOrWhiteSpace(_lastExportedPackagePath)
            || !File.Exists(_lastExportedPackagePath))
        {
            ConnectionStatus = "Export a deployment package before publishing it.";
            return;
        }

        await RunConnectionActionAsync(async client =>
        {
            var published = await client.PublishPackageAsync(
                _lastExportedPackagePath,
                ConnectionStationId);
            SaveConnectionProfile(client.Settings);
            ConnectionStatus = $"Published {published.ProductId}/{published.RecipeId} "
                + $"to {published.TargetStationId} · {published.PackageId:D}.";
        });
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
        ProjectDatasetStatus = "Using an externally prepared Pascal VOC dataset.";
    }

    [RelayCommand]
    private async Task PrepareProjectDatasetAsync()
    {
        if (_project is null || IsPreparingProjectDataset)
        {
            ProjectDatasetStatus = "Open a project before preparing its training dataset.";
            return;
        }

        IsPreparingProjectDataset = true;
        ProjectDatasetStatus = "Validating annotations and creating a Pascal VOC snapshot…";
        try
        {
            var projectSnapshot = _project;
            var result = await Task.Run(() => ProjectDatasetExporter.Export(
                projectSnapshot,
                _projectDirectory));
            TrainingDatasetPath = result.DatasetDirectory;
            TrainingStatus = $"Dataset ready: {result.TrainingImageCount} train, "
                + $"{result.ValidationImageCount} valid, {result.TestImageCount} test.";
            ProjectDatasetStatus = $"Snapshot contains {result.AnnotationCount} bounding boxes. "
                + $"Saved in {result.DatasetDirectory}.";
            ActivityMessage = "Project annotations are ready for model training.";
        }
        catch (Exception exception)
        {
            TrainingDatasetPath = string.Empty;
            ProjectDatasetStatus = $"Dataset preparation failed: {exception.Message}";
            TrainingStatus = "Resolve the dataset issue before starting training.";
        }
        finally
        {
            IsPreparingProjectDataset = false;
        }
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
            if (string.IsNullOrWhiteSpace(TrainingDatasetPath) && _project is not null)
            {
                await PrepareProjectDatasetAsync();
            }

            if (string.IsNullOrWhiteSpace(TrainingDatasetPath))
            {
                throw new InvalidOperationException(
                    "Prepare the current project dataset or select an external Pascal VOC dataset.");
            }

            var runName = $"training-{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}";
            var outputRoot = _project is null
                ? Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    "AIVQC",
                    "Trainer",
                    "Runs")
                : Path.Combine(_projectDirectory, "runs");
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

    private async Task RunConnectionActionAsync(Func<AivqcServerApiClient, Task> action)
    {
        if (IsConnectionBusy)
        {
            return;
        }

        IsConnectionBusy = true;
        ConnectionStatus = "Connecting…";
        try
        {
            var settings = CreateConnectionSettings();
            using var client = new AivqcServerApiClient(settings, ConnectionApiKey);
            await action(client);
        }
        catch (Exception exception)
        {
            ConnectionStatus = $"Connection failed: {exception.Message}";
        }
        finally
        {
            IsConnectionBusy = false;
        }
    }

    private AivqcConnectionSettings CreateConnectionSettings()
    {
        if (!Uri.TryCreate(ConnectionEndpoint, UriKind.Absolute, out var endpoint))
        {
            throw new InvalidOperationException("Enter a valid absolute connection URL.");
        }

        return new AivqcConnectionSettings(
            ConnectionModeIndex == 1 ? AivqcConnectionMode.Direct : AivqcConnectionMode.Server,
            endpoint,
            ConnectionClientId,
            ConnectionStationId,
            AllowInsecureConnection);
    }

    private void TryLoadConnectionProfile()
    {
        try
        {
            var profile = AivqcConnectionProfileStore.Load(GetConnectionProfilePath());
            if (profile is null)
            {
                return;
            }

            ConnectionModeIndex = profile.Mode == AivqcConnectionMode.Direct ? 1 : 0;
            ConnectionEndpoint = profile.Endpoint.AbsoluteUri;
            ConnectionClientId = profile.ClientId;
            ConnectionStationId = profile.StationId ?? string.Empty;
            AllowInsecureConnection = profile.AllowInsecureHttp;
        }
        catch (Exception exception)
        {
            ConnectionStatus = $"Connection profile could not be loaded: {exception.Message}";
        }
    }

    private void SaveConnectionProfile(AivqcConnectionSettings settings)
    {
        AivqcConnectionProfileStore.Save(GetConnectionProfilePath(), settings);
    }

    private static string GetConnectionProfilePath() => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "AIVQC",
        "Trainer",
        "connection.json");

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _autosaveCancellation?.Cancel();
        _autosaveCancellation?.Dispose();
        _trainingCancellation?.Cancel();

        if (_project is not null && !string.IsNullOrWhiteSpace(ProjectName))
        {
            try
            {
                var finalProject = _project with
                {
                    Name = ProjectName.Trim(),
                    UpdatedAtUtc = DateTimeOffset.UtcNow,
                };
                TrainerProjectStore.Save(_projectDirectory, finalProject);
            }
            catch
            {
                // The latest successful autosave remains intact if shutdown saving fails.
            }
        }

        DisposeProjectImages();
        _disposed = true;
    }

    partial void OnProjectNameChanged(string value)
    {
        if (!_isApplyingProject && _project is not null)
        {
            ScheduleAutosave();
        }
    }

    partial void OnSelectedProjectImageChanged(ProjectImageViewModel? value)
    {
        SelectedAnnotationId = Guid.Empty;
        RefreshSelectedImage();
    }

    private void ApplyProject(string projectDirectory, TrainerProjectManifest project)
    {
        _autosaveCancellation?.Cancel();
        _projectDirectory = Path.GetFullPath(projectDirectory);
        _project = project;
        _isApplyingProject = true;
        try
        {
            ProjectName = project.Name;
            DeploymentProductId = project.ProductId;
        }
        finally
        {
            _isApplyingProject = false;
        }

        IsProjectLoaded = true;
        ProjectPath = _projectDirectory;
        TrainingDatasetPath = string.Empty;
        ProjectDatasetStatus = "Prepare a dataset snapshot from the current project annotations.";
        TrainingStatus = "Prepare the annotated project dataset before training.";
        RefreshProjectImages();
        RefreshAnnotationWorkspace();
        TryUpdateRecentProjects(project.Name);
        ImageImportStatus = project.Images.Count == 0
            ? "Project ready. Import JPG, PNG, BMP, or WebP images."
            : $"Loaded {project.Images.Count} project image(s).";
    }

    private async Task SaveProjectAsync(TrainerProjectManifest project)
    {
        await _projectWriteLock.WaitAsync();
        try
        {
            await Task.Run(() => TrainerProjectStore.Save(_projectDirectory, project));
            _project = project;
            TryUpdateRecentProjects(project.Name);
        }
        finally
        {
            _projectWriteLock.Release();
        }
    }

    private void ScheduleAutosave()
    {
        _autosaveCancellation?.Cancel();
        _autosaveCancellation?.Dispose();
        _autosaveCancellation = new CancellationTokenSource();
        _ = AutosaveAfterDelayAsync(_autosaveCancellation.Token);
    }

    private async Task AutosaveAfterDelayAsync(CancellationToken cancellationToken)
    {
        try
        {
            await Task.Delay(600, cancellationToken);
            if (_project is null || string.IsNullOrWhiteSpace(ProjectName))
            {
                ProjectHealth = "Project name cannot be empty; changes have not been saved.";
                return;
            }

            var updated = _project with
            {
                Name = ProjectName.Trim(),
                UpdatedAtUtc = DateTimeOffset.UtcNow,
            };
            await SaveProjectAsync(updated);
            ProjectHealth = MissingImageCount == 0
                ? $"Saved automatically · {ProjectImageCount} image(s) available."
                : $"Saved · {MissingImageCount} source image(s) are missing.";
        }
        catch (OperationCanceledException)
        {
            // A newer edit superseded this autosave.
        }
        catch (Exception exception)
        {
            ProjectHealth = $"Autosave failed: {exception.Message}";
        }
    }

    private void RefreshProjectImages(Guid? preferredImageId = null)
    {
        var selectedImageId = preferredImageId ?? SelectedProjectImage?.ImageId;
        DisposeProjectImages();

        if (_project is null)
        {
            ProjectImages = [];
            ProjectImageCount = 0;
            MissingImageCount = 0;
            ImageWarningCount = 0;
            SelectedProjectImage = null;
            RefreshAnnotationWorkspace();
            return;
        }

        var items = new List<ProjectImageViewModel>(_project.Images.Count);
        var missingImages = 0;
        var missingThumbnails = 0;

        foreach (var image in _project.Images)
        {
            if (!File.Exists(TrainerProjectStore.ResolveImagePath(_projectDirectory, image)))
            {
                missingImages++;
            }

            var thumbnailPath = TrainerProjectStore.ResolveThumbnailPath(_projectDirectory, image);
            if (!File.Exists(thumbnailPath))
            {
                missingThumbnails++;
                continue;
            }

            try
            {
                items.Add(new ProjectImageViewModel(
                    image.ImageId,
                    image.SourceFileName,
                    $"{image.Width} × {image.Height} · {image.Format.ToUpperInvariant()}",
                    image.StorageMode == ImageStorageMode.Copy ? "Project copy" : "External reference",
                    image.Warnings.Count == 0 ? "Ready" : string.Join(" ", image.Warnings),
                    $"{image.Annotations?.Count ?? 0} annotation(s)",
                    new Bitmap(thumbnailPath)));
            }
            catch
            {
                missingThumbnails++;
            }
        }

        ProjectImages = items;
        SelectedProjectImage = items.FirstOrDefault(image => image.ImageId == selectedImageId)
            ?? items.FirstOrDefault();
        ProjectImageCount = _project.Images.Count;
        MissingImageCount = missingImages;
        ImageWarningCount = _project.Images.Sum(image => image.Warnings.Count) + missingThumbnails;
        ProjectHealth = missingImages == 0 && missingThumbnails == 0
            ? $"Project healthy · {ProjectImageCount} image(s) available."
            : $"Attention required · missing images: {missingImages}; missing thumbnails: {missingThumbnails}.";
        RefreshAnnotationWorkspace();
    }

    private void RefreshAnnotationWorkspace()
    {
        DefectClasses = _project?.DefectClasses.ToArray() ?? [];
        DefectClassCount = DefectClasses.Count;
        if (SelectedDefectClass is null
            || !DefectClasses.Contains(SelectedDefectClass, StringComparer.OrdinalIgnoreCase))
        {
            SelectedDefectClass = DefectClasses.FirstOrDefault();
        }

        if (_project is null)
        {
            AnnotationCount = 0;
            AnnotatedImageCount = 0;
            AnnotationProgressDisplay = "0%";
            DatasetReadinessMessage = "Import the first images to unlock annotation and training.";
            RefreshSelectedImage();
            return;
        }

        AnnotationCount = _project.Images.Sum(image => image.Annotations?.Count ?? 0);
        AnnotatedImageCount = _project.Images.Count(image => (image.Annotations?.Count ?? 0) > 0);
        var progress = _project.Images.Count == 0
            ? 0
            : 100d * AnnotatedImageCount / _project.Images.Count;
        AnnotationProgressDisplay = $"{progress:0}%";
        DatasetReadinessMessage = _project.Images.Count == 0
            ? "Import the first images to unlock annotation and training."
            : DefectClasses.Count == 0
                ? "Add at least one defect class before annotating images."
                : $"{AnnotatedImageCount} of {_project.Images.Count} images contain annotations.";
        RefreshSelectedImage();
    }

    private void RefreshSelectedImage()
    {
        if (_project is null || SelectedProjectImage is null)
        {
            SelectedImagePath = null;
            SelectedImageAnnotations = [];
            AnnotationListItems = [];
            return;
        }

        var image = _project.Images.FirstOrDefault(
            item => item.ImageId == SelectedProjectImage.ImageId);
        if (image is null)
        {
            SelectedImagePath = null;
            SelectedImageAnnotations = [];
            AnnotationListItems = [];
            return;
        }

        var imagePath = TrainerProjectStore.ResolveImagePath(_projectDirectory, image);
        SelectedImagePath = File.Exists(imagePath) ? imagePath : null;
        SelectedImageAnnotations = image.Annotations?.ToArray() ?? [];
        AnnotationListItems = SelectedImageAnnotations
            .Select(annotation => new AnnotationListItemViewModel(
                annotation.AnnotationId,
                $"{annotation.ClassName} · x {annotation.X:P0}, y {annotation.Y:P0}, "
                + $"w {annotation.Width:P0}, h {annotation.Height:P0}"))
            .ToArray();
    }

    private void SelectRelativeImage(int offset)
    {
        if (ProjectImages.Count == 0)
        {
            return;
        }

        var currentIndex = SelectedProjectImage is null
            ? 0
            : ProjectImages.ToList().FindIndex(image => image.ImageId == SelectedProjectImage.ImageId);
        var nextIndex = Math.Clamp(currentIndex + offset, 0, ProjectImages.Count - 1);
        SelectedProjectImage = ProjectImages[nextIndex];
    }

    private void InvalidatePreparedProjectDataset()
    {
        if (string.IsNullOrWhiteSpace(TrainingDatasetPath) || string.IsNullOrWhiteSpace(_projectDirectory))
        {
            return;
        }

        var datasetsDirectory = Path.GetFullPath(Path.Combine(
            _projectDirectory,
            ProjectDatasetExporter.DatasetsDirectoryName));
        var selectedDataset = Path.GetFullPath(TrainingDatasetPath);
        if (selectedDataset.StartsWith(
            datasetsDirectory + Path.DirectorySeparatorChar,
            StringComparison.OrdinalIgnoreCase))
        {
            TrainingDatasetPath = string.Empty;
            ProjectDatasetStatus = "Annotations changed. Prepare a fresh dataset snapshot before training.";
            TrainingStatus = "The previous project dataset snapshot is stale.";
        }
    }

    private void DisposeProjectImages()
    {
        foreach (var image in ProjectImages)
        {
            image.Dispose();
        }
    }

    private void RollBackImportedFiles(IReadOnlyList<ProjectImageAsset> images)
    {
        foreach (var image in images)
        {
            if (image.StorageMode == ImageStorageMode.Copy)
            {
                var imagePath = TrainerProjectStore.ResolveImagePath(_projectDirectory, image);
                if (File.Exists(imagePath))
                {
                    File.Delete(imagePath);
                }
            }

            var thumbnailPath = TrainerProjectStore.ResolveThumbnailPath(_projectDirectory, image);
            if (File.Exists(thumbnailPath))
            {
                File.Delete(thumbnailPath);
            }
        }
    }

    private void RefreshRecentProjects()
    {
        RecentProjects = MapRecentProjects(_recentProjectStore.Load());
    }

    private void TryUpdateRecentProjects(string projectName)
    {
        try
        {
            RecentProjects = MapRecentProjects(_recentProjectStore.Add(projectName, _projectDirectory));
        }
        catch (Exception exception) when (exception is IOException or UnauthorizedAccessException)
        {
            ActivityMessage = $"Project opened, but recent-project history could not be updated: {exception.Message}";
        }
    }

    private static IReadOnlyList<RecentProjectViewModel> MapRecentProjects(
        IReadOnlyList<RecentProjectEntry> entries) =>
        entries.Select(entry => new RecentProjectViewModel(
            entry.Name,
            entry.ProjectDirectory,
            entry.LastOpenedAtUtc.ToLocalTime().ToString("g"),
            !File.Exists(Path.Combine(entry.ProjectDirectory, TrainerProjectStore.ManifestFileName))))
        .ToArray();

    private static string CreateUniqueProjectDirectory(string parentDirectory, string baseName)
    {
        var parent = Path.GetFullPath(parentDirectory);
        Directory.CreateDirectory(parent);

        for (var suffix = 1; suffix < 10_000; suffix++)
        {
            var directoryName = suffix == 1 ? baseName : $"{baseName}-{suffix}";
            var candidate = Path.Combine(parent, directoryName);
            if (!Directory.Exists(candidate) && !File.Exists(candidate))
            {
                return candidate;
            }
        }

        throw new IOException("A unique project directory could not be allocated.");
    }

    private static string CreateSlug(string value)
    {
        var characters = value.Trim().ToLowerInvariant()
            .Select(character => char.IsLetterOrDigit(character) ? character : '-')
            .ToArray();
        var slug = string.Join(
            '-',
            new string(characters).Split('-', StringSplitOptions.RemoveEmptyEntries));
        return string.IsNullOrWhiteSpace(slug)
            ? "aivqc-project"
            : slug[..Math.Min(slug.Length, 64)];
    }
}
