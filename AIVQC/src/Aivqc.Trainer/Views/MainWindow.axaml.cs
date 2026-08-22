using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Aivqc.Trainer.Controls;
using Aivqc.Trainer.ViewModels;

namespace Aivqc.Trainer.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closed += OnWindowClosed;
    }

    private async void OnCreateProjectClick(object? sender, RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select a parent folder for the new AIVQC project",
            AllowMultiple = false,
        });

        var parentDirectory = folders.FirstOrDefault()?.TryGetLocalPath();
        if (parentDirectory is not null && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.CreateProjectAsync(parentDirectory);
        }
    }

    private async void OnOpenProjectClick(object? sender, RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = $"Select a folder containing {Aivqc.Core.Projects.TrainerProjectStore.ManifestFileName}",
            AllowMultiple = false,
        });

        var projectDirectory = folders.FirstOrDefault()?.TryGetLocalPath();
        if (projectDirectory is not null && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.OpenProjectAsync(projectDirectory);
        }
    }

    private async void OnRecentProjectClick(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string projectDirectory }
            && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.OpenProjectAsync(projectDirectory);
        }
    }

    private async void OnImportImagesClick(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Import project images",
            AllowMultiple = true,
            FileTypeFilter =
            [
                new FilePickerFileType("Supported images")
                {
                    Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"],
                    MimeTypes = ["image/jpeg", "image/png", "image/bmp", "image/webp"],
                },
            ],
        });

        var filePaths = files
            .Select(file => file.TryGetLocalPath())
            .Where(path => path is not null)
            .Cast<string>()
            .ToArray();
        if (filePaths.Length > 0 && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.ImportImagesAsync(filePaths);
        }
    }

    private async void OnImportOnnxClick(object? sender, RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Import ONNX model",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("ONNX model")
                {
                    Patterns = ["*.onnx"],
                    MimeTypes = ["application/octet-stream"],
                },
            ],
        });

        var filePath = files.FirstOrDefault()?.TryGetLocalPath();
        if (filePath is not null && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.ImportOnnxAsync(filePath);
        }
    }

    private async void OnSettingsClick(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        var settingsWindow = new SettingsWindow
        {
            DataContext = new SettingsWindowViewModel(viewModel),
        };
        await settingsWindow.ShowDialog(this);
    }

    private async void OnAnnotationCreated(object? sender, AnnotationCreatedEventArgs e)
    {
        if (DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.CreateAnnotationAsync(e.Bounds);
        }
    }

    private void OnAnnotationSelected(object? sender, AnnotationSelectedEventArgs e)
    {
        if (DataContext is MainWindowViewModel viewModel)
        {
            viewModel.SelectAnnotation(e.AnnotationId);
        }
    }

    private void OnWindowKeyDown(object? sender, KeyEventArgs e)
    {
        if (TopLevel.GetTopLevel(this)?.FocusManager?.GetFocusedElement() is TextBox
            || DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        var shortcut = e.Key switch
        {
            Key.D1 or Key.NumPad1 => 1,
            Key.D2 or Key.NumPad2 => 2,
            Key.D3 or Key.NumPad3 => 3,
            Key.D4 or Key.NumPad4 => 4,
            Key.D5 or Key.NumPad5 => 5,
            Key.D6 or Key.NumPad6 => 6,
            Key.D7 or Key.NumPad7 => 7,
            Key.D8 or Key.NumPad8 => 8,
            Key.D9 or Key.NumPad9 => 9,
            _ => 0,
        };
        if (shortcut > 0)
        {
            viewModel.SelectDefectClassByShortcut(shortcut);
            e.Handled = true;
        }
    }

    private async void OnSelectTrainingDatasetClick(object? sender, RoutedEventArgs e)
    {
        var folders = await StorageProvider.OpenFolderPickerAsync(new FolderPickerOpenOptions
        {
            Title = "Select Pascal VOC dataset",
            AllowMultiple = false,
        });

        var directoryPath = folders.FirstOrDefault()?.TryGetLocalPath();
        if (directoryPath is not null && DataContext is MainWindowViewModel viewModel)
        {
            viewModel.SelectTrainingDataset(directoryPath);
        }
    }

    private async void OnExportPackageClick(object? sender, RoutedEventArgs e)
    {
        if (DataContext is not MainWindowViewModel viewModel)
        {
            return;
        }

        var file = await StorageProvider.SaveFilePickerAsync(new FilePickerSaveOptions
        {
            Title = "Export AIVQC deployment package",
            SuggestedFileName = $"{viewModel.DeploymentRecipeId}{Aivqc.Core.Deployment.DeploymentPackageArchive.FileExtension}",
            DefaultExtension = Aivqc.Core.Deployment.DeploymentPackageArchive.FileExtension.TrimStart('.'),
            FileTypeChoices =
            [
                new FilePickerFileType("AIVQC deployment package")
                {
                    Patterns = [$"*{Aivqc.Core.Deployment.DeploymentPackageArchive.FileExtension}"],
                },
            ],
        });

        var filePath = file?.TryGetLocalPath();
        if (filePath is not null)
        {
            if (!filePath.EndsWith(
                Aivqc.Core.Deployment.DeploymentPackageArchive.FileExtension,
                StringComparison.OrdinalIgnoreCase))
            {
                filePath += Aivqc.Core.Deployment.DeploymentPackageArchive.FileExtension;
            }

            await viewModel.ExportDeploymentPackageAsync(filePath);
        }
    }

    private void OnWindowClosed(object? sender, EventArgs e)
    {
        if (DataContext is IDisposable disposable)
        {
            disposable.Dispose();
        }
    }
}
