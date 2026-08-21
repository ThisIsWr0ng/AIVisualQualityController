using Aivqc.Production.ViewModels;
using Avalonia.Controls;
using Avalonia.Platform.Storage;

namespace Aivqc.Production.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closed += OnWindowClosed;
    }

    private async void OnSelectModelClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Load AIVQC ONNX model",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("ONNX model")
                {
                    Patterns = ["*.onnx"],
                },
            ],
        });

        if (files.Count == 1 && DataContext is MainWindowViewModel viewModel)
        {
            await viewModel.LoadModelAsync(files[0].Path.LocalPath);
        }
    }

    private async void OnSelectImageClick(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
    {
        var files = await StorageProvider.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "Select an inspection image",
            AllowMultiple = false,
            FileTypeFilter =
            [
                new FilePickerFileType("Image")
                {
                    Patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"],
                },
            ],
        });

        if (files.Count == 1 && DataContext is MainWindowViewModel viewModel)
        {
            viewModel.LoadImage(files[0].Path.LocalPath);
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
