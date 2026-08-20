using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Platform.Storage;
using Aivqc.Trainer.ViewModels;

namespace Aivqc.Trainer.Views;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
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
}
