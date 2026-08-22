using Avalonia.Controls;
using Aivqc.Trainer.ViewModels;

namespace Aivqc.Trainer.Views;

public partial class SettingsWindow : Window
{
    public SettingsWindow()
    {
        InitializeComponent();
        DataContextChanged += OnDataContextChanged;
    }

    private void OnDataContextChanged(object? sender, EventArgs e)
    {
        if (DataContext is not SettingsWindowViewModel viewModel)
        {
            return;
        }

        viewModel.SaveRequested += OnCloseRequested;
        viewModel.CancelRequested += OnCloseRequested;
    }

    private void OnCloseRequested(object? sender, EventArgs e) => Close();
}
