using Avalonia.Media.Imaging;
using Avalonia.Media;

namespace Aivqc.Trainer.Models;

public sealed class ProjectImageViewModel : IDisposable
{
    public ProjectImageViewModel(
        Guid imageId,
        string fileName,
        string resolution,
        string storage,
        string warningSummary,
        string annotationSummary,
        Bitmap thumbnail)
    {
        ImageId = imageId;
        FileName = fileName;
        Resolution = resolution;
        Storage = storage;
        WarningSummary = warningSummary;
        AnnotationSummary = annotationSummary;
        Thumbnail = thumbnail;
    }

    public Guid ImageId { get; }

    public string FileName { get; }

    public string Resolution { get; }

    public string Storage { get; }

    public string WarningSummary { get; }

    public string AnnotationSummary { get; }

    public Bitmap Thumbnail { get; }

    public void Dispose() => Thumbnail.Dispose();
}

public sealed record DefectClassViewModel(
    string ClassName,
    int Shortcut,
    string ColorHex,
    IBrush ColorBrush)
{
    public string ShortcutDisplay => Shortcut.ToString();
}

public static class AnnotationColorPalette
{
    private static readonly string[] Colors =
    [
        "#39D9B1",
        "#FF6B7A",
        "#58A6FF",
        "#FFD166",
        "#C77DFF",
        "#FF922B",
        "#4DDB6D",
        "#F065C2",
        "#5CE1E6",
    ];

    public static IReadOnlyList<DefectClassViewModel> Create(IReadOnlyList<string> classNames) =>
        classNames.Take(Colors.Length)
            .Select((className, index) => new DefectClassViewModel(
                className,
                index + 1,
                Colors[index],
                new SolidColorBrush(Color.Parse(Colors[index]))))
            .ToArray();

    public static Color GetColor(string className, IReadOnlyList<string> classNames)
    {
        var index = classNames
            .Select((name, position) => (name, position))
            .FirstOrDefault(item => string.Equals(
                item.name,
                className,
                StringComparison.OrdinalIgnoreCase))
            .position;
        if (classNames.Count == 0
            || index < 0
            || index >= classNames.Count
            || !string.Equals(classNames[index], className, StringComparison.OrdinalIgnoreCase))
        {
            index = (StringComparer.OrdinalIgnoreCase.GetHashCode(className) & int.MaxValue) % Colors.Length;
        }

        return Color.Parse(Colors[index % Colors.Length]);
    }
}

public sealed record RecentProjectViewModel(
    string Name,
    string ProjectDirectory,
    string LastOpened,
    bool IsMissing)
{
    public string Status => IsMissing ? "Missing project" : $"Opened {LastOpened}";
}
