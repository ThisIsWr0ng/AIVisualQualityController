using Avalonia.Media.Imaging;

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

public sealed record AnnotationListItemViewModel(Guid AnnotationId, string Summary);

public sealed record RecentProjectViewModel(
    string Name,
    string ProjectDirectory,
    string LastOpened,
    bool IsMissing)
{
    public string Status => IsMissing ? "Missing project" : $"Opened {LastOpened}";
}
