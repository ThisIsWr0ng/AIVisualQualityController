namespace Aivqc.Core.Projects;

/// <summary>
/// Applies validated manual object-detection annotations to Trainer projects.
/// </summary>
public static class TrainerProjectAnnotations
{
    public static TrainerProjectManifest AddClass(
        TrainerProjectManifest project,
        string className)
    {
        ArgumentNullException.ThrowIfNull(project);
        ValidateClassName(className);

        var normalizedName = className.Trim();
        if (project.DefectClasses.Count >= 9)
        {
            throw new InvalidOperationException(
                "Manual annotation supports up to nine defect classes with shortcuts 1–9.");
        }

        if (project.DefectClasses.Contains(normalizedName, StringComparer.OrdinalIgnoreCase))
        {
            throw new InvalidOperationException($"Defect class '{normalizedName}' already exists.");
        }

        return project with
        {
            DefectClasses = project.DefectClasses.Append(normalizedName).ToArray(),
            UpdatedAtUtc = DateTimeOffset.UtcNow,
        };
    }

    public static TrainerProjectManifest Add(
        TrainerProjectManifest project,
        Guid imageId,
        string className,
        NormalizedBoundingBox bounds)
    {
        ArgumentNullException.ThrowIfNull(project);
        ValidateClassName(className);
        ValidateBounds(bounds);

        var canonicalClass = project.DefectClasses.FirstOrDefault(
            item => string.Equals(item, className.Trim(), StringComparison.OrdinalIgnoreCase))
            ?? throw new InvalidOperationException($"Defect class '{className.Trim()}' does not exist.");
        var imageIndex = FindImage(project, imageId);
        var image = project.Images[imageIndex];
        var annotation = new ProjectObjectAnnotation(
            Guid.NewGuid(),
            canonicalClass,
            bounds.X,
            bounds.Y,
            bounds.Width,
            bounds.Height,
            DateTimeOffset.UtcNow);
        var updatedImage = image with
        {
            Annotations = (image.Annotations ?? []).Append(annotation).ToArray(),
        };

        return ReplaceImage(project, imageIndex, updatedImage);
    }

    public static TrainerProjectManifest Remove(
        TrainerProjectManifest project,
        Guid imageId,
        Guid annotationId)
    {
        ArgumentNullException.ThrowIfNull(project);
        var imageIndex = FindImage(project, imageId);
        var image = project.Images[imageIndex];
        var annotations = image.Annotations ?? [];
        if (!annotations.Any(annotation => annotation.AnnotationId == annotationId))
        {
            throw new KeyNotFoundException($"Annotation '{annotationId}' does not exist.");
        }

        var updatedImage = image with
        {
            Annotations = annotations
                .Where(annotation => annotation.AnnotationId != annotationId)
                .ToArray(),
        };
        return ReplaceImage(project, imageIndex, updatedImage);
    }

    private static TrainerProjectManifest ReplaceImage(
        TrainerProjectManifest project,
        int imageIndex,
        ProjectImageAsset updatedImage)
    {
        var images = project.Images.ToArray();
        images[imageIndex] = updatedImage;
        return project with
        {
            Images = images,
            UpdatedAtUtc = DateTimeOffset.UtcNow,
        };
    }

    private static int FindImage(TrainerProjectManifest project, Guid imageId)
    {
        for (var index = 0; index < project.Images.Count; index++)
        {
            if (project.Images[index].ImageId == imageId)
            {
                return index;
            }
        }

        throw new KeyNotFoundException($"Project image '{imageId}' does not exist.");
    }

    private static void ValidateClassName(string? className)
    {
        if (string.IsNullOrWhiteSpace(className)
            || className.Length > 64
            || className.Any(char.IsControl))
        {
            throw new ArgumentException("A defect class must contain 1–64 printable characters.", nameof(className));
        }
    }

    private static void ValidateBounds(NormalizedBoundingBox bounds)
    {
        if (!double.IsFinite(bounds.X)
            || !double.IsFinite(bounds.Y)
            || !double.IsFinite(bounds.Width)
            || !double.IsFinite(bounds.Height)
            || bounds.X < 0
            || bounds.Y < 0
            || bounds.Width <= 0
            || bounds.Height <= 0
            || bounds.X + bounds.Width > 1
            || bounds.Y + bounds.Height > 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(bounds),
                "Annotation coordinates must describe a positive rectangle inside the image.");
        }
    }
}

public readonly record struct NormalizedBoundingBox(
    double X,
    double Y,
    double Width,
    double Height);
