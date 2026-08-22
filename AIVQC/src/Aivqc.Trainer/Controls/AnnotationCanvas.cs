using Aivqc.Core.Projects;
using Aivqc.Trainer.Models;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using System.Globalization;

namespace Aivqc.Trainer.Controls;

/// <summary>
/// Displays one project image and turns pointer drags into normalized bounding boxes.
/// </summary>
public sealed class AnnotationCanvas : Control, IDisposable
{
    public static readonly StyledProperty<string?> ImagePathProperty =
        AvaloniaProperty.Register<AnnotationCanvas, string?>(nameof(ImagePath));

    public static readonly StyledProperty<IReadOnlyList<ProjectObjectAnnotation>> AnnotationsProperty =
        AvaloniaProperty.Register<AnnotationCanvas, IReadOnlyList<ProjectObjectAnnotation>>(
            nameof(Annotations),
            []);

    public static readonly StyledProperty<Guid> SelectedAnnotationIdProperty =
        AvaloniaProperty.Register<AnnotationCanvas, Guid>(nameof(SelectedAnnotationId));

    public static readonly StyledProperty<string?> SelectedClassNameProperty =
        AvaloniaProperty.Register<AnnotationCanvas, string?>(nameof(SelectedClassName));

    public static readonly StyledProperty<IReadOnlyList<string>> DefectClassesProperty =
        AvaloniaProperty.Register<AnnotationCanvas, IReadOnlyList<string>>(
            nameof(DefectClasses),
            []);

    private Bitmap? _image;
    private Point? _dragStart;
    private Point? _dragEnd;

    public AnnotationCanvas()
    {
        Focusable = true;
        ClipToBounds = true;
    }

    public event EventHandler<AnnotationCreatedEventArgs>? AnnotationCreated;

    public event EventHandler<AnnotationSelectedEventArgs>? AnnotationSelected;

    public string? ImagePath
    {
        get => GetValue(ImagePathProperty);
        set => SetValue(ImagePathProperty, value);
    }

    public IReadOnlyList<ProjectObjectAnnotation> Annotations
    {
        get => GetValue(AnnotationsProperty);
        set => SetValue(AnnotationsProperty, value);
    }

    public Guid SelectedAnnotationId
    {
        get => GetValue(SelectedAnnotationIdProperty);
        set => SetValue(SelectedAnnotationIdProperty, value);
    }

    public string? SelectedClassName
    {
        get => GetValue(SelectedClassNameProperty);
        set => SetValue(SelectedClassNameProperty, value);
    }

    public IReadOnlyList<string> DefectClasses
    {
        get => GetValue(DefectClassesProperty);
        set => SetValue(DefectClassesProperty, value);
    }

    public override void Render(DrawingContext context)
    {
        base.Render(context);
        context.FillRectangle(new SolidColorBrush(Color.Parse("#080D17")), Bounds);

        if (_image is null || !TryGetImageRectangle(out var imageRectangle))
        {
            return;
        }

        context.DrawImage(_image, new Rect(_image.Size), imageRectangle);
        foreach (var annotation in Annotations)
        {
            var rectangle = ToDisplayRectangle(annotation, imageRectangle);
            var isSelected = annotation.AnnotationId == SelectedAnnotationId;
            var color = AnnotationColorPalette.GetColor(annotation.ClassName, DefectClasses);
            context.DrawRectangle(
                isSelected ? new SolidColorBrush(Color.FromArgb(48, color.R, color.G, color.B)) : null,
                new Pen(new SolidColorBrush(color), isSelected ? 4 : 2),
                rectangle);
            DrawClassLabel(context, annotation.ClassName, color, rectangle);
        }

        if (_dragStart is { } start && _dragEnd is { } end)
        {
            var preview = ClipToImage(CreateRectangle(start, end), imageRectangle);
            var previewColor = string.IsNullOrWhiteSpace(SelectedClassName)
                ? Color.Parse("#39D9B1")
                : AnnotationColorPalette.GetColor(SelectedClassName, DefectClasses);
            context.DrawRectangle(
                new SolidColorBrush(Color.FromArgb(36, previewColor.R, previewColor.G, previewColor.B)),
                new Pen(new SolidColorBrush(previewColor), 2, dashStyle: DashStyle.Dash),
                preview);
        }
    }

    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);
        if (change.Property == ImagePathProperty)
        {
            LoadImage();
        }

        if (change.Property == AnnotationsProperty
            || change.Property == SelectedAnnotationIdProperty
            || change.Property == SelectedClassNameProperty
            || change.Property == DefectClassesProperty)
        {
            InvalidateVisual();
        }
    }

    protected override void OnPointerPressed(PointerPressedEventArgs e)
    {
        base.OnPointerPressed(e);
        if (_image is null || !TryGetImageRectangle(out var imageRectangle))
        {
            return;
        }

        var point = e.GetPosition(this);
        if (!imageRectangle.Contains(point))
        {
            return;
        }

        Focus();
        var existing = Annotations.LastOrDefault(annotation =>
            ToDisplayRectangle(annotation, imageRectangle).Contains(point));
        if (existing is not null)
        {
            AnnotationSelected?.Invoke(this, new AnnotationSelectedEventArgs(existing.AnnotationId));
            e.Handled = true;
            return;
        }

        if (string.IsNullOrWhiteSpace(SelectedClassName))
        {
            return;
        }

        _dragStart = point;
        _dragEnd = point;
        e.Pointer.Capture(this);
        e.Handled = true;
    }

    protected override void OnPointerMoved(PointerEventArgs e)
    {
        base.OnPointerMoved(e);
        if (_dragStart is null)
        {
            return;
        }

        _dragEnd = e.GetPosition(this);
        InvalidateVisual();
        e.Handled = true;
    }

    protected override void OnPointerReleased(PointerReleasedEventArgs e)
    {
        base.OnPointerReleased(e);
        if (_dragStart is not { } start
            || _image is null
            || !TryGetImageRectangle(out var imageRectangle))
        {
            return;
        }

        var end = e.GetPosition(this);
        var rectangle = ClipToImage(CreateRectangle(start, end), imageRectangle);
        _dragStart = null;
        _dragEnd = null;
        e.Pointer.Capture(null);
        InvalidateVisual();
        e.Handled = true;

        if (rectangle.Width < 4 || rectangle.Height < 4)
        {
            return;
        }

        var bounds = new NormalizedBoundingBox(
            (rectangle.X - imageRectangle.X) / imageRectangle.Width,
            (rectangle.Y - imageRectangle.Y) / imageRectangle.Height,
            rectangle.Width / imageRectangle.Width,
            rectangle.Height / imageRectangle.Height);
        AnnotationCreated?.Invoke(this, new AnnotationCreatedEventArgs(bounds));
    }

    private void LoadImage()
    {
        _image?.Dispose();
        _image = null;
        _dragStart = null;
        _dragEnd = null;

        if (!string.IsNullOrWhiteSpace(ImagePath) && File.Exists(ImagePath))
        {
            try
            {
                _image = new Bitmap(ImagePath);
            }
            catch
            {
                // The workspace reports missing or unreadable assets separately.
            }
        }

        InvalidateVisual();
    }

    private bool TryGetImageRectangle(out Rect rectangle)
    {
        rectangle = default;
        if (_image is null || Bounds.Width <= 0 || Bounds.Height <= 0)
        {
            return false;
        }

        var scale = Math.Min(Bounds.Width / _image.Size.Width, Bounds.Height / _image.Size.Height);
        var width = _image.Size.Width * scale;
        var height = _image.Size.Height * scale;
        rectangle = new Rect((Bounds.Width - width) / 2, (Bounds.Height - height) / 2, width, height);
        return true;
    }

    private static Rect ToDisplayRectangle(ProjectObjectAnnotation annotation, Rect imageRectangle) =>
        new(
            imageRectangle.X + annotation.X * imageRectangle.Width,
            imageRectangle.Y + annotation.Y * imageRectangle.Height,
            annotation.Width * imageRectangle.Width,
            annotation.Height * imageRectangle.Height);

    private void DrawClassLabel(
        DrawingContext context,
        string className,
        Color color,
        Rect rectangle)
    {
        var text = new FormattedText(
            className,
            CultureInfo.CurrentCulture,
            FlowDirection.LeftToRight,
            new Typeface("Inter", FontStyle.Normal, FontWeight.SemiBold),
            12,
            Brushes.Black);
        var left = Math.Clamp(rectangle.Left, 0, Math.Max(0, Bounds.Width - 1));
        var top = Math.Clamp(rectangle.Top, 0, Math.Max(0, Bounds.Height - text.Height - 6));
        var width = Math.Min(text.Width + 12, Math.Max(1, Bounds.Width - left));
        var background = new Rect(left, top, width, text.Height + 6);
        context.FillRectangle(new SolidColorBrush(color), background);
        context.DrawText(text, new Point(left + 6, top + 3));
    }

    private static Rect CreateRectangle(Point first, Point second) =>
        new(
            Math.Min(first.X, second.X),
            Math.Min(first.Y, second.Y),
            Math.Abs(first.X - second.X),
            Math.Abs(first.Y - second.Y));

    private static Rect ClipToImage(Rect rectangle, Rect imageRectangle)
    {
        var left = Math.Clamp(rectangle.Left, imageRectangle.Left, imageRectangle.Right);
        var top = Math.Clamp(rectangle.Top, imageRectangle.Top, imageRectangle.Bottom);
        var right = Math.Clamp(rectangle.Right, imageRectangle.Left, imageRectangle.Right);
        var bottom = Math.Clamp(rectangle.Bottom, imageRectangle.Top, imageRectangle.Bottom);
        return new Rect(left, top, Math.Max(0, right - left), Math.Max(0, bottom - top));
    }

    public void Dispose()
    {
        _image?.Dispose();
        _image = null;
    }
}

public sealed class AnnotationCreatedEventArgs(NormalizedBoundingBox bounds) : EventArgs
{
    public NormalizedBoundingBox Bounds { get; } = bounds;
}

public sealed class AnnotationSelectedEventArgs(Guid annotationId) : EventArgs
{
    public Guid AnnotationId { get; } = annotationId;
}
