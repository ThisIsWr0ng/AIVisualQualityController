using System.Diagnostics;
using System.Text.Json;
using Aivqc.Core.Inspection;
using Aivqc.Production.Models;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SkiaSharp;

namespace Aivqc.Production.Services;

/// <summary>
/// Owns one ONNX Runtime session and executes AIVQC object-detection models.
/// </summary>
public sealed class OnnxInspectionSession : IDisposable
{
    private static readonly string[] RequiredOutputs = ["boxes", "scores", "labels"];

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly IReadOnlyDictionary<int, string> _classNames;
    private readonly int _inputWidth;
    private readonly int _inputHeight;
    private bool _disposed;

    public OnnxInspectionSession(string modelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);

        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException("The selected ONNX model does not exist.", modelPath);
        }

        if (!string.Equals(Path.GetExtension(modelPath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Select a file with the .onnx extension.");
        }

        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
        };

        try
        {
            _session = new InferenceSession(modelPath, options);
        }
        finally
        {
            options.Dispose();
        }

        try
        {
            (_inputName, _inputWidth, _inputHeight) = ValidateContract(_session);
            _classNames = LoadClassNames(modelPath);
            Information = new OnnxModelInformation(
                Path.GetFileName(modelPath),
                Path.GetFullPath(modelPath),
                _inputWidth,
                _inputHeight,
                _classNames);
        }
        catch
        {
            _session.Dispose();
            throw;
        }
    }

    public OnnxModelInformation Information { get; }

    public Task<OnnxInspectionResult> InspectAsync(
        string imagePath,
        float confidenceThreshold,
        CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);

        if (confidenceThreshold is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(confidenceThreshold),
                "The confidence threshold must be between 0 and 1.");
        }

        return Task.Run(
            () => Inspect(imagePath, confidenceThreshold, cancellationToken),
            cancellationToken);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session.Dispose();
        _disposed = true;
    }

    private OnnxInspectionResult Inspect(
        string imagePath,
        float confidenceThreshold,
        CancellationToken cancellationToken)
    {
        if (!File.Exists(imagePath))
        {
            throw new FileNotFoundException("The selected inspection image does not exist.", imagePath);
        }

        cancellationToken.ThrowIfCancellationRequested();

        using var source = SKBitmap.Decode(imagePath)
            ?? throw new InvalidDataException("The selected file is not a supported image.");
        using var resized = source.Resize(
            new SKImageInfo(_inputWidth, _inputHeight, SKColorType.Rgba8888),
            new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.None))
            ?? throw new InvalidOperationException("The inspection image could not be resized.");

        var tensor = CreateInputTensor(resized, cancellationToken);
        var input = NamedOnnxValue.CreateFromTensor(_inputName, tensor);
        var stopwatch = Stopwatch.StartNew();

        using var outputs = _session.Run([input], RequiredOutputs);
        stopwatch.Stop();
        cancellationToken.ThrowIfCancellationRequested();

        var candidates = ReadDetections(outputs, source.Width, source.Height);
        var decision = InspectionDecisionEngine.Decide(
            candidates,
            defaultThreshold: confidenceThreshold);
        var annotatedImage = RenderAnnotatedImage(source, decision.RejectedDetections);

        return new OnnxInspectionResult(
            decision,
            candidates,
            annotatedImage,
            source.Width,
            source.Height,
            stopwatch.Elapsed);
    }

    private DenseTensor<float> CreateInputTensor(SKBitmap image, CancellationToken cancellationToken)
    {
        var tensor = new DenseTensor<float>([1, 3, _inputHeight, _inputWidth]);

        for (var y = 0; y < _inputHeight; y++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            for (var x = 0; x < _inputWidth; x++)
            {
                var pixel = image.GetPixel(x, y);
                tensor[0, 0, y, x] = pixel.Red / 255f;
                tensor[0, 1, y, x] = pixel.Green / 255f;
                tensor[0, 2, y, x] = pixel.Blue / 255f;
            }
        }

        return tensor;
    }

    private IReadOnlyList<DefectDetection> ReadDetections(
        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs,
        int sourceWidth,
        int sourceHeight)
    {
        var boxes = outputs.Single(output => output.Name == "boxes").AsTensor<float>().ToArray();
        var scores = outputs.Single(output => output.Name == "scores").AsTensor<float>().ToArray();
        var labels = outputs.Single(output => output.Name == "labels").AsTensor<long>().ToArray();
        var count = Math.Min(scores.Length, Math.Min(labels.Length, boxes.Length / 4));
        var scaleX = sourceWidth / (float)_inputWidth;
        var scaleY = sourceHeight / (float)_inputHeight;
        var detections = new List<DefectDetection>(count);

        for (var index = 0; index < count; index++)
        {
            var offset = index * 4;
            var left = Math.Clamp(boxes[offset] * scaleX, 0, sourceWidth);
            var top = Math.Clamp(boxes[offset + 1] * scaleY, 0, sourceHeight);
            var right = Math.Clamp(boxes[offset + 2] * scaleX, 0, sourceWidth);
            var bottom = Math.Clamp(boxes[offset + 3] * scaleY, 0, sourceHeight);
            var classId = checked((int)labels[index]);

            if (!float.IsFinite(scores[index]) || right <= left || bottom <= top)
            {
                continue;
            }

            detections.Add(new DefectDetection(
                classId,
                _classNames.GetValueOrDefault(classId, $"Class {classId}"),
                scores[index],
                new DetectionBox(left, top, right - left, bottom - top)));
        }

        return detections;
    }

    private static byte[] RenderAnnotatedImage(
        SKBitmap source,
        IReadOnlyList<DefectDetection> detections)
    {
        using var annotated = source.Copy()
            ?? throw new InvalidOperationException("The inspection preview could not be created.");
        using var canvas = new SKCanvas(annotated);
        using var outlinePaint = new SKPaint
        {
            Color = new SKColor(255, 83, 112),
            IsAntialias = true,
            Style = SKPaintStyle.Stroke,
            StrokeWidth = Math.Max(2f, source.Width / 400f),
        };
        using var labelPaint = new SKPaint
        {
            Color = SKColors.White,
            IsAntialias = true,
        };
        using var labelBackgroundPaint = new SKPaint
        {
            Color = new SKColor(167, 22, 48, 230),
            Style = SKPaintStyle.Fill,
        };
        using var font = new SKFont(SKTypeface.Default, Math.Max(14f, source.Width / 55f));

        foreach (var detection in detections)
        {
            var box = detection.Box;
            var rectangle = new SKRect(box.X, box.Y, box.X + box.Width, box.Y + box.Height);
            canvas.DrawRect(rectangle, outlinePaint);

            var label = $"{detection.ClassName} {detection.Confidence:P0}";
            var labelWidth = font.MeasureText(label, labelPaint);
            var labelHeight = font.Size * 1.35f;
            var labelTop = Math.Max(0, rectangle.Top - labelHeight);
            canvas.DrawRect(
                new SKRect(rectangle.Left, labelTop, rectangle.Left + labelWidth + 12, labelTop + labelHeight),
                labelBackgroundPaint);
            canvas.DrawText(label, rectangle.Left + 6, labelTop + font.Size, font, labelPaint);
        }

        using var image = SKImage.FromBitmap(annotated);
        using var encoded = image.Encode(SKEncodedImageFormat.Png, 95);
        return encoded.ToArray();
    }

    private static (string Name, int Width, int Height) ValidateContract(InferenceSession session)
    {
        if (session.InputMetadata.Count != 1)
        {
            throw new InvalidDataException("The model must expose exactly one image input.");
        }

        var input = session.InputMetadata.Single();
        var dimensions = input.Value.Dimensions;
        if (input.Value.ElementType != typeof(float)
            || dimensions.Length != 4
            || dimensions[1] != 3
            || dimensions[2] <= 0
            || dimensions[3] <= 0)
        {
            throw new InvalidDataException(
                "The model input must be a Float tensor shaped [1, 3, height, width].");
        }

        foreach (var outputName in RequiredOutputs)
        {
            if (!session.OutputMetadata.ContainsKey(outputName))
            {
                throw new InvalidDataException($"The model is missing the required '{outputName}' output.");
            }
        }

        return (input.Key, dimensions[3], dimensions[2]);
    }

    private static IReadOnlyDictionary<int, string> LoadClassNames(string modelPath)
    {
        var classesPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "classes.json");
        if (!File.Exists(classesPath))
        {
            return new Dictionary<int, string>();
        }

        try
        {
            var classToId = JsonSerializer.Deserialize<Dictionary<string, int>>(
                File.ReadAllText(classesPath));
            return classToId?.ToDictionary(item => item.Value, item => item.Key)
                ?? new Dictionary<int, string>();
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The classes.json file is invalid.", exception);
        }
    }
}
