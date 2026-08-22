using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using Aivqc.Trainer.Models;
using Microsoft.ML.OnnxRuntime;

namespace Aivqc.Trainer.Services;

public static class OnnxModelInspector
{
    public static Task<OnnxModelSummary> InspectAsync(
        string filePath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(filePath);

        return Task.Run(() => Inspect(filePath, cancellationToken), cancellationToken);
    }

    private static OnnxModelSummary Inspect(string filePath, CancellationToken cancellationToken)
    {
        if (!File.Exists(filePath))
        {
            throw new FileNotFoundException("The selected ONNX model does not exist.", filePath);
        }

        if (!string.Equals(Path.GetExtension(filePath), ".onnx", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException("Select a file with the .onnx extension.");
        }

        cancellationToken.ThrowIfCancellationRequested();

        var fileInfo = new FileInfo(filePath);
        var sha256 = CalculateSha256(filePath, cancellationToken);

        using var sessionOptions = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_BASIC,
        };
        using var session = new InferenceSession(filePath, sessionOptions);

        var inputs = session.InputMetadata
            .Select(item => FormatTensor(item.Key, item.Value.Dimensions, item.Value.ElementType))
            .ToArray();
        var outputs = session.OutputMetadata
            .Select(item => FormatTensor(item.Key, item.Value.Dimensions, item.Value.ElementType))
            .ToArray();
        var primaryInput = session.InputMetadata.Count == 1
            ? session.InputMetadata.Single().Value.Dimensions
            : [];
        var inputHeight = primaryInput.Length == 4 ? primaryInput[2] : 0;
        var inputWidth = primaryInput.Length == 4 ? primaryInput[3] : 0;
        var classNames = LoadClassNames(filePath);

        return new OnnxModelSummary(
            fileInfo.Name,
            fileInfo.FullName,
            fileInfo.Length,
            sha256,
            inputWidth,
            inputHeight,
            classNames,
            inputs,
            outputs);
    }

    private static string CalculateSha256(string filePath, CancellationToken cancellationToken)
    {
        using var stream = File.OpenRead(filePath);
        using var algorithm = SHA256.Create();

        var buffer = new byte[1024 * 1024];
        int bytesRead;
        while ((bytesRead = stream.Read(buffer, 0, buffer.Length)) > 0)
        {
            cancellationToken.ThrowIfCancellationRequested();
            algorithm.TransformBlock(buffer, 0, bytesRead, null, 0);
        }

        algorithm.TransformFinalBlock([], 0, 0);
        return Convert.ToHexString(algorithm.Hash!);
    }

    private static string FormatTensor(string name, int[] dimensions, Type elementType)
    {
        var shape = string.Join(
            " × ",
            dimensions.Select(dimension => dimension <= 0
                ? "dynamic"
                : dimension.ToString(CultureInfo.InvariantCulture)));

        return $"{name} · {elementType.Name} [{shape}]";
    }

    private static IReadOnlyDictionary<string, int> LoadClassNames(string modelPath)
    {
        var classesPath = Path.Combine(Path.GetDirectoryName(modelPath)!, "classes.json");
        if (!File.Exists(classesPath))
        {
            return new Dictionary<string, int>();
        }

        try
        {
            return JsonSerializer.Deserialize<Dictionary<string, int>>(File.ReadAllText(classesPath))
                ?? new Dictionary<string, int>();
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The classes.json file next to the model is invalid.", exception);
        }
    }
}
