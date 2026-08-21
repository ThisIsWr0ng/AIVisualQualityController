using System.Text.Json.Serialization;

namespace Aivqc.Core.Training;

/// <summary>
/// Represents one progress event emitted by an isolated training backend.
/// </summary>
public sealed record TrainingJobEvent
{
    [JsonPropertyName("type")]
    public string Type { get; init; } = string.Empty;

    [JsonPropertyName("message")]
    public string? Message { get; init; }

    [JsonPropertyName("device")]
    public string? Device { get; init; }

    [JsonPropertyName("epoch")]
    public int? Epoch { get; init; }

    [JsonPropertyName("epochs")]
    public int? Epochs { get; init; }

    [JsonPropertyName("train_loss")]
    public double? TrainLoss { get; init; }

    [JsonPropertyName("map50")]
    public double? Map50 { get; init; }

    [JsonPropertyName("map50_95")]
    public double? Map50To95 { get; init; }

    [JsonPropertyName("precision")]
    public double? Precision { get; init; }

    [JsonPropertyName("recall")]
    public double? Recall { get; init; }

    [JsonPropertyName("f1")]
    public double? F1 { get; init; }

    [JsonPropertyName("onnx_path")]
    public string? OnnxPath { get; init; }

    [JsonPropertyName("checkpoint_path")]
    public string? CheckpointPath { get; init; }

    [JsonPropertyName("metrics_path")]
    public string? MetricsPath { get; init; }

    [JsonPropertyName("run_directory")]
    public string? RunDirectory { get; init; }
}
