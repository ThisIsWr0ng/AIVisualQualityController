namespace Aivqc.Core.Inspection;

/// <summary>
/// Represents one object-detection candidate in source-image coordinates.
/// </summary>
public sealed record DefectDetection(
    int ClassId,
    string ClassName,
    float Confidence,
    DetectionBox Box);

public sealed record DetectionBox(
    float X,
    float Y,
    float Width,
    float Height);
