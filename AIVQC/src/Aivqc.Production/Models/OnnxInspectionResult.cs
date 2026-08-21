using Aivqc.Core.Inspection;

namespace Aivqc.Production.Models;

public sealed record OnnxInspectionResult(
    InspectionDecision Decision,
    IReadOnlyList<DefectDetection> Candidates,
    byte[] AnnotatedImagePng,
    int SourceWidth,
    int SourceHeight,
    TimeSpan InferenceDuration);
