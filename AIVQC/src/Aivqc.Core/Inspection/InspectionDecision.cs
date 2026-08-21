namespace Aivqc.Core.Inspection;

public sealed record InspectionDecision(
    InspectionDecisionState State,
    IReadOnlyList<DefectDetection> RejectedDetections,
    string Reason);

public enum InspectionDecisionState
{
    Ok,
    Nok,
    Error,
}
