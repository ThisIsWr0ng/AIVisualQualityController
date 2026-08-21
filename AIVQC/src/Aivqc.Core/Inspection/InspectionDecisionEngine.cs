namespace Aivqc.Core.Inspection;

/// <summary>
/// Applies deterministic per-class confidence thresholds to model detections.
/// </summary>
public static class InspectionDecisionEngine
{
    public static InspectionDecision Decide(
        IReadOnlyList<DefectDetection> detections,
        IReadOnlyDictionary<int, float>? classThresholds = null,
        float defaultThreshold = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(detections);

        if (defaultThreshold is < 0 or > 1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(defaultThreshold),
                "The confidence threshold must be between 0 and 1.");
        }

        if (classThresholds?.Any(item => item.Value is < 0 or > 1) == true)
        {
            throw new ArgumentOutOfRangeException(
                nameof(classThresholds),
                "Every class threshold must be between 0 and 1.");
        }

        var rejected = detections
            .Where(detection =>
            {
                var threshold = classThresholds is not null
                    && classThresholds.TryGetValue(detection.ClassId, out var configured)
                        ? configured
                        : defaultThreshold;
                return detection.Confidence >= threshold;
            })
            .OrderByDescending(detection => detection.Confidence)
            .ToArray();

        return rejected.Length == 0
            ? new InspectionDecision(
                InspectionDecisionState.Ok,
                rejected,
                "No defect exceeded its confidence threshold.")
            : new InspectionDecision(
                InspectionDecisionState.Nok,
                rejected,
                $"Detected {rejected.Length} defect{(rejected.Length == 1 ? string.Empty : "s")} above threshold.");
    }

    public static InspectionDecision Error(string reason)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(reason);
        return new InspectionDecision(InspectionDecisionState.Error, [], reason);
    }
}
