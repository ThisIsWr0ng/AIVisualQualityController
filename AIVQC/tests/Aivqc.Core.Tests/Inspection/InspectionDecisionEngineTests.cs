using Aivqc.Core.Inspection;

namespace Aivqc.Core.Tests.Inspection;

public sealed class InspectionDecisionEngineTests
{
    [Fact]
    public void Decide_ReturnsOk_WhenNoDetectionReachesThreshold()
    {
        var detections = new[]
        {
            CreateDetection(classId: 1, confidence: 0.49f),
        };

        var result = InspectionDecisionEngine.Decide(detections, defaultThreshold: 0.5f);

        Assert.Equal(InspectionDecisionState.Ok, result.State);
        Assert.Empty(result.RejectedDetections);
    }

    [Fact]
    public void Decide_ReturnsNok_AndUsesPerClassThreshold()
    {
        var detections = new[]
        {
            CreateDetection(classId: 1, confidence: 0.6f),
            CreateDetection(classId: 2, confidence: 0.8f),
        };
        var thresholds = new Dictionary<int, float>
        {
            [1] = 0.7f,
            [2] = 0.75f,
        };

        var result = InspectionDecisionEngine.Decide(detections, thresholds);

        Assert.Equal(InspectionDecisionState.Nok, result.State);
        Assert.Collection(
            result.RejectedDetections,
            detection => Assert.Equal(2, detection.ClassId));
    }

    [Fact]
    public void Decide_RejectsInvalidThresholds()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            InspectionDecisionEngine.Decide([], defaultThreshold: 1.01f));
    }

    private static DefectDetection CreateDetection(int classId, float confidence) =>
        new(classId, $"Class {classId}", confidence, new DetectionBox(0, 0, 10, 10));
}
