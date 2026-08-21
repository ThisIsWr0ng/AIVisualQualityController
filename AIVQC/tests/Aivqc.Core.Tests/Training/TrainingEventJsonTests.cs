using Aivqc.Core.Training;

namespace Aivqc.Core.Tests.Training;

public sealed class TrainingEventJsonTests
{
    [Fact]
    public void Parse_ReadsEpochMetrics()
    {
        const string Json = """
            {"type":"epoch","epoch":2,"epochs":10,"train_loss":1.25,"map50":0.75,"map50_95":0.5,"precision":0.8,"recall":0.7}
            """;

        var result = TrainingEventJson.Parse(Json);

        Assert.NotNull(result);
        Assert.Equal("epoch", result.Type);
        Assert.Equal(2, result.Epoch);
        Assert.Equal(10, result.Epochs);
        Assert.Equal(1.25, result.TrainLoss);
        Assert.Equal(0.75, result.Map50);
        Assert.Equal(0.5, result.Map50To95);
        Assert.Equal(0.8, result.Precision);
        Assert.Equal(0.7, result.Recall);
    }

    [Theory]
    [InlineData("")]
    [InlineData("downloading weights")]
    [InlineData("{not-json}")]
    public void Parse_IgnoresNonProtocolOutput(string line)
    {
        Assert.Null(TrainingEventJson.Parse(line));
    }
}
