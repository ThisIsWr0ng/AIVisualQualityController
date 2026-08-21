using System.Text.Json;

namespace Aivqc.Core.Training;

/// <summary>
/// Parses the JSON Lines protocol shared by Trainer and its backend process.
/// </summary>
public static class TrainingEventJson
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);

    public static TrainingJobEvent? Parse(string line)
    {
        if (string.IsNullOrWhiteSpace(line) || line[0] != '{')
        {
            return null;
        }

        try
        {
            return JsonSerializer.Deserialize<TrainingJobEvent>(line, JsonOptions);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}
