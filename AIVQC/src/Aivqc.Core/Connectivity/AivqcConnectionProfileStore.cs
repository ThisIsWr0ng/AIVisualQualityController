using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aivqc.Core.Connectivity;

/// <summary>
/// Persists non-secret connection preferences. API keys are intentionally excluded.
/// </summary>
public static class AivqcConnectionProfileStore
{
    private static readonly JsonSerializerOptions JsonOptions = CreateJsonOptions();

    public static AivqcConnectionSettings? Load(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            var settings = JsonSerializer.Deserialize<AivqcConnectionSettings>(
                File.ReadAllText(path, Encoding.UTF8),
                JsonOptions)
                ?? throw new InvalidDataException("The connection profile is empty.");
            settings.Validate();
            return settings;
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The connection profile contains invalid JSON.", exception);
        }
    }

    public static void Save(string path, AivqcConnectionSettings settings)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(settings);
        settings.Validate();

        var fullPath = Path.GetFullPath(path);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        var temporaryPath = $"{fullPath}.{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(settings, JsonOptions),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }

    private static JsonSerializerOptions CreateJsonOptions()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true,
        };
        options.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        return options;
    }
}
