using System.Text;
using System.Text.Json;

namespace Aivqc.Trainer.Services;

public sealed class TrainerSettingsStore
{
    public const int CurrentSchemaVersion = 1;

    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web)
    {
        WriteIndented = true,
    };

    private readonly string _filePath;

    public TrainerSettingsStore(string? filePath = null)
    {
        _filePath = filePath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AIVQC",
            "Trainer",
            "settings.json");
    }

    public TrainerApplicationSettings Load()
    {
        if (!File.Exists(_filePath))
        {
            return TrainerApplicationSettings.CreateDefault();
        }

        try
        {
            var settings = JsonSerializer.Deserialize<TrainerApplicationSettings>(
                File.ReadAllText(_filePath, Encoding.UTF8),
                JsonOptions)
                ?? throw new InvalidDataException("The Trainer settings file is empty.");
            Validate(settings);
            return settings;
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException("The Trainer settings file contains invalid JSON.", exception);
        }
    }

    public void Save(TrainerApplicationSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        Validate(settings);

        var fullPath = Path.GetFullPath(_filePath);
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

    private static void Validate(TrainerApplicationSettings settings)
    {
        if (settings.SchemaVersion != CurrentSchemaVersion)
        {
            throw new InvalidDataException($"Unsupported Trainer settings schema {settings.SchemaVersion}.");
        }

        if (settings.Lines.Count == 0)
        {
            throw new InvalidDataException("At least one production line must be configured.");
        }

        if (settings.Lines.Any(line =>
                string.IsNullOrWhiteSpace(line.Id)
                || string.IsNullOrWhiteSpace(line.Name)))
        {
            throw new InvalidDataException("Every production line requires an ID and name.");
        }

        if (settings.Lines.Select(line => line.Id).Distinct(StringComparer.OrdinalIgnoreCase).Count()
            != settings.Lines.Count)
        {
            throw new InvalidDataException("Production line IDs must be unique.");
        }

        if (!settings.Lines.Any(line => string.Equals(
                line.Id,
                settings.SelectedLineId,
                StringComparison.OrdinalIgnoreCase)))
        {
            throw new InvalidDataException("The selected production line does not exist.");
        }
    }
}

public sealed record TrainerApplicationSettings(
    int SchemaVersion,
    string SelectedLineId,
    bool ExpertModeEnabled,
    IReadOnlyList<DeploymentLineSettings> Lines)
{
    public static TrainerApplicationSettings CreateDefault() => new(
        TrainerSettingsStore.CurrentSchemaVersion,
        "line-1",
        false,
        [new DeploymentLineSettings("line-1", "Production line 1", [], 0, null)]);
}

public sealed record DeploymentLineSettings(
    string Id,
    string Name,
    IReadOnlyList<string> ProductIds,
    int DeploymentCount,
    DateTimeOffset? LastDeploymentUtc);
