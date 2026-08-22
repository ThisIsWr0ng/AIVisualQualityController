using System.Text;
using System.Text.Json;

namespace Aivqc.Trainer.Services;

public sealed class RecentProjectStore
{
    private const int MaximumEntries = 8;
    private readonly string _filePath;

    public RecentProjectStore(string? filePath = null)
    {
        _filePath = filePath ?? Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
            "AIVQC",
            "Trainer",
            "recent-projects.json");
    }

    public IReadOnlyList<RecentProjectEntry> Load()
    {
        if (!File.Exists(_filePath))
        {
            return [];
        }

        try
        {
            return JsonSerializer.Deserialize<List<RecentProjectEntry>>(File.ReadAllText(_filePath))
                ?.OrderByDescending(entry => entry.LastOpenedAtUtc)
                .Take(MaximumEntries)
                .ToArray()
                ?? [];
        }
        catch (Exception exception) when (
            exception is JsonException or IOException or UnauthorizedAccessException)
        {
            return [];
        }
    }

    public IReadOnlyList<RecentProjectEntry> Add(string name, string projectDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(projectDirectory);

        var normalizedPath = Path.GetFullPath(projectDirectory);
        var entries = Load()
            .Where(entry => !string.Equals(entry.ProjectDirectory, normalizedPath, StringComparison.OrdinalIgnoreCase))
            .Prepend(new RecentProjectEntry(name, normalizedPath, DateTimeOffset.UtcNow))
            .Take(MaximumEntries)
            .ToArray();
        Save(entries);
        return entries;
    }

    private void Save(IReadOnlyList<RecentProjectEntry> entries)
    {
        var directory = Path.GetDirectoryName(_filePath)
            ?? throw new InvalidOperationException("The recent-project file directory could not be resolved.");
        Directory.CreateDirectory(directory);
        var temporaryPath = Path.Combine(directory, $".{Path.GetFileName(_filePath)}.{Guid.NewGuid():N}.tmp");

        try
        {
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(entries, new JsonSerializerOptions { WriteIndented = true }),
                new UTF8Encoding(encoderShouldEmitUTF8Identifier: false));
            File.Move(temporaryPath, _filePath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
            {
                File.Delete(temporaryPath);
            }
        }
    }
}

public sealed record RecentProjectEntry(
    string Name,
    string ProjectDirectory,
    DateTimeOffset LastOpenedAtUtc);
