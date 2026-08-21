namespace Aivqc.Trainer.Services;

public static class PythonEnvironmentLocator
{
    public static string FindDefault()
    {
        var configuredPath = Environment.GetEnvironmentVariable("AIVQC_PYTHON");
        if (!string.IsNullOrWhiteSpace(configuredPath) && File.Exists(configuredPath))
        {
            return configuredPath;
        }

        foreach (var startPath in new[] { Directory.GetCurrentDirectory(), AppContext.BaseDirectory })
        {
            var directory = new DirectoryInfo(startPath);
            while (directory is not null)
            {
                foreach (var relativePath in new[]
                {
                    Path.Combine("training", ".venv", "Scripts", "python.exe"),
                    Path.Combine("AIVQC", "training", ".venv", "Scripts", "python.exe"),
                })
                {
                    var candidate = Path.Combine(directory.FullName, relativePath);
                    if (File.Exists(candidate))
                    {
                        return candidate;
                    }
                }

                directory = directory.Parent;
            }
        }

        return "py";
    }
}
