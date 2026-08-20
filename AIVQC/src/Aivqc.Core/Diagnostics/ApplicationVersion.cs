using System.Reflection;

namespace Aivqc.Core.Diagnostics;

/// <summary>
/// Reads the semantic product version embedded in a compiled application.
/// </summary>
public static class ApplicationVersion
{
    public static string DisplayFromAssembly(Assembly assembly)
    {
        return FromAssembly(assembly).Split('+', 2)[0];
    }

    public static string FromAssembly(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);

        var informationalVersion = assembly
            .GetCustomAttribute<AssemblyInformationalVersionAttribute>()?
            .InformationalVersion;

        if (!string.IsNullOrWhiteSpace(informationalVersion))
        {
            return informationalVersion;
        }

        return assembly.GetName().Version?.ToString(3) ?? "unknown";
    }
}
