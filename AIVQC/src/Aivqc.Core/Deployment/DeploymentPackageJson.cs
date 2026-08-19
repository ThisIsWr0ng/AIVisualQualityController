using System.Text.Json;
using System.Text.Json.Serialization;

namespace Aivqc.Core.Deployment;

public static class DeploymentPackageJson
{
    private static readonly JsonSerializerOptions SerializerOptions = CreateOptions();

    public static string Serialize(DeploymentPackageManifest manifest) =>
        JsonSerializer.Serialize(manifest, SerializerOptions);

    public static DeploymentPackageManifest Deserialize(string json) =>
        JsonSerializer.Deserialize<DeploymentPackageManifest>(json, SerializerOptions)
        ?? throw new JsonException("The deployment package manifest is empty or invalid.");

    private static JsonSerializerOptions CreateOptions()
    {
        var options = new JsonSerializerOptions(JsonSerializerDefaults.Web)
        {
            WriteIndented = true
        };

        options.Converters.Add(new JsonStringEnumConverter(JsonNamingPolicy.CamelCase));
        return options;
    }
}
