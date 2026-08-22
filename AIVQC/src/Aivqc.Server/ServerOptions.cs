namespace Aivqc.Server;

public sealed class ServerOptions
{
    public const string SectionName = "AivqcServer";

    public string DataDirectory { get; set; } = "/data";

    public string ApiKeysFile { get; set; } = "/run/secrets/aivqc_api_keys.json";

    public long MaximumPackageBytes { get; set; } = 1024L * 1024 * 1024;

    public long StorageQuotaBytes { get; set; } = 5L * 1024 * 1024 * 1024;

    public int AcknowledgedPackageRetentionDays { get; set; } = 30;

    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(DataDirectory)
            || string.IsNullOrWhiteSpace(ApiKeysFile)
            || MaximumPackageBytes is < 1024 or > 10L * 1024 * 1024 * 1024
            || StorageQuotaBytes < MaximumPackageBytes
            || AcknowledgedPackageRetentionDays is < 1 or > 3650)
        {
            throw new InvalidOperationException("AIVQC Server storage configuration is invalid.");
        }
    }
}
