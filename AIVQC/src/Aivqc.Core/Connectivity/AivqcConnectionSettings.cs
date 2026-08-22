namespace Aivqc.Core.Connectivity;

/// <summary>
/// Selects whether a desktop client uses the central Server or a compatible direct peer endpoint.
/// </summary>
public sealed record AivqcConnectionSettings(
    AivqcConnectionMode Mode,
    Uri Endpoint,
    string ClientId,
    string? StationId,
    bool AllowInsecureHttp = false)
{
    public void Validate()
    {
        if (!Endpoint.IsAbsoluteUri
            || (Endpoint.Scheme != Uri.UriSchemeHttps
                && !(AllowInsecureHttp && Endpoint.Scheme == Uri.UriSchemeHttp)))
        {
            throw new InvalidOperationException(
                "AIVQC connections require an absolute HTTPS endpoint unless insecure HTTP is explicitly enabled for local setup.");
        }

        ValidateIdentifier(ClientId, "client ID");
        if (Mode == AivqcConnectionMode.Server && string.IsNullOrWhiteSpace(StationId))
        {
            throw new InvalidOperationException("Server connections require a target station ID.");
        }

        if (!string.IsNullOrWhiteSpace(StationId))
        {
            ValidateIdentifier(StationId, "station ID");
        }
    }

    private static void ValidateIdentifier(string value, string description)
    {
        if (string.IsNullOrWhiteSpace(value)
            || value.Length > 128
            || value.Any(character =>
                !(char.IsAsciiLetterOrDigit(character) || character is '-' or '_' or '.')))
        {
            throw new InvalidOperationException(
                $"The AIVQC {description} must contain only letters, numbers, hyphens, underscores, or dots.");
        }
    }
}

public enum AivqcConnectionMode
{
    Server,
    Direct,
}
