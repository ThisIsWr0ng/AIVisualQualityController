using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.AspNetCore.Authentication;
using Microsoft.Extensions.Options;

namespace Aivqc.Server.Security;

public static class AivqcRoles
{
    public const string Administrator = "administrator";
    public const string Trainer = "trainer";
    public const string Production = "production";
}

public sealed record ApiClientDefinition(
    string Id,
    string KeySha256,
    IReadOnlyList<string> Roles,
    string? StationId);

public sealed record ApiClientFile(IReadOnlyList<ApiClientDefinition> Clients);

public sealed class ApiKeyStore
{
    private readonly IReadOnlyList<ApiClientDefinition> _clients;

    public ApiKeyStore(IOptions<ServerOptions> options)
    {
        var path = Path.GetFullPath(options.Value.ApiKeysFile);
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                "The API key definition file is missing. Mount a server secret before startup.",
                path);
        }

        var configuration = JsonSerializer.Deserialize<ApiClientFile>(
            File.ReadAllText(path),
            new JsonSerializerOptions(JsonSerializerDefaults.Web))
            ?? throw new InvalidDataException("The API key definition file is empty.");
        Validate(configuration.Clients);
        _clients = configuration.Clients;
    }

    public ApiClientDefinition? Authenticate(string rawApiKey)
    {
        var candidateHash = SHA256.HashData(Encoding.UTF8.GetBytes(rawApiKey));
        foreach (var client in _clients)
        {
            var configuredHash = Convert.FromHexString(client.KeySha256);
            if (CryptographicOperations.FixedTimeEquals(candidateHash, configuredHash))
            {
                return client;
            }
        }

        return null;
    }

    private static void Validate(IReadOnlyList<ApiClientDefinition>? clients)
    {
        var allowedRoles = new HashSet<string>(StringComparer.OrdinalIgnoreCase)
        {
            AivqcRoles.Administrator,
            AivqcRoles.Trainer,
            AivqcRoles.Production,
        };
        if (clients is null
            || clients.Count == 0
            || clients.Select(client => client.Id).Distinct(StringComparer.OrdinalIgnoreCase).Count() != clients.Count)
        {
            throw new InvalidDataException("At least one uniquely identified API client is required.");
        }

        foreach (var client in clients)
        {
            if (string.IsNullOrWhiteSpace(client.Id)
                || client.Id.Length > 128
                || client.KeySha256.Length != 64
                || !client.KeySha256.All(Uri.IsHexDigit)
                || client.Roles is null
                || client.Roles.Count == 0
                || client.Roles.Any(role => !allowedRoles.Contains(role))
                || (client.Roles.Contains(AivqcRoles.Production, StringComparer.OrdinalIgnoreCase)
                    && string.IsNullOrWhiteSpace(client.StationId)))
            {
                throw new InvalidDataException($"API client '{client.Id}' is invalid.");
            }
        }
    }
}

public sealed class ApiKeyAuthenticationHandler(
    IOptionsMonitor<AuthenticationSchemeOptions> options,
    ILoggerFactory logger,
    UrlEncoder encoder,
    ApiKeyStore apiKeyStore)
    : AuthenticationHandler<AuthenticationSchemeOptions>(options, logger, encoder)
{
    public const string SchemeName = "AivqcApiKey";

    protected override Task<AuthenticateResult> HandleAuthenticateAsync()
    {
        const string bearerPrefix = "Bearer ";
        var authorization = Request.Headers.Authorization.ToString();
        if (!authorization.StartsWith(bearerPrefix, StringComparison.OrdinalIgnoreCase))
        {
            return Task.FromResult(AuthenticateResult.NoResult());
        }

        var rawApiKey = authorization[bearerPrefix.Length..].Trim();
        var client = string.IsNullOrWhiteSpace(rawApiKey) ? null : apiKeyStore.Authenticate(rawApiKey);
        if (client is null)
        {
            return Task.FromResult(AuthenticateResult.Fail("The API key is invalid."));
        }

        var declaredClientId = Request.Headers["X-AIVQC-Client-Id"].ToString();
        if (!string.Equals(declaredClientId, client.Id, StringComparison.Ordinal))
        {
            return Task.FromResult(AuthenticateResult.Fail("The declared client ID does not match the API key."));
        }

        var claims = new List<Claim>
        {
            new(ClaimTypes.NameIdentifier, client.Id),
            new(ClaimTypes.Name, client.Id),
        };
        claims.AddRange(client.Roles.Select(role => new Claim(ClaimTypes.Role, role)));
        if (!string.IsNullOrWhiteSpace(client.StationId))
        {
            claims.Add(new Claim("station_id", client.StationId));
        }

        var principal = new ClaimsPrincipal(new ClaimsIdentity(claims, SchemeName));
        return Task.FromResult(AuthenticateResult.Success(
            new AuthenticationTicket(principal, SchemeName)));
    }
}
