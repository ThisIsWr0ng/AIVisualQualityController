using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Text.Json;

namespace Aivqc.Core.Connectivity;

/// <summary>
/// Uses the versioned AIVQC package-routing API for both central and direct endpoints.
/// </summary>
public sealed class AivqcServerApiClient : IDisposable
{
    private readonly HttpClient _httpClient;
    private readonly bool _ownsClient;

    public AivqcServerApiClient(
        AivqcConnectionSettings settings,
        string apiKey,
        HttpClient? httpClient = null)
    {
        ArgumentNullException.ThrowIfNull(settings);
        ArgumentException.ThrowIfNullOrWhiteSpace(apiKey);
        settings.Validate();

        Settings = settings;
        _httpClient = httpClient ?? new HttpClient();
        _ownsClient = httpClient is null;
        _httpClient.BaseAddress = EnsureTrailingSlash(settings.Endpoint);
        _httpClient.DefaultRequestHeaders.Authorization =
            new AuthenticationHeaderValue("Bearer", apiKey);
        _httpClient.DefaultRequestHeaders.Add("X-AIVQC-Client-Id", settings.ClientId);
        _httpClient.DefaultRequestHeaders.UserAgent.ParseAdd("AIVQC-Desktop/1.0");
    }

    public AivqcConnectionSettings Settings { get; }

    public async Task<AivqcServerInfo> GetInfoAsync(CancellationToken cancellationToken = default) =>
        await _httpClient.GetFromJsonAsync<AivqcServerInfo>("api/v1/info", cancellationToken)
        ?? throw new InvalidDataException("The AIVQC endpoint returned an empty server response.");

    public async Task RegisterStationAsync(
        string stationId,
        string stationName,
        CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.PostAsJsonAsync(
            "api/v1/stations",
            new { stationId, name = stationName },
            cancellationToken);
        await EnsureSuccessAsync(response, cancellationToken);
    }

    public async Task<PublishedPackageInfo> PublishPackageAsync(
        string packagePath,
        string targetStationId,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(packagePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(targetStationId);

        await using var stream = File.OpenRead(packagePath);
        using var content = new MultipartFormDataContent();
        using var packageContent = new StreamContent(stream);
        packageContent.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        content.Add(packageContent, "package", Path.GetFileName(packagePath));
        content.Add(new StringContent(targetStationId), "targetStationId");

        using var response = await _httpClient.PostAsync("api/v1/packages", content, cancellationToken);
        await EnsureSuccessAsync(response, cancellationToken);
        return await response.Content.ReadFromJsonAsync<PublishedPackageInfo>(cancellationToken)
            ?? throw new InvalidDataException("The AIVQC endpoint returned empty package metadata.");
    }

    public async Task<PublishedPackageInfo?> GetLatestPackageAsync(
        string stationId,
        CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.GetAsync(
            $"api/v1/stations/{Uri.EscapeDataString(stationId)}/packages/latest",
            cancellationToken);
        if (response.StatusCode == System.Net.HttpStatusCode.NotFound)
        {
            return null;
        }

        await EnsureSuccessAsync(response, cancellationToken);
        return await response.Content.ReadFromJsonAsync<PublishedPackageInfo>(cancellationToken);
    }

    public async Task DownloadPackageAsync(
        string stationId,
        Guid packageId,
        string destinationPath,
        CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.GetAsync(
            $"api/v1/stations/{Uri.EscapeDataString(stationId)}/packages/{packageId:D}/content",
            HttpCompletionOption.ResponseHeadersRead,
            cancellationToken);
        await EnsureSuccessAsync(response, cancellationToken);

        var fullPath = Path.GetFullPath(destinationPath);
        Directory.CreateDirectory(Path.GetDirectoryName(fullPath)!);
        var temporaryPath = $"{fullPath}.{Guid.NewGuid():N}.tmp";
        try
        {
            await using var source = await response.Content.ReadAsStreamAsync(cancellationToken);
            await using var target = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                1024 * 1024,
                useAsync: true);
            await source.CopyToAsync(target, cancellationToken);
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

    public async Task AcknowledgeAsync(
        string stationId,
        Guid packageId,
        PackageAcknowledgementStatus status,
        string? message = null,
        CancellationToken cancellationToken = default)
    {
        using var response = await _httpClient.PostAsJsonAsync(
            $"api/v1/stations/{Uri.EscapeDataString(stationId)}/packages/{packageId:D}/acknowledgements",
            new PackageAcknowledgementRequest(status, message),
            cancellationToken);
        await EnsureSuccessAsync(response, cancellationToken);
    }

    public void Dispose()
    {
        if (_ownsClient)
        {
            _httpClient.Dispose();
        }
    }

    private static async Task EnsureSuccessAsync(
        HttpResponseMessage response,
        CancellationToken cancellationToken)
    {
        if (response.IsSuccessStatusCode)
        {
            return;
        }

        var body = await response.Content.ReadAsStringAsync(cancellationToken);
        string? detail = null;
        try
        {
            detail = JsonDocument.Parse(body).RootElement.TryGetProperty("detail", out var value)
                ? value.GetString()
                : null;
        }
        catch (JsonException)
        {
            // Do not expose an arbitrary server error page to the desktop UI.
        }

        throw new HttpRequestException(
            $"AIVQC endpoint returned {(int)response.StatusCode} ({response.ReasonPhrase})"
            + (string.IsNullOrWhiteSpace(detail) ? "." : $": {detail}"),
            null,
            response.StatusCode);
    }

    private static Uri EnsureTrailingSlash(Uri endpoint) =>
        endpoint.AbsoluteUri.EndsWith('/') ? endpoint : new Uri(endpoint.AbsoluteUri + "/");
}

public sealed record AivqcServerInfo(string Name, string ApiVersion, string ServerVersion, DateTimeOffset UtcNow);

public sealed record PublishedPackageInfo(
    Guid PackageId,
    string ProductId,
    string RecipeId,
    string TargetStationId,
    DateTimeOffset PublishedAtUtc,
    bool Revoked,
    long SizeBytes,
    string Sha256);

public sealed record PackageAcknowledgementRequest(PackageAcknowledgementStatus Status, string? Message);

public enum PackageAcknowledgementStatus
{
    Downloaded,
    Activated,
    Failed,
}
