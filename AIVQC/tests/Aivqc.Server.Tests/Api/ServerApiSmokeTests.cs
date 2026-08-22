using System.Diagnostics;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Aivqc.Server.Security;

namespace Aivqc.Server.Tests.Api;

public sealed class ServerApiSmokeTests
{
    [Fact]
    public async Task RunningServer_RequiresAuthenticationAndRegistersStation()
    {
        var root = CreateTemporaryDirectory();
        Process? process = null;

        try
        {
            const string trainerToken = "integration-test-trainer-token";
            const string productionToken = "integration-test-production-token";
            var apiKeysPath = Path.Combine(root, "api-keys.json");
            var clients = new ApiClientFile(
            [
                CreateClient("trainer-test", trainerToken, [AivqcRoles.Trainer], null),
                CreateClient("production-test", productionToken, [AivqcRoles.Production], "line-1"),
            ]);
            await File.WriteAllTextAsync(
                apiKeysPath,
                JsonSerializer.Serialize(clients, new JsonSerializerOptions(JsonSerializerDefaults.Web)));

            var port = GetAvailablePort();
            var endpoint = new Uri($"http://127.0.0.1:{port}/");
            var serverAssembly = Path.Combine(AppContext.BaseDirectory, "Aivqc.Server.dll");
            process = Process.Start(new ProcessStartInfo
            {
                FileName = "dotnet",
                UseShellExecute = false,
                CreateNoWindow = true,
                ArgumentList =
                {
                    serverAssembly,
                    "--urls", endpoint.AbsoluteUri,
                    "--AivqcServer:DataDirectory", Path.Combine(root, "data"),
                    "--AivqcServer:ApiKeysFile", apiKeysPath,
                    "--AivqcServer:MaximumPackageBytes", "10485760",
                    "--AivqcServer:StorageQuotaBytes", "20971520",
                },
            }) ?? throw new InvalidOperationException("The test server could not be started.");

            using var client = new HttpClient { BaseAddress = endpoint };
            await WaitUntilHealthyAsync(client, process);

            using var anonymousInfo = await client.GetAsync("api/v1/info");
            Assert.Equal(HttpStatusCode.Unauthorized, anonymousInfo.StatusCode);

            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", trainerToken);
            client.DefaultRequestHeaders.Add("X-AIVQC-Client-Id", "trainer-test");
            using var info = await client.GetAsync("api/v1/info");
            Assert.Equal(HttpStatusCode.OK, info.StatusCode);
            using var registration = await client.PostAsJsonAsync(
                "api/v1/stations",
                new { stationId = "line-1", name = "Integration test line" });
            Assert.Equal(HttpStatusCode.Created, registration.StatusCode);

            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", productionToken);
            client.DefaultRequestHeaders.Remove("X-AIVQC-Client-Id");
            client.DefaultRequestHeaders.Add("X-AIVQC-Client-Id", "production-test");
            using var latest = await client.GetAsync("api/v1/stations/line-1/packages/latest");
            Assert.Equal(HttpStatusCode.NotFound, latest.StatusCode);
            using var otherStation = await client.GetAsync("api/v1/stations/line-2/packages/latest");
            Assert.Equal(HttpStatusCode.Forbidden, otherStation.StatusCode);
        }
        finally
        {
            if (process is { HasExited: false })
            {
                process.Kill(entireProcessTree: true);
                await process.WaitForExitAsync();
            }

            process?.Dispose();
            Directory.Delete(root, recursive: true);
        }
    }

    private static ApiClientDefinition CreateClient(
        string id,
        string token,
        IReadOnlyList<string> roles,
        string? stationId) =>
        new(
            id,
            Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(token))),
            roles,
            stationId);

    private static async Task WaitUntilHealthyAsync(HttpClient client, Process process)
    {
        for (var attempt = 0; attempt < 100; attempt++)
        {
            if (process.HasExited)
            {
                throw new InvalidOperationException($"AIVQC Server exited with code {process.ExitCode}.");
            }

            try
            {
                using var response = await client.GetAsync("health/live");
                if (response.IsSuccessStatusCode)
                {
                    return;
                }
            }
            catch (HttpRequestException)
            {
                // Kestrel is still starting.
            }

            await Task.Delay(100);
        }

        throw new TimeoutException("AIVQC Server did not become healthy within 10 seconds.");
    }

    private static int GetAvailablePort()
    {
        var listener = new TcpListener(IPAddress.Loopback, 0);
        listener.Start();
        var port = ((IPEndPoint)listener.LocalEndpoint).Port;
        listener.Stop();
        return port;
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-server-api-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
