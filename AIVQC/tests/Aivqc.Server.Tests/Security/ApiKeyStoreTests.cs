using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Aivqc.Server;
using Aivqc.Server.Security;
using Microsoft.Extensions.Options;

namespace Aivqc.Server.Tests.Security;

public sealed class ApiKeyStoreTests
{
    [Fact]
    public void Authenticate_UsesHashedKeysAndReturnsUniqueClientIdentity()
    {
        var root = CreateTemporaryDirectory();

        try
        {
            const string rawToken = "test-token-that-is-not-stored-in-the-file";
            var path = Path.Combine(root, "api-keys.json");
            var configuration = new ApiClientFile(
            [
                new ApiClientDefinition(
                    "production-line-1",
                    Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(rawToken))),
                    [AivqcRoles.Production],
                    "line-1"),
            ]);
            File.WriteAllText(path, JsonSerializer.Serialize(
                configuration,
                new JsonSerializerOptions(JsonSerializerDefaults.Web)));
            var store = new ApiKeyStore(Options.Create(new ServerOptions { ApiKeysFile = path }));

            var client = store.Authenticate(rawToken);

            Assert.NotNull(client);
            Assert.Equal("production-line-1", client.Id);
            Assert.Equal("line-1", client.StationId);
            Assert.Null(store.Authenticate("wrong-token"));
            Assert.DoesNotContain(rawToken, File.ReadAllText(path), StringComparison.Ordinal);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    private static string CreateTemporaryDirectory()
    {
        var path = Path.Combine(Path.GetTempPath(), $"aivqc-api-key-tests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }
}
