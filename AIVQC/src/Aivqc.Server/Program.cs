using System.Security.Claims;
using System.Threading.RateLimiting;
using Aivqc.Core.Connectivity;
using Aivqc.Server;
using Aivqc.Server.Security;
using Aivqc.Server.Storage;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.RateLimiting;

if (args.FirstOrDefault() == "--health-check")
{
    return await RunHealthCheckAsync(args.ElementAtOrDefault(1));
}

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddProblemDetails();
builder.Services.AddOptions<ServerOptions>()
    .Bind(builder.Configuration.GetSection(ServerOptions.SectionName))
    .Validate(options =>
    {
        try
        {
            options.Validate();
            return true;
        }
        catch
        {
            return false;
        }
    }, "AIVQC Server options are invalid.")
    .ValidateOnStart();
var configuredOptions = builder.Configuration
    .GetSection(ServerOptions.SectionName)
    .Get<ServerOptions>() ?? new ServerOptions();
builder.Services.Configure<FormOptions>(options =>
    options.MultipartBodyLengthLimit = configuredOptions.MaximumPackageBytes + 1024 * 1024);
builder.WebHost.ConfigureKestrel(options =>
    options.Limits.MaxRequestBodySize = configuredOptions.MaximumPackageBytes + 1024 * 1024);

builder.Services.AddSingleton<ApiKeyStore>();
builder.Services.AddSingleton<ServerPackageStore>();
builder.Services.AddAuthentication(ApiKeyAuthenticationHandler.SchemeName)
    .AddScheme<AuthenticationSchemeOptions, ApiKeyAuthenticationHandler>(
        ApiKeyAuthenticationHandler.SchemeName,
        _ => { });
builder.Services.AddAuthorizationBuilder()
    .AddPolicy("trainer", policy => policy.RequireRole(AivqcRoles.Trainer, AivqcRoles.Administrator))
    .AddPolicy("production", policy => policy.RequireRole(
        AivqcRoles.Production,
        AivqcRoles.Trainer,
        AivqcRoles.Administrator))
    .AddPolicy("administrator", policy => policy.RequireRole(AivqcRoles.Administrator));
builder.Services.AddRateLimiter(options =>
{
    options.RejectionStatusCode = StatusCodes.Status429TooManyRequests;
    options.GlobalLimiter = PartitionedRateLimiter.Create<HttpContext, string>(context =>
        RateLimitPartition.GetFixedWindowLimiter(
            context.Connection.RemoteIpAddress?.ToString() ?? "unknown",
            _ => new FixedWindowRateLimiterOptions
            {
                PermitLimit = 120,
                Window = TimeSpan.FromMinutes(1),
                QueueLimit = 0,
            }));
});

var app = builder.Build();
_ = app.Services.GetRequiredService<ApiKeyStore>();
_ = app.Services.GetRequiredService<ServerPackageStore>();

app.UseExceptionHandler(exceptionHandlerApp =>
{
    exceptionHandlerApp.Run(async context =>
    {
        var exception = context.Features.Get<Microsoft.AspNetCore.Diagnostics.IExceptionHandlerFeature>()?.Error;
        var (status, title) = exception switch
        {
            KeyNotFoundException => (StatusCodes.Status404NotFound, "Resource not found"),
            FileNotFoundException => (StatusCodes.Status404NotFound, "Transfer file not found"),
            UnauthorizedAccessException => (StatusCodes.Status403Forbidden, "Access denied"),
            InvalidDataException or InvalidOperationException or ArgumentException =>
                (StatusCodes.Status400BadRequest, "Request rejected"),
            IOException => (StatusCodes.Status507InsufficientStorage, "Storage operation failed"),
            _ => (StatusCodes.Status500InternalServerError, "Server error"),
        };
        context.Response.StatusCode = status;
        await Results.Problem(
            statusCode: status,
            title: title,
            detail: status == StatusCodes.Status500InternalServerError
                ? "The request could not be completed."
                : exception?.Message).ExecuteAsync(context);
    });
});
app.UseRateLimiter();
app.UseAuthentication();
app.UseAuthorization();

app.MapGet("/health/live", () => Results.Ok(new
{
    status = "healthy",
    utcNow = DateTimeOffset.UtcNow,
})).AllowAnonymous();

app.MapGet("/api/v1/info", () => new AivqcServerInfo(
    "AIVQC Server",
    "v1",
    Aivqc.Core.Diagnostics.ApplicationVersion.DisplayFromAssembly(typeof(Program).Assembly),
    DateTimeOffset.UtcNow)).RequireAuthorization();

app.MapGet("/api/v1/stations", (ServerPackageStore store) =>
    Results.Ok(store.GetState().Stations)).RequireAuthorization("trainer");

app.MapPost("/api/v1/stations", async (
    RegisterStationRequest request,
    ClaimsPrincipal principal,
    ServerPackageStore store,
    CancellationToken cancellationToken) =>
{
    var station = await store.RegisterStationAsync(
        request.StationId,
        request.Name,
        GetActor(principal),
        cancellationToken);
    return Results.Created($"/api/v1/stations/{station.StationId}", station);
}).RequireAuthorization("trainer");

app.MapPost("/api/v1/packages", async (
    HttpRequest request,
    ClaimsPrincipal principal,
    ServerPackageStore store,
    Microsoft.Extensions.Options.IOptions<ServerOptions> options,
    CancellationToken cancellationToken) =>
{
    if (!request.HasFormContentType)
    {
        return Results.Problem(statusCode: 400, detail: "Use multipart/form-data.");
    }

    var form = await request.ReadFormAsync(cancellationToken);
    var package = form.Files.GetFile("package");
    var targetStationId = form["targetStationId"].ToString();
    if (package is null
        || string.IsNullOrWhiteSpace(targetStationId)
        || !package.FileName.EndsWith(".aivqcpkg", StringComparison.OrdinalIgnoreCase)
        || package.Length <= 0
        || package.Length > options.Value.MaximumPackageBytes)
    {
        return Results.Problem(statusCode: 400, detail: "A valid AIVQC package and target station are required.");
    }

    var incomingDirectory = Path.Combine(Path.GetFullPath(options.Value.DataDirectory), "incoming");
    Directory.CreateDirectory(incomingDirectory);
    var temporaryPath = Path.Combine(incomingDirectory, $"{Guid.NewGuid():N}.aivqcpkg.tmp");
    try
    {
        await using (var output = new FileStream(
            temporaryPath,
            FileMode.CreateNew,
            FileAccess.Write,
            FileShare.None,
            1024 * 1024,
            useAsync: true))
        {
            await package.CopyToAsync(output, cancellationToken);
        }

        var result = await store.PublishAsync(
            temporaryPath,
            targetStationId,
            GetActor(principal),
            cancellationToken);
        return Results.Created(
            $"/api/v1/packages/{result.Package.PackageId:D}",
            ToPublishedPackageInfo(result.Package, result.Assignment));
    }
    finally
    {
        if (File.Exists(temporaryPath))
        {
            File.Delete(temporaryPath);
        }
    }
}).DisableAntiforgery().RequireAuthorization("trainer");

app.MapGet("/api/v1/stations/{stationId}/packages/latest", (
    string stationId,
    ClaimsPrincipal principal,
    ServerPackageStore store) =>
{
    EnsureStationAccess(principal, stationId);
    var result = store.GetLatest(stationId);
    return result is null
        ? Results.NotFound()
        : Results.Ok(ToPublishedPackageInfo(result.Value.Package, result.Value.Assignment));
}).RequireAuthorization("production");

app.MapGet("/api/v1/stations/{stationId}/packages/{packageId:guid}/content", (
    string stationId,
    Guid packageId,
    ClaimsPrincipal principal,
    ServerPackageStore store) =>
{
    EnsureStationAccess(principal, stationId);
    var path = store.GetPackageContentPath(stationId, packageId);
    return Results.File(path, "application/octet-stream", $"{packageId:D}.aivqcpkg");
}).RequireAuthorization("production");

app.MapPost("/api/v1/stations/{stationId}/packages/{packageId:guid}/acknowledgements", async (
    string stationId,
    Guid packageId,
    PackageAcknowledgementRequest request,
    ClaimsPrincipal principal,
    ServerPackageStore store,
    CancellationToken cancellationToken) =>
{
    EnsureStationAccess(principal, stationId);
    await store.AcknowledgeAsync(
        stationId,
        packageId,
        new PackageAcknowledgement(request.Status, request.Message),
        GetActor(principal),
        cancellationToken);
    return Results.NoContent();
}).RequireAuthorization("production");

app.MapPost("/api/v1/packages/{packageId:guid}/revoke", async (
    Guid packageId,
    ClaimsPrincipal principal,
    ServerPackageStore store,
    CancellationToken cancellationToken) =>
{
    await store.RevokeAsync(packageId, GetActor(principal), cancellationToken);
    return Results.NoContent();
}).RequireAuthorization("administrator");

await app.RunAsync();
return 0;

static PublishedPackageInfo ToPublishedPackageInfo(StoredPackage package, PackageAssignment assignment) =>
    new(
        package.PackageId,
        package.ProductId,
        package.RecipeId,
        assignment.StationId,
        package.PublishedAtUtc,
        package.Revoked,
        package.SizeBytes,
        package.Sha256);

static string GetActor(ClaimsPrincipal principal) =>
    principal.FindFirstValue(ClaimTypes.NameIdentifier)
    ?? throw new UnauthorizedAccessException("The authenticated client has no identity.");

static void EnsureStationAccess(ClaimsPrincipal principal, string stationId)
{
    if (principal.IsInRole(AivqcRoles.Trainer) || principal.IsInRole(AivqcRoles.Administrator))
    {
        return;
    }

    if (!string.Equals(
        principal.FindFirstValue("station_id"),
        stationId,
        StringComparison.OrdinalIgnoreCase))
    {
        throw new UnauthorizedAccessException("A Production client can access only its assigned station.");
    }
}

static async Task<int> RunHealthCheckAsync(string? endpoint)
{
    using var client = new HttpClient { Timeout = TimeSpan.FromSeconds(5) };
    try
    {
        using var response = await client.GetAsync(endpoint ?? "http://127.0.0.1:8080/health/live");
        return response.IsSuccessStatusCode ? 0 : 1;
    }
    catch
    {
        return 1;
    }
}

public sealed record RegisterStationRequest(string StationId, string Name);

public partial class Program
{
}
