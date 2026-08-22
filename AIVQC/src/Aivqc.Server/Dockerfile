FROM mcr.microsoft.com/dotnet/sdk:10.0-noble AS build
WORKDIR /source

COPY Directory.Build.props Version.props ./
COPY src/Aivqc.Core/Aivqc.Core.csproj src/Aivqc.Core/
COPY src/Aivqc.Server/Aivqc.Server.csproj src/Aivqc.Server/
RUN dotnet restore src/Aivqc.Server/Aivqc.Server.csproj

COPY src/Aivqc.Core/ src/Aivqc.Core/
COPY src/Aivqc.Server/ src/Aivqc.Server/
RUN dotnet publish src/Aivqc.Server/Aivqc.Server.csproj \
    --configuration Release \
    --output /app/publish \
    --no-restore \
    /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:10.0-noble AS runtime
WORKDIR /app
ENV ASPNETCORE_HTTP_PORTS=8080 \
    DOTNET_EnableDiagnostics=0 \
    AivqcServer__DataDirectory=/data \
    AivqcServer__ApiKeysFile=/run/secrets/aivqc_api_keys.json

RUN mkdir -p /data /run/secrets && chown -R app:app /data /run/secrets
COPY --from=build --chown=app:app /app/publish .
USER app
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=6s --start-period=15s --retries=3 \
    CMD ["dotnet", "Aivqc.Server.dll", "--health-check", "http://127.0.0.1:8080/health/live"]
ENTRYPOINT ["dotnet", "Aivqc.Server.dll"]
