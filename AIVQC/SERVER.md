# AIVQC Server

AIVQC Server is the on-premises package-routing service between Trainer and
Production. Production inspections do not depend on a live server connection:
each station downloads a complete `.aivqcpkg`, verifies it locally, activates
it explicitly, and retains it in its local cache for offline operation.

## Implemented API scope

- versioned `/api/v1` endpoints,
- unique API client identities and SHA-256 API-key verification,
- Trainer, Production, and Administrator roles,
- Production station registration,
- integrity-checked package publication to one selected station,
- latest-package discovery and bounded package download,
- downloaded, activated, and failed acknowledgements,
- package revocation,
- atomic JSON state, append-only JSON Lines audit events, storage quotas, and
  upload limits,
- anonymous liveness endpoint containing no protected application data,
- per-source request rate limiting.

This is the first server increment. Central user/password authentication, MFA,
package signatures, automatic transfer-file expiry, sample upload, and a Server
administration UI remain release requirements and are not represented as
complete here.

## Connection modes

Both desktop applications expose two modes:

- **AIVQC Server** uses the central Docker service and routes packages by
  station ID.
- **Direct connection** uses the same versioned HTTPS API contract against a
  compatible peer endpoint. The setting and shared client contract are ready;
  the embedded direct HTTP listener in Production is not enabled in this
  increment. Until that listener is implemented, select AIVQC Server.

Only endpoint, mode, client ID, station ID, and the explicit HTTP-development
flag are persisted. API keys remain in memory for the current desktop session.

## Local Docker quick start

From `AIVQC/deploy/server`:

```powershell
Copy-Item .env.example .env
.\generate-api-keys.ps1
docker compose build
docker compose up -d
docker compose ps
```

The script displays three raw tokens once and writes only their SHA-256 hashes
to `secrets/api-keys.json`. Store the raw values in a password manager. The
secret file, runtime data, and `.env` are excluded from Git.

The Compose default publishes the server only on `127.0.0.1:8088`. For a short
isolated-LAN test you can set `AIVQC_BIND_ADDRESS` to the NAS address and enable
**Allow HTTP for isolated setup** in the desktop clients. Normal use should put
the service behind a trusted HTTPS reverse proxy and leave the loopback binding
unchanged.

## Synology Container Manager

The deployment targets DSM 7.2 Container Manager projects. Synology supports
creating a project from an uploaded or editor-provided `docker-compose.yml`:
<https://kb.synology.com/en-us/DSM/help/ContainerManager/docker_project>.

1. Confirm that the NAS model supports Container Manager and has enough space
   for the configured transfer quota.
2. Copy the entire repository, or at minimum the `AIVQC` build context, to a
   protected NAS share.
3. Open `AIVQC/deploy/server` over SSH and run:

   ```sh
   cp .env.example .env
   chmod +x generate-api-keys.sh
   ./generate-api-keys.sh
   mkdir -p data
   id
   ```

4. Put the account UID and GID reported by `id` into `.env`. That account must
   own `data` and be able to read `secrets/api-keys.json`.
5. In **Container Manager → Project → Create**, select
   `deploy/server` as the project path and use the included Compose file.
6. Build and start the project. The multi-stage image uses Microsoft's official
   .NET 10 SDK only for compilation and the smaller ASP.NET runtime for the
   final container. Official .NET images and their platform variants are
   documented at
   <https://learn.microsoft.com/en-us/dotnet/core/docker/container-images>.
7. In DSM **Login Portal → Advanced → Reverse Proxy**, create an HTTPS endpoint
   whose destination is `http://127.0.0.1:8088`. Install a certificate trusted
   by every Trainer and Production computer. Do not disable certificate
   validation in clients.
8. Back up `deploy/server/data`, `deploy/server/secrets`, and the offline copy of
   the raw API tokens. Test restoration before relying on the server.

If you do not use a reverse proxy, ASP.NET Core can also terminate HTTPS with a
mounted certificate. Microsoft's supported container HTTPS configuration is
documented at
<https://learn.microsoft.com/en-us/aspnet/core/security/docker-https?view=aspnetcore-10.0>.

## First end-to-end route

1. Start the server and copy the `trainer-main` token into Trainer.
2. Set mode to **AIVQC Server**, the HTTPS endpoint, client ID
   `trainer-main`, and station ID `line-1`.
3. Test the connection and register the station.
4. Export a deployment package, then select **Publish last package**.
5. In Production, enter the Production token, client ID
   `production-line-1`, and station ID `line-1`.
6. Test the connection and select **Sync latest**.
7. Production downloads the package, validates its archive and ONNX contract,
   activates it locally, and acknowledges activation to Server.

## Storage and backup

The bind-mounted `data` directory contains:

```text
data/
├── server-state.json       # atomic routing and acknowledgement state
├── audit.jsonl             # security and lifecycle events
├── packages/               # routed .aivqcpkg transfer files
├── incoming/               # temporary uploads, normally empty
└── verification-cache/     # verified extracted package content
```

Default limits are a 1 GiB package, 5 GiB total package storage, and a 30-day
acknowledged-package retention setting. Automated expiry is not implemented
yet, so monitor storage and do not assume that the retention setting already
deletes files.

Never expose the container port directly to the Internet. Restrict the reverse
proxy and firewall to the authorized Trainer and Production networks. Rotate a
token by replacing its hash in `api-keys.json` and restarting the container.

## API summary

All endpoints except `/health/live` require `Authorization: Bearer <token>` and
an `X-AIVQC-Client-Id` header matching the identity bound to that token.

| Method | Endpoint | Required role |
|---|---|---|
| `GET` | `/health/live` | none; liveness only |
| `GET` | `/api/v1/info` | any authenticated client |
| `GET` | `/api/v1/stations` | Trainer or Administrator |
| `POST` | `/api/v1/stations` | Trainer or Administrator |
| `POST` | `/api/v1/packages` | Trainer or Administrator |
| `GET` | `/api/v1/stations/{station}/packages/latest` | assigned Production, Trainer, or Administrator |
| `GET` | `/api/v1/stations/{station}/packages/{id}/content` | assigned Production, Trainer, or Administrator |
| `POST` | `/api/v1/stations/{station}/packages/{id}/acknowledgements` | assigned Production, Trainer, or Administrator |
| `POST` | `/api/v1/packages/{id}/revoke` | Administrator |
