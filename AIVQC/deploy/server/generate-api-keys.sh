#!/bin/sh
set -eu
umask 077

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
output_path=${1:-"$script_dir/secrets/api-keys.json"}
station_id=${2:-line-1}
case "$station_id" in
  ''|*[!A-Za-z0-9._-]*)
    printf '%s\n' "Station ID may contain only letters, numbers, dots, underscores, and hyphens." >&2
    exit 1
    ;;
esac
mkdir -p "$(dirname -- "$output_path")"

trainer_token=$(openssl rand -hex 32)
production_token=$(openssl rand -hex 32)
administrator_token=$(openssl rand -hex 32)
trainer_hash=$(printf %s "$trainer_token" | openssl dgst -sha256 -r | awk '{print toupper($1)}')
production_hash=$(printf %s "$production_token" | openssl dgst -sha256 -r | awk '{print toupper($1)}')
administrator_hash=$(printf %s "$administrator_token" | openssl dgst -sha256 -r | awk '{print toupper($1)}')

sed \
  -e "s/TRAINER_HASH/$trainer_hash/" \
  -e "s/PRODUCTION_HASH/$production_hash/" \
  -e "s/ADMINISTRATOR_HASH/$administrator_hash/" \
  -e "s/STATION_ID/$station_id/g" > "$output_path" <<'EOF'
{
  "clients": [
    { "id": "trainer-main", "keySha256": "TRAINER_HASH", "roles": ["trainer"], "stationId": null },
    { "id": "production-STATION_ID", "keySha256": "PRODUCTION_HASH", "roles": ["production"], "stationId": "STATION_ID" },
    { "id": "server-admin", "keySha256": "ADMINISTRATOR_HASH", "roles": ["administrator"], "stationId": null }
  ]
}
EOF

printf '%s\n' "API key hashes saved to $output_path"
printf '%s\n' "Store these raw tokens in a password manager; they are shown only now."
printf '%s\n' "trainer-main: $trainer_token"
printf '%s\n' "production-$station_id: $production_token"
printf '%s\n' "server-admin: $administrator_token"
