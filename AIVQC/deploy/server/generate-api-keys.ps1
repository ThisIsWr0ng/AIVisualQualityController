[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "secrets\api-keys.json"),
    [string]$StationId = "line-1"
)

$ErrorActionPreference = "Stop"

if ($StationId -notmatch '^[A-Za-z0-9._-]{1,128}$') {
    throw "StationId may contain only letters, numbers, dots, underscores, and hyphens."
}

function New-AivqcToken {
    $bytes = [byte[]]::new(32)
    [System.Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
    return [Convert]::ToHexString($bytes)
}

function Get-Sha256Hex([string]$Value) {
    $bytes = [Text.Encoding]::UTF8.GetBytes($Value)
    return [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($bytes))
}

$trainerToken = New-AivqcToken
$productionToken = New-AivqcToken
$administratorToken = New-AivqcToken
$configuration = @{
    clients = @(
        @{ id = "trainer-main"; keySha256 = Get-Sha256Hex $trainerToken; roles = @("trainer"); stationId = $null }
        @{ id = "production-$StationId"; keySha256 = Get-Sha256Hex $productionToken; roles = @("production"); stationId = $StationId }
        @{ id = "server-admin"; keySha256 = Get-Sha256Hex $administratorToken; roles = @("administrator"); stationId = $null }
    )
}

$directory = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $directory | Out-Null
$configuration | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $OutputPath -Encoding utf8NoBOM

Write-Host "API key hashes saved to $OutputPath"
Write-Host "Store these raw tokens in a password manager; they are shown only now."
Write-Host "trainer-main: $trainerToken"
Write-Host "production-$StationId: $productionToken"
Write-Host "server-admin: $administratorToken"
