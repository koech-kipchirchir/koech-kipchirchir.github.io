# ============================================================
# AIOS Production Deployment Script (Windows / PowerShell)
# ============================================================

param(
    [switch]$gpu,
    [switch]$monitor,
    [switch]$build,
    [switch]$down,
    [switch]$logs,
    [switch]$restart,
    [string]$envFile = ""
)

$ProjectDir = Split-Path -Path (Split-Path -Path $PSScriptRoot -Parent) -Parent
$DeployDir = Join-Path $ProjectDir "deploy"
$ComposeFiles = @("-f", (Join-Path $DeployDir "docker-compose.yml"))
$EnvArg = @()

if (-not $envFile) {
    $envFile = Join-Path $DeployDir ".env"
}

# Copy .env.example if .env doesn't exist
if (-not (Test-Path $envFile)) {
    Write-Warning ".env file not found — copying from .env.example"
    Copy-Item (Join-Path $DeployDir ".env.example") $envFile
}
$EnvArg = @("--env-file", $envFile)

# GPU support
if ($gpu) {
    $ComposeFiles += @("-f", (Join-Path $DeployDir "docker-compose.gpu.yml"))
    Write-Host "[INFO] GPU acceleration enabled" -ForegroundColor Green
}

# Monitoring
if ($monitor) {
    $ComposeFiles += @("-f", (Join-Path $DeployDir "docker-compose.monitoring.yml"))
    Write-Host "[INFO] Monitoring stack enabled" -ForegroundColor Green
}

Set-Location $ProjectDir

if ($down) {
    Write-Host "[INFO] Stopping AIOS production stack..." -ForegroundColor Green
    & docker compose @ComposeFiles @EnvArg down
    Write-Host "[INFO] All services stopped." -ForegroundColor Green
}
elseif ($logs) {
    & docker compose @ComposeFiles @EnvArg logs -f
}
elseif ($restart) {
    Write-Host "[INFO] Restarting AIOS production stack..." -ForegroundColor Green
    & docker compose @ComposeFiles @EnvArg restart
    Write-Host "[INFO] All services restarted." -ForegroundColor Green
}
else {
    $buildArg = @()
    if ($build) { $buildArg = @("--build") }

    Write-Host "[INFO] Starting AIOS production stack..." -ForegroundColor Green
    & docker compose @ComposeFiles @EnvArg up -d @buildArg

    Write-Host "[INFO] AIOS is running:" -ForegroundColor Green
    Write-Host "  API:       http://localhost:8000" -ForegroundColor Green
    Write-Host "  Docs:      http://localhost:8000/docs" -ForegroundColor Green
    if ($monitor) {
        Write-Host "  Grafana:   http://localhost:3000 (admin:aiosadmin)" -ForegroundColor Green
        Write-Host "  Prometheus: http://localhost:9090" -ForegroundColor Green
    }
}
