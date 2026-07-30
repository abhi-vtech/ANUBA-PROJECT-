$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExternalDir = Split-Path -Parent $ScriptDir
$BinPath = Join-Path $ExternalDir "bin\oad-pipeline.exe"
$LibrariesDir = Join-Path $ExternalDir "libraries"

if (-not (Test-Path $BinPath)) {
    Write-Error "Binary $BinPath not found. Rebuild the bundle before running it."
    exit 1
}

if (Test-Path $LibrariesDir) {
    if ($env:PATH) {
        $env:PATH = "$LibrariesDir;$env:PATH"
    } else {
        $env:PATH = $LibrariesDir
    }
}

$Null = New-Item -ItemType Directory -Force -Path `
    (Join-Path $ExternalDir "output"), `
    (Join-Path $ExternalDir "logs")

if (-not $env:PYTHONUNBUFFERED) {
    $env:PYTHONUNBUFFERED = "1"
}
if (-not $env:LOG_LEVEL) {
    $env:LOG_LEVEL = "WARNING"
}
if (-not $env:LOG_METRICS_INTERVAL) {
    $env:LOG_METRICS_INTERVAL = "5"
}

Write-Host "Starting oad-pipeline..."
Write-Host "  EXTERNAL_DIR      = $ExternalDir"
Write-Host "  VIDEO_SOURCE      = $(if ($env:VIDEO_SOURCE) { $env:VIDEO_SOURCE } else { '<from config/model.yaml>' })"
Write-Host "  MODEL_PATH        = $(if ($env:MODEL_PATH) { $env:MODEL_PATH } else { '<from config/model.yaml>' })"
Write-Host "  LOG_LEVEL         = $env:LOG_LEVEL"
Write-Host "  Dashboard         = http://localhost:8000"
Write-Host ""
Write-Host "  Tip: to override config, set env vars before .\setup\run.ps1"
Write-Host '       $env:VIDEO_SOURCE = "input\my_video.mp4"'
Write-Host ""

Push-Location $ExternalDir
try {
    & $BinPath @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
