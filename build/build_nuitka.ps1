param(
    [string]$PythonBin = "",
    [string]$ModelPath = "",
    [string]$NuitkaWork = ""
)

$ErrorActionPreference = "Stop"

function Invoke-PythonCheck {
    param(
        [string]$PythonExe,
        [string]$Code,
        [string]$FailureMessage
    )

    & $PythonExe -c $Code
    if ($LASTEXITCODE -ne 0) {
        throw $FailureMessage
    }
}

function Copy-DirectoryContents {
    param(
        [string]$SourceDir,
        [string]$DestinationDir
    )

    if (-not (Test-Path $SourceDir)) {
        return
    }

    Get-ChildItem -LiteralPath $SourceDir -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $DestinationDir -Recurse -Force
    }
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$InternalDir = Split-Path -Parent $ScriptDir
$RepoRoot = Split-Path -Parent $InternalDir
$ExternalDir = Join-Path $RepoRoot "External"

if (-not $PythonBin) {
    $VenvPython = Join-Path $InternalDir ".venv\Scripts\python.exe"
    if (Test-Path $VenvPython) {
        $PythonBin = $VenvPython
    } else {
        $PythonBin = "python"
    }
}

if (-not $NuitkaWork) {
    $NuitkaWork = Join-Path $env:TEMP "oad-nuitka-build"
}

$BinName = "oad-pipeline.exe"

Set-Location $InternalDir

Write-Host "==> Checking environment"
Invoke-PythonCheck -PythonExe $PythonBin -Code "import nuitka" `
    -FailureMessage "Nuitka is not importable. Install it with: $PythonBin -m pip install nuitka"
Invoke-PythonCheck -PythonExe $PythonBin `
    -Code "import ultralytics, fastapi, uvicorn, cv2, yaml" `
    -FailureMessage "One or more runtime dependencies are missing. Install the project requirements first."

if (-not $ModelPath) {
    $ModelCandidates = @(
        "rf_trained\yolo26n.pt",
        "rf_trained\yolo11n.pt",
        "rf_trained\yolo26s_trained.mlpackage"
    )
    $ModelPath = $ModelCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}

if (-not $ModelPath -or -not (Test-Path $ModelPath)) {
    throw "Model not found. Set -ModelPath to a valid .pt or .mlpackage under Internal/."
}

$ResolvedModelPath = (Resolve-Path $ModelPath).Path

Write-Host "==> Cleaning previous External/ at $ExternalDir"
if (Test-Path $ExternalDir) {
    Remove-Item -LiteralPath $ExternalDir -Recurse -Force
}

$Null = New-Item -ItemType Directory -Force -Path `
    (Join-Path $ExternalDir "bin"), `
    (Join-Path $ExternalDir "libraries"), `
    (Join-Path $ExternalDir "rf_trained"), `
    (Join-Path $ExternalDir "config"), `
    (Join-Path $ExternalDir "templates"), `
    (Join-Path $ExternalDir "input"), `
    (Join-Path $ExternalDir "output"), `
    (Join-Path $ExternalDir "logs"), `
    (Join-Path $ExternalDir "setup")

Write-Host "==> Compiling with Nuitka (this can take a few minutes on first run)"
$Null = New-Item -ItemType Directory -Force -Path $NuitkaWork

$CustomerConfig = Join-Path $NuitkaWork "customer_config"
if (Test-Path $CustomerConfig) {
    Remove-Item -LiteralPath $CustomerConfig -Recurse -Force
}
$Null = New-Item -ItemType Directory -Force -Path $CustomerConfig

Get-ChildItem -LiteralPath (Join-Path $InternalDir "config") -File | Where-Object {
    $_.Extension -in @(".yaml", ".yml", ".json") -and
    $_.Name -notmatch "(_v.*\.json|_bck\.json|_backup\.json)$"
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $CustomerConfig -Force
}

$CustomerModelConfig = Join-Path $CustomerConfig "model.yaml"
if (Test-Path $CustomerModelConfig) {
    $ModelConfigText = Get-Content -LiteralPath $CustomerModelConfig -Raw
    $UpdatedConfigText = [regex]::Replace(
        $ModelConfigText,
        "(?m)^source:.*$",
        'source: "input/sample.mp4"'
    )
    Set-Content -LiteralPath $CustomerModelConfig -Value $UpdatedConfigText
}

$NuitkaArgs = @(
    "-m", "nuitka",
    "--standalone",
    "--assume-yes-for-downloads",
    "--output-filename=$BinName",
    "--output-dir=$NuitkaWork",
    "--include-package=ultralytics",
    "--include-package=cv2",
    "--include-package=fastapi",
    "--include-package=uvicorn",
    "--include-package=starlette",
    "--include-package=jinja2",
    "--include-package-data=jinja2",
    "--include-package=yaml",
    "--include-package=lap",
    "--include-package=deep_sort_realtime",
    "--include-package=src",
    "--include-data-dir=$CustomerConfig=config",
    "--include-data-dir=$InternalDir\templates=templates",
    "--module-parameter=torch-disable-jit=yes",
    "--remove-output",
    "src/main.py"
)

& $PythonBin @NuitkaArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nuitka build failed."
}

$NuitkaDist = Join-Path $NuitkaWork "main.dist"
$BuiltBinary = Join-Path $NuitkaDist $BinName
if (-not (Test-Path $BuiltBinary)) {
    throw "Nuitka did not produce $BuiltBinary"
}

Write-Host "==> Assembling External/ deliverable"

Copy-Item -LiteralPath $BuiltBinary -Destination (Join-Path $ExternalDir "bin\$BinName") -Force

Write-Host "    - splitting bundled libraries into External/libraries/"
Get-ChildItem -LiteralPath $NuitkaDist -File -Filter *.dll | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $ExternalDir "libraries") -Force
}

Get-ChildItem -LiteralPath $NuitkaDist -Force | ForEach-Object {
    if ($_.Name -eq $BinName -or $_.Extension -eq ".dll") {
        return
    }
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $ExternalDir "bin") -Recurse -Force
}

if ((Get-Item -LiteralPath $ResolvedModelPath) -is [System.IO.DirectoryInfo]) {
    Copy-Item -LiteralPath $ResolvedModelPath -Destination (Join-Path $ExternalDir "rf_trained") -Recurse -Force
} else {
    Copy-Item -LiteralPath $ResolvedModelPath -Destination (Join-Path $ExternalDir "rf_trained") -Force
}

Copy-DirectoryContents -SourceDir $CustomerConfig -DestinationDir (Join-Path $ExternalDir "config")
Copy-DirectoryContents -SourceDir (Join-Path $InternalDir "templates") -DestinationDir (Join-Path $ExternalDir "templates")

$SampleCandidates = @(
    (Join-Path $RepoRoot "videos\clips\v5\seg_seg004.mp4"),
    (Join-Path $InternalDir "input\sample.mp4")
)
foreach ($Candidate in $SampleCandidates) {
    if (Test-Path $Candidate) {
        Copy-Item -LiteralPath $Candidate -Destination (Join-Path $ExternalDir "input\sample.mp4") -Force
        break
    }
}

Copy-Item -LiteralPath (Join-Path $ScriptDir "customer_readme.md") -Destination (Join-Path $ExternalDir "readme.md") -Force
Copy-Item -LiteralPath (Join-Path $ScriptDir "run.ps1") -Destination (Join-Path $ExternalDir "setup\run.ps1") -Force
Copy-Item -LiteralPath (Join-Path $ScriptDir "run.cmd") -Destination (Join-Path $ExternalDir "setup\run.cmd") -Force

Write-Host ""
Write-Host "==> Release ready at $ExternalDir"
Write-Host ""
Write-Host "Key outputs:"
Write-Host "  bin\$BinName"
Write-Host "  libraries\"
Write-Host "  rf_trained\"
Write-Host "  setup\run.ps1"
Write-Host "  setup\run.cmd"
Write-Host ""
Write-Host "To test locally:"
Write-Host "  cd $ExternalDir"
Write-Host "  .\setup\run.ps1"
Write-Host ""
Write-Host "Then open http://localhost:8000 in a browser."
