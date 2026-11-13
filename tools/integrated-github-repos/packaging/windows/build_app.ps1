$ErrorActionPreference = "Stop"
$ProjectDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $ProjectDir

$BuildVenv = if ($env:BUILD_VENV) { $env:BUILD_VENV } else { ".packaging-venv" }
if (-not (Test-Path (Join-Path $BuildVenv "Scripts\python.exe"))) {
  py -m venv $BuildVenv
}
$BuildPython = Join-Path $BuildVenv "Scripts\python.exe"
& $BuildPython -m pip install ".[dev]"
& $BuildPython packaging\generate_icons.py
& $BuildPython -m PyInstaller --noconfirm --clean --windowed `
  --name "Integrated GitHub Repos" `
  --icon assets\app-icon.ico `
  --add-data "assets\app-icon.png:assets" `
  --paths src `
  src\integrated_github_repos\app.py

Write-Host "Built: $ProjectDir\dist\Integrated GitHub Repos\Integrated GitHub Repos.exe"
