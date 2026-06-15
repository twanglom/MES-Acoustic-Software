$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

pyinstaller --clean --noconfirm MES-Acoustic.spec

$iscc = "C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if (-not (Test-Path $iscc)) {
    throw "Inno Setup 6 was not found at: $iscc"
}

& $iscc "installer\MES-Acoustic.iss"

Write-Host ""
Write-Host "Release build complete:"
Write-Host "  App:       dist\MES-Acoustic\MES-Acoustic.exe"
Write-Host "  Installer: installer-output\"
