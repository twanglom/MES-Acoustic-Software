# Publishing A Release

## Why The Installer Is Not Committed

Regular GitHub repositories block files larger than 100 MiB. The Windows
installer is approximately 150 MB, so it must not be committed to the Git
history.

GitHub Releases accepts individual release assets under 2 GiB, making it the
correct place to distribute the installer.

`installer-output/`, `dist/`, and other generated build directories are
excluded by `.gitignore`.

## Build

Install Python 3.11, project dependencies, PyInstaller, and Inno Setup 6.

```powershell
.\build-release.ps1
```

Expected output:

```text
installer-output\MES-Acoustic-0.1.0-Windows-x64-Setup.exe
```

## Verify

Before publishing:

1. Install the application using the generated setup file.
2. Start the application from the installed shortcut.
3. Import a small `.msh` test case.
4. Confirm Physical Surface selection.
5. Compute or load view factors.
6. Run a small solver case.
7. Uninstall the application.
8. Optionally publish a SHA-256 checksum with the installer.

Generate a checksum with:

```powershell
Get-FileHash installer-output\MES-Acoustic-0.1.0-Windows-x64-Setup.exe -Algorithm SHA256
```

## Publish Through The GitHub Website

1. Push source code and documentation to GitHub.
2. Open the repository's **Releases** page.
3. Select **Draft a new release**.
4. Create a tag such as `v0.1.0`.
5. Enter a release title and release notes.
6. Attach the generated setup `.exe`.
7. Attach a checksum text file if available.
8. Publish the release.

Do not attach local geometry, project data, or generated view-factor matrices
unless they are intentionally prepared as separate example assets.

## Suggested Release Notes

```text
MES-Acoustic v0.1.0

Highlights
- Gmsh Physical Surface material assignment
- Automatic surface-normal correction
- GUI view-factor and solver progress
- Point and standard-plane receivers
- Receiver selection before solver execution
- PyVista SPL contour visualization

Windows
- Download MES-Acoustic-0.1.0-Windows-x64-Setup.exe
- Approximately 150 MB download and 600 MB installed
```
