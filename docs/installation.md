# Installation

## Windows Installer

1. Open the repository's **Releases** page.
2. Open the latest release.
3. Download `MES-Acoustic-<version>-Windows-x64-Setup.exe`.
4. Run the installer and follow the setup wizard.
5. Start MES-Acoustic from the Start menu or desktop shortcut.

The installer is approximately 150 MB. Installation requires approximately
600 MB of disk space.

Windows may show a SmartScreen warning because community builds are not
code-signed. Verify that the installer came from the official repository
release before running it.

## Run From Source

Requirements:

- 64-bit Windows
- Python 3.11
- Git

```powershell
git clone <repository-url>
cd MES-ACOUSTIC
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main_run.py
```

## Application Data

The installer does not include geometry or generated analysis data. Clone or
download the source repository to obtain the public `room.msh` and `a320.msh`
test meshes.
Keep the following files together when moving a saved project:

- The project `.json` file
- Its associated `<project-name>_files` directory
- The referenced mesh file

View-factor matrices can be large. Store them outside the Git repository or
inside an ignored local `geo` directory.

## Uninstall

Use **Settings > Apps > Installed apps > MES-Acoustic > Uninstall**, or run
the uninstaller from the MES-Acoustic installation directory.
