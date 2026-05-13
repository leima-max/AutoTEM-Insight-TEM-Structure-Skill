$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
  throw "Project virtual environment not found at .venv. Create it first: python -m venv .venv"
}

& $pythonExe -m pytest @args
