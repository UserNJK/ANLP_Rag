$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Python venv not found. Create it with: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

& $python "src\build_index.py"
& $python "src\rag_compare.py"
& $python "src\evaluate.py"
