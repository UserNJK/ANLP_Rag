param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Question,
    
    [Parameter(Mandatory=$false)]
    [int]$TopK = 3,
    
    [switch]$NoContext,
    [switch]$Interactive
)

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Python venv not found. Run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

$args = @("src\query.py")

if ($Interactive) {
    $args += "--interactive"
} else {
    $args += $Question
}

if ($TopK -ne 3) {
    $args += "--top-k", $TopK
}

if ($NoContext) {
    $args += "--no-context"
}

& $python $args
