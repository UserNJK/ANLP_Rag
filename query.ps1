param(
    [Parameter(Mandatory=$false, Position=0)]
    [string]$Question = "",
    
    [Parameter(Mandatory=$false)]
    [int]$TopK = 10,
    
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
    if ([string]::IsNullOrWhiteSpace($Question)) {
        Write-Host "Question is required unless -Interactive is used." -ForegroundColor Yellow
        exit 1
    }
    $args += $Question
}

if ($TopK -ne 10) {
    $args += "--top-k", $TopK
}

if ($NoContext) {
    $args += "--no-context"
}

& $python $args
