$ErrorActionPreference = "Stop"
$env:PYTHONDONTWRITEBYTECODE = "1"

python -m pytest tests -q
if ($LASTEXITCODE -ne 0) {
    throw "Public unit/synthetic tests failed."
}

python scripts/validate_public_release.py
if ($LASTEXITCODE -ne 0) {
    throw "Public manuscript-output validation failed."
}

Write-Output "PUBLIC_RELEASE_VALIDATION_PASS"
