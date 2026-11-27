$ErrorActionPreference = 'Stop'
Push-Location (Join-Path $PSScriptRoot '..')
try {
    node --test tests/api.test.js
    if ($LASTEXITCODE -ne 0) { throw 'API checks failed.' }
} finally {
    Pop-Location
}
