[CmdletBinding()]
param(
    [Parameter(Mandatory, Position=0)][ValidateSet('init','verify')][string]$Action,
    [Parameter(Mandatory)][string]$Project,
    [Parameter(Mandatory)][string]$Release,
    [Parameter(Mandatory)][string]$Record,
    [string]$HostName,
    [string]$Case,
    [string]$Variant,
    [string]$Summary
)
$ErrorActionPreference = 'Stop'
try {
    $projectPath = (Resolve-Path -LiteralPath $Project).ProviderPath
    $active = Get-Content -LiteralPath (Join-Path $projectPath '.autotem/active-runtime.json') -Raw -Encoding UTF8 | ConvertFrom-Json
    if ($active.manifest_sha256 -cnotmatch '^[0-9a-f]{64}$' -or $active.runtime_id -cne ('rt-' + $active.manifest_sha256.Substring(0,20))) {
        throw 'Invalid installed runtime identity.'
    }
    $python = Join-Path $projectPath ('.autotem/runtimes/' + $active.runtime_id + '/Scripts/python.exe')
    $params = @($Action, '--project', $projectPath, '--release', (Resolve-Path -LiteralPath $Release).ProviderPath, '--record', $Record)
    foreach ($pair in @(@('--host', $HostName), @('--case', $Case), @('--variant', $Variant), @('--summary', $Summary))) {
        if ($pair[1]) { $params += $pair }
    }
    & $python -I -B -X utf8 (Join-Path $PSScriptRoot 'collect.py') @params
    exit $LASTEXITCODE
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 2
}
