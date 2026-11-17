[CmdletBinding(DefaultParameterSetName = 'Candidates')]
param(
    [Parameter(Mandatory)] [string] $Root,
    [Parameter(Mandatory, ParameterSetName = 'Candidates')] [string] $ReleaseId,
    [Parameter(Mandatory, ParameterSetName = 'Previous')] [switch] $PreparePrevious,
    [string] $OfflineInputs
)
$ErrorActionPreference = 'Stop'
$Root = [IO.Path]::GetFullPath($Root)
if ($OfflineInputs) { $OfflineInputs = (Resolve-Path -LiteralPath $OfflineInputs).Path }
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))
$corePolicy = Get-Content 'configs/release/traditional-core-mcp.json' -Raw | ConvertFrom-Json
$seed = Join-Path $Root $corePolicy.python_nupkg.filename
$py = Join-Path $Root 'psf/tools/python.exe'
if ($PreparePrevious) {
    $carrier = (Join-Path (Get-Location) '.github/release-baselines/windows-core-rc8.zip')
    $helper = Join-Path $Root 'frozen/tools/release/ci_upgrade.py'
    $lock = Join-Path $Root 'previous-inputs.txt'
    $stage = 'pinned_historical_release_inputs'
    try {
        $inputJson = & $py -I -B $helper inputs --carrier $carrier --out $lock
        if ($LASTEXITCODE) { throw 'Historical source, manifest or fixture members unavailable or invalid' }
        $old = ($inputJson -join [Environment]::NewLine) | ConvertFrom-Json
        if ($old.python_nupkg.sha256 -ne $corePolicy.python_nupkg.sha256) {
            throw 'The historical evidence requires a different Windows CPython target'
        }
        $stage = 'pinned_historical_wheel_pool'
        if ($old.build_inputs_sha256 -eq (Get-FileHash -LiteralPath $corePolicy.build_inputs_lock).Hash) {
            $pool = Join-Path $Root 'core-pool'
        } elseif ($OfflineInputs) {
            $pool = Join-Path $OfflineInputs "pools/$($old.build_inputs_sha256)"
        } else {
            $pool = Join-Path $Root 'previous-pool'
            & $py -I -m pip download --disable-pip-version-check --no-cache-dir --only-binary=:all: --require-hashes --dest $pool -r $lock
            if ($LASTEXITCODE) { throw 'Could not obtain the exact historical wheel pool' }
        }
        $stage = 'restored_bytes_and_distinct_code_generations'
        $pair = & $py -I -B $helper restore --carrier $carrier --wheelhouse $pool --python-nupkg $seed --bundle "$Root/core" --out "$Root/previous"
        if ($LASTEXITCODE) { throw 'Could not reproduce the verified predecessor or prove different application code' }
        [IO.File]::WriteAllText("$Root/generation-pair.json", ($pair -join [Environment]::NewLine), [Text.UTF8Encoding]::new($false))
    } catch {
        $diagnostic = @{schema_version='1.0.0'; case_verdict='failed'; stage=$stage; reason=$_.Exception.Message}
        [IO.File]::WriteAllText("$Root/previous-failure.json", ($diagnostic | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
        throw
    }
    return
}
if (Test-Path -LiteralPath $Root) { throw "A new CI workspace is required: $Root" }
$null = New-Item -ItemType Directory -Path $Root
$null = New-Item -ItemType Directory -Path (Join-Path $Root 'archives')
if ($OfflineInputs) {
    Copy-Item -LiteralPath (Join-Path $OfflineInputs $corePolicy.python_nupkg.filename) -Destination $seed
} else {
    Invoke-WebRequest -Uri $corePolicy.python_nupkg.url -OutFile $seed
}
if ((Get-FileHash -LiteralPath $seed -Algorithm SHA256).Hash -ne $corePolicy.python_nupkg.sha256) {
    throw 'PSF package hash mismatch'
}
[System.IO.Compression.ZipFile]::ExtractToDirectory($seed, (Join-Path $Root 'psf'))
$setup = Join-Path $Root 'setup'
& $py -I tools/setup/build_setup.py --output $setup
if ($LASTEXITCODE) { exit $LASTEXITCODE }

# Pool reuse follows the exact build-input lock digest. Every edition still resolves its own closure.
$pools = @{}
$editions = [ordered]@{'traditional-core-mcp' = 'core'; 'traditional-cli' = 'cli'; 'traditional-simulation' = 'sim'}
foreach ($edition in $editions.Keys) {
    $short = $editions[$edition]
    $policy = Get-Content "configs/release/$edition.json" -Raw | ConvertFrom-Json
    if ($policy.python_nupkg.sha256 -ne $corePolicy.python_nupkg.sha256) { throw "Different PSF pin for $edition" }
    $hash = (Get-FileHash -LiteralPath $policy.build_inputs_lock -Algorithm SHA256).Hash.ToLowerInvariant()
    if (-not $pools.ContainsKey($hash)) {
        $pool = Join-Path $Root "$short-pool"
        $poolSource = @()
        if ($OfflineInputs) {
            $poolSource = @('--no-index', '--find-links', (Resolve-Path -LiteralPath (Join-Path $OfflineInputs "pools/$hash")).Path)
        }
        & $py -I -m pip download --disable-pip-version-check --no-cache-dir --only-binary=:all: --require-hashes @poolSource --dest $pool -r $policy.build_inputs_lock
        if ($LASTEXITCODE) { exit $LASTEXITCODE }
        $pools[$hash] = $pool
    }
    $bundle = Join-Path $Root $short
    $reuse = @()
    if ($edition -ne 'traditional-core-mcp') {
        $reuse = @('--source-archive', (Join-Path "$Root/core" $identity.source.archive),
                  '--source-sha256', $identity.source.archive_sha256)
    }
    $wheelPool = $pools[$hash]
    & $py -I tools/release/build_release.py --profile $edition --release-id "$ReleaseId-$short" --output $bundle --scratch-parent "$Root/bld" --wheelhouse $wheelPool --python-nupkg $seed --launcher "$setup/AutoTEM-Setup.exe" --installer-app "$setup/AutoTEM-install.pyz" @reuse
    if ($LASTEXITCODE) { exit $LASTEXITCODE }
    $manifest = Get-Content "$bundle/release-manifest.json" -Raw | ConvertFrom-Json
    $wheel = @($manifest.artifacts | Where-Object path -Like 'wheelhouse/*/autotem_insight-*.whl')
    if ($wheel.Count -ne 1 -or (Get-FileHash -LiteralPath (Join-Path $bundle $wheel[0].path)).Hash -ne $wheel[0].sha256) {
        throw "Wrong AutoTEM wheel for $edition"
    }
    if ($edition -eq 'traditional-core-mcp') {
        $identity = $manifest
        $applicationHash = $wheel[0].sha256
    } elseif ($wheel[0].sha256 -ne $applicationHash -or $manifest.source.archive_sha256 -ne $identity.source.archive_sha256) {
        throw "$edition does not share the core source and AutoTEM wheel"
    }
}
& $py -I -B tools/release/ci_upgrade.py stage --bundle "$Root/core" --out "$Root/frozen"
if ($LASTEXITCODE) { exit $LASTEXITCODE }
