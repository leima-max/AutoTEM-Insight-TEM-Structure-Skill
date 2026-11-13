# Build the headless Windows release

The Windows x64 target uses CPython 3.12.10 and its venv seed pip 25.0.1. Each directory contains one profile. The default `traditional-core-mcp` uses the `mcp` extra. `traditional-cli` resolves the base package with CLI/JSONL transport. The private `traditional-simulation` candidate resolves `[mcp,simulation]`, with `diffsims` supplying `orix` and `diffpy.structure` through its dependencies. [Core policy](../../configs/release/traditional-core-mcp.json), [CLI policy](../../configs/release/traditional-cli.json), [simulation policy](../../configs/release/traditional-simulation.json)

## Prepare the inputs

Use the target interpreter for pip and the builder. The upstream lock includes the runtime closure and the isolated build backend. Obtain these inputs on the build machine. The subsequent build and user install read local files.

```powershell
$ProfileId = 'traditional-core-mcp'
$policy = Get-Content -Raw "configs/release/$ProfileId.json" | ConvertFrom-Json
& $TargetPython -I -m pip download --require-hashes --only-binary=:all: --dest $WheelPool -r $policy.build_inputs_lock
& $TargetPython -I tools/setup/build_setup.py --profile $ProfileId --output $SetupDir
& $TargetPython -I tools/release/build_release.py --profile $ProfileId --output $NewBundle --wheelhouse $WheelPool --release-id $ReleaseId --python-nupkg $PythonPackage --launcher (Join-Path $SetupDir AutoTEM-Setup.exe) --installer-app (Join-Path $SetupDir AutoTEM-install.pyz)
```

`$TargetPython` is the exact interpreter executable. `$WheelPool` contains the upstream wheels listed in the lock. `$PythonPackage` is the original PSF NuGet archive. `$NewBundle` must be a new directory. `$ReleaseId` identifies this build. The installer app and launcher are built from the same source snapshot. Build them again after editing bootstrap or installation modules.

During wheel collection the builder resolves the complete runtime graph from the archived AutoTEM wheel with the selected extras. CLI and core use the same verified build input pool; only the resolved CLI closure ships in a CLI release. Simulation uses its own upstream pins and file origins. Use a fresh wheel pool when changing the build input lock. Missing dependencies, inconsistent hashes and absent compatible wheels fail the build. Installation uses each bundle's full lock with dependency checks enabled. Changing profiles selects a different private runtime and preserves the run store.

For an engineering bundle that runs under an existing exact interpreter, omit the three deployment arguments. To rebuild the same source bytes later, add `--source-archive <source/autotem-source.zip> --source-sha256 <recorded digest>`. Windows long paths can be avoided by passing `--scratch-parent` with a short, writable project directory. An interrupted build can be retried with a new output directory.

## Verify the result

The builder publishes the directory after verifying the lock, manifest, source archive, wheel contents and resource mirror. The source ZIP contains a per-file inventory, including uncommitted runtime sources and the Skill/Profile resources used in the wheel. Each wheel has an exact hash and metadata identity. The license inventory records declarations and hashes of notices inside the distributed wheels. The private Python license is also copied into the bundle.

The archived wheel source index ties the input pins to the corresponding files published on PyPI. To prepare a new version of this index, run `tools/release/freeze_wheel_sources.py --wheelhouse <input pool> --output <new index>` on the networked build machine. The release builder checks that index offline. [Frozen upstream files](../../configs/release/win-amd64-cp312-wheel-sources.json)

The release manifest records build-time acceptance as `not_evaluated`. Cold installation, end-to-end host use, signing and redistribution review have separate evidence. The current project lacks an AutoTEM license declaration. The release owner must resolve that item before approving public redistribution.

### Windows CI evidence

The Windows workflow builds all three editions on one disposable runner. The CLI project executes
J01 through its stable JSONL entry. The simulation project supplies J15, its five fault cases and
the subprocessor import audit. Once both projects have finished, the active core private Python
rechecks the original bundles and project paths for RG0, RG1 and RG2. The published summary keeps
the core and simulation manifest and lock hashes separate. Host observations, full source
regression and signed reviews require their own evidence. [Workflow](../../.github/workflows/windows-release.yml)

The `windows-release-summary` artifact contains per-edition identities, ZIP digests and gate
verdicts. The `windows-synthetic-qualification` artifact contains the combined index, acquisition
receipts and three portable archives of the CI-generated fixtures. On failure, the additional
`windows-incomplete-synthetic-runs` artifact retains the available diagnostic files. Both
artifact channels inherit repository access. Real host sessions, real samples, credentials,
installation directories and manual reviews stay with their owners.

After downloading a portable archive, use standard Python and the corresponding digest from
`public-summary.json`. The offline verifier reads the original SQLite files and any live WAL
from temporary copies, resolves the recorded Windows paths inside the ZIP, and checks input
snapshots, registered artifacts, HTML links and scientific receipts.

```powershell
$summary = Get-Content -Raw public-summary.json | ConvertFrom-Json
python -I tools/release/ci_evidence.py verify archives/simulation.zip --sha256 $summary.editions.'traditional-simulation'.archive_sha256
```

See the [release plan](../../docs/HARNESS_RELEASE_TECHNICAL_DEVELOPMENT_PLAN.md), [upstream wheel pins](../../requirements/release/win-amd64-cp312-build-inputs.txt), [pip hash-checking mode](https://pip.pypa.io/en/stable/topics/secure-installs/) and [Python's Windows distribution options](https://docs.python.org/3.12/using/windows.html).

### Frozen Windows source regression

Use the issued PSF CPython 3.12.10 and matching core and simulation bundles. Supply three new, short paths on the test runner. From the local Git worktree the runner fetches the exact support commit recorded in the source ZIP with a depth of one. This provides files such as the tested Streamlit app. The runner verifies the detached commit, overlays the archived source files before testing and checks their bytes again after pytest.

```powershell
& $TargetPython -I -B "$VerifierRoot/tools/release/run_source_tests.py" `
  --bundle "$ci/core" --simulation-bundle "$ci/sim" --support-repo $GitWorktree `
  --checkout "$ci/src" --scratch "$ci/st" --out "$ci/source"
```

The runner creates a private test venv, installs the simulation wheel dependencies and adds the hash-locked pytest, Streamlit, CIF and build dependencies. It installs the archived AutoTEM sources editable and checks their import location in a separate isolated process. By default pip fetches the additional pinned wheels from PyPI. Pass `--test-wheelhouse $LocalTestWheels` for an offline run. A historical source archive without the developer lock requires an explicit `--test-lock` and records its external origin. [Test-only wheel pins](../../requirements/release/win-amd64-cp312-source-tests.txt)

`qualify_release.py --source-tests "$ci/source/source-regression.json"` rechecks the JUnit, collected node IDs, both issued manifests and locks, the test wheel hashes and installed versions. Retain the `source-regression*` files together. The receipt distinguishes the full source suite, including its two real diffsims tests, from installed-product or host qualification.
