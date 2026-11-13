# Windows upgrade baseline

The Windows release workflow exercises the original `20261112-traditional-core-mcp-rc8`
release. The initial release-automation PR predates a usable release builder in Git history.
The fixture therefore preserves the release's exact manifest and project-produced bytes.
Ordinary PRs and manual dispatches use the same pins. The intended scope is synthetic
Windows CI. Publisher signature and redistribution review retain their original
`not_evaluated` status.

`windows-core-rc8.zip` is 1,330,716 bytes. Its SHA-256 is
`b52d4826b566cbe1c3db97b80c2a6f81d761245841530f76850ff4812e71f923`.
It contains the following eight members.

| Member | SHA-256 |
| --- | --- |
| release-manifest.json | `28b37d7ec5da3347660590df1f60e6eb035005387837ca8eef0640016e49a8fe` |
| source/autotem-source.zip | `bc491c1611812a7f53db3390043655c9f2aaa64abcd1cab47603f2afc41dd80e` |
| AutoTEM-Setup.exe | `c5db90ffafd63a95613f8b20a09a8c3c8225e221d412dd4bfaee4f87844141de` |
| AutoTEM-install.pyz | `b3fc1a9417e7027d0a2d88897c5d0c5109c05ff259b66ea6d1907af01ed23d01` |
| wheelhouse/traditional-core-mcp/autotem_insight-0.1.0-py3-none-any.whl | `5ea7fd4530fdfb498f510b8e0ae927a7de3e8649f61af959bc642c313d9f35c9` |
| locks/traditional-core-mcp.txt | `1debd09617075a9c2fe5987b7a9bc668e19cf24c9977ae49faf4eadb8159c2e2` |
| wheelhouse/traditional-core-mcp/wheelhouse-manifest.json | `182a405b44b807a92dbbb7d9d5849d67e1153c8087e2e1f4227fd4af5f483dbb` |
| licenses/inventory.json | `d24b7e574ca74191e21a3bb30b3c906aa939822a52422ab5e4164a185aa8ab9f` |

The source archive inventories the original 274 files, including the C# launcher and
Python installer/application sources. The workflow restores the 51 upstream wheels
from the historical hash-locked build pool. PSF CPython 3.12.10 for Windows x64 uses
NuGet SHA-256 `0eb85c2dfccccf1b17352de4c397f69194035b7d37149eacc16f1147d93de3b8`.
Its archived license supplies the original CPython license file. The verifier checks every
restored file, the installation lock and first-party wheel against the original manifest
and source. Both a changed application wheel and changed Python module bytes are required
before starting the native upgrade. A missing baseline, incompatible target or digest
mismatch fails the predecessor step and records its reason.

The retained local RC8 → RC12 → RC8 run used this exact previous manifest. Its
`outputs/harness-qualification-20261112/rc12/upgrade.json` SHA-256 is
`2e96449e26239d9b35f60fcbbe45121f5bc95a65b4860cb4eebe1a69ad6d7d04`.
The three separate synthetic J01 jobs registered 75 artifacts. This record identifies
the source of the fixture. Each CI run performs its own upgrade and rollback.

## Local replay

Run the following from the repository checkout with a new short working directory.
The first command obtains the pinned inputs and builds the current packages.

```powershell
$ci = 'D:/atci'
./tools/release/build_windows_ci.ps1 -Root $ci -ReleaseId local-qualified
./tools/release/build_windows_ci.ps1 -Root $ci -PreparePrevious
& "$ci/psf/tools/python.exe" -I -B "$ci/frozen/tools/setup/verify_upgrade.py" --previous-bundle "$ci/previous" --bundle "$ci/core" --project "$ci/up" --output "$ci/upgrade.json"
```

For existing local bundles use the pinned target CPython and select the previous manifest
explicitly. The `pair` command compares both releases without installing them.

```powershell
& $py -I -B tools/release/ci_upgrade.py pair --previous-bundle $previous --bundle $current --previous-manifest-sha256 $trustedPreviousManifest
```

To reproduce the fixture from the complete original release, run `capture` with
`--bundle`, `--previous-manifest-sha256` and a new `--out` path. Replacing the
predecessor requires reviewing the independently obtained manifest, the sparse
archive pins and the changes in the shipped application.
