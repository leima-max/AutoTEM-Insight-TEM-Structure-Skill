"""Build the Windows launcher and its stdlib installer from this source tree."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autotem.installation.release import PROFILE, PROFILE_EXTRAS  # noqa: E402


def build(output: Path, profile: str = PROFILE):
    if profile not in PROFILE_EXTRAS:
        raise ValueError("Unknown release profile")
    target = json.loads((ROOT / f"configs/release/{profile}.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    app = output / "AutoTEM-install.pyz"
    sources = [
        ROOT / "autotem/__init__.py",
        ROOT / "autotem/bootstrap.py",
        ROOT / "autotem/harness/__init__.py",
        ROOT / "autotem/harness/hosts.py",
        *sorted((ROOT / "autotem/installation").glob("*.py")),
    ]
    with zipfile.ZipFile(app, "w") as archive:
        for name, payload in [
            (
                "__main__.py",
                b"from autotem.installation.managed import main\nraise SystemExit(main())\n",
            ),
            *((p.relative_to(ROOT).as_posix(), p.read_bytes()) for p in sources),
        ]:
            item = zipfile.ZipInfo(name)
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, payload)
    pin = output / "BundlePins.cs"
    supported = ", ".join(map(json.dumps, PROFILE_EXTRAS))
    pin.write_text(
        "internal static class BundlePins {\n"
        f'    internal const string App = "{hashlib.sha256(app.read_bytes()).hexdigest()}";\n'
        f'    internal const string Python = "{target["python_nupkg"]["sha256"]}";\n'
        f'    internal const string Version = "{target["python_version"]}";\n'
        f'    internal static readonly string[] Profiles = {{ {supported} }};\n'
        "}\n",
        encoding="ascii",
    )
    compiler = Path(os.environ["SystemRoot"]) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
    subprocess.run(
        [
            str(compiler),
            "/nologo",
            "/optimize+",
            "/target:winexe",
            "/platform:x64",
            "/reference:System.IO.Compression.dll",
            "/reference:System.Windows.Forms.dll",
            "/reference:System.Drawing.dll",
            "/reference:System.Web.Extensions.dll",
            f"/out:{output / 'AutoTEM-Setup.exe'}",
            str(ROOT / "tools/setup/Setup.cs"),
            str(ROOT / "tools/setup/SetupPreview.cs"),
            str(pin),
        ],
        check=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILE_EXTRAS, default=PROFILE)
    args = parser.parse_args()
    build(args.output.resolve(), args.profile)
