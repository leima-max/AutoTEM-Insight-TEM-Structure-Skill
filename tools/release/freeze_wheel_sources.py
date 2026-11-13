"""Freeze upstream URLs by matching local wheel bytes against published PyPI file digests."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from autotem.installation.release import SCHEMA, sha256, wheel_identity  # noqa: E402


def published_file(wheel: Path) -> tuple[str, dict]:
    name, version = wheel_identity(wheel)
    api = f"https://pypi.org/pypi/{quote(name, safe='')}/{quote(version, safe='')}/json"
    with urllib.request.urlopen(api, timeout=25) as response:
        release = json.load(response)
    matches = [f for f in release["urls"] if f["filename"] == wheel.name]
    if len(matches) != 1 or matches[0]["digests"]["sha256"] != sha256(wheel):
        raise ValueError(f"Local file has no matching published SHA256: {wheel.name}")
    record = matches[0]
    link = urlsplit(record["url"])
    if (link.scheme, link.netloc) != ("https", "files.pythonhosted.org"):
        raise ValueError(f"Unexpected PyPI file origin for {wheel.name}")
    return wheel.name, {"url": record["url"], "sha256": record["digests"]["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    paths = sorted(args.wheelhouse.glob("*.whl"))
    if not paths:
        parser.error("An upstream pool of wheels is required")
    with ThreadPoolExecutor(max_workers=5) as executor:
        wheels = dict(executor.map(published_file, paths))
    payload = json.dumps({"schema_version": SCHEMA, "wheels": wheels}, indent=2, sort_keys=True)
    # Publish only after every wheel matched its upstream record. Preserve any existing lock.
    with args.output.open("x", encoding="ascii", newline="\n") as file:
        file.write(payload + "\n")


if __name__ == "__main__":
    main()
