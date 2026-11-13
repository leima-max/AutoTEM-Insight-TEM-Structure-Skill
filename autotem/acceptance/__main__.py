"""CLI for installed-package disk-to-report checks."""

import argparse
import json
from pathlib import Path

from .driver import run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", required=True, type=Path, help="New or empty evidence directory"
    )
    parser.add_argument(
        "--require-installed",
        action="store_true",
        help="Require a non-editable package in an isolated private interpreter",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Also exercise GPA, atom strain, SAED and NBED disk cases",
    )
    parser.add_argument(
        "--release-bundle",
        type=Path,
        help="Verify a release bundle and bind it to the installed runtime marker",
    )
    args = parser.parse_args(argv)
    try:
        evidence = run(
            args.output,
            require_installed=args.require_installed,
            extended=args.extended,
            release_bundle=args.release_bundle,
        )
    except (OSError, ValueError) as exc:
        print(json.dumps({"case_verdict": "failed", "error": str(exc)}))
        return 2
    print(
        json.dumps(
            {
                "case_verdict": evidence["case_verdict"],
                "evidence": "evidence.json",
                "cases": {c["case_id"]: c["case_verdict"] for c in evidence["cases"]},
                "error": evidence.get("error"),
            }
        )
    )
    return 0 if evidence["case_verdict"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
