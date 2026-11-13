"""CPython startup observer, used only in a disposable qualification environment."""

import json
import os
import re
import sys
import time
from pathlib import Path


def attach(audit_root, monitored_roots, sentinel):
    if getattr(sys, "_autotem_qualification_observer", False):
        return
    sys._autotem_qualification_observer = True
    argv = sys.orig_argv[1:]
    module = argv[argv.index("-m") + 1] if "-m" in argv else ""
    role = "other"
    run_id = None
    if module == "autotem.harness.worker":
        task = argv[argv.index("-m") + 2]
        if task == "execute" and re.fullmatch("[a-f0-9]{32}", argv[-1]):
            role, run_id = "worker", argv[-1]
        elif task == "schedule":
            role = "scheduler"
    elif module == "autotem.harness.cli":
        role = "bridge"
    elif module == "autotem.simulation.diffsims_worker" and "--input" in argv:
        role = "diffsims_worker"
        request_path = Path(argv[argv.index("--input") + 1])
        run_id = next(
            (
                p.name
                for p in request_path.parents
                if p.parent.name == "runs" and re.fullmatch("[a-f0-9]{32}", p.name)
            ),
            None,
        )
    elif module == "autotem.acceptance":
        role = "disk_driver"
    elif any(Path(arg).name == "verify_variants.py" for arg in argv if not arg.startswith("-")):
        role = "variant_driver"
    elif "-c" in argv:
        role = "positive_control" if argv[-1] == sentinel else "python_command"

    pid = os.getpid()
    log = Path(audit_root) / f"{pid}-{time.time_ns()}.jsonl"
    with log.open("x", encoding="ascii") as stream:
        print(
            json.dumps(dict(role=role, pid=pid, run_id=run_id, isolated=sys.flags.isolated)),
            file=stream,
        )

    roots = set(monitored_roots) | {sentinel}
    seen = set()

    def record(name, event):
        root = name.partition(".")[0]
        if root in roots and (root, event) not in seen:
            seen.add((root, event))
            with log.open("a", encoding="ascii") as stream:
                print(json.dumps(dict(module=root, event=event)), file=stream)

    def audit(event, args):
        if event == "import":
            record(args[0], "import")

    class Finder:
        def find_spec(self, fullname, path=None, target=None):
            record(fullname, "find_spec")
            return None

    for name in tuple(sys.modules):
        record(name, "already_loaded")
    sys.addaudithook(audit)
    sys.meta_path.insert(0, Finder())


# The disposable interpreter supplies these values through runpy.run_path.
attach(globals()["audit_root"], globals()["monitored_roots"], globals()["sentinel"])
