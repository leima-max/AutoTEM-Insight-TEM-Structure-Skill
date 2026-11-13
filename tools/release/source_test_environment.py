"""Provision and independently check the Windows source-test venv and its dependency evidence."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import site
import subprocess
import sys
import zipfile
from pathlib import Path

TEST_LOCK = "requirements/release/win-amd64-cp312-source-tests.txt"
PRODUCT = "autotem-insight"
PROFILES = {"core": "traditional-core-mcp", "simulation": "traditional-simulation"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normal(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def pins(path):
    result = {}
    for line in path.read_text("utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        match = re.fullmatch(
            r"([\w.-]+)(?:\[[\w,-]+\])?==([\w.!+]+) --hash=sha256:([a-f0-9]{64})", line
        )
        require(match, f"Invalid hash-locked source-test requirement in {path.name}")
        name, version, digest = match.groups()
        name = normal(name)
        require(name not in result, f"Duplicate source-test pin for {name}")
        result[name] = (version, digest)
    require(result, "Empty source-test dependency lock")
    return result


def process_env(scratch):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.upper().startswith(("PIP_", "PYTEST_", "PYTHON", "CONDA_", "AUTOTEM_", "NUMBA_"))
        and k.upper() not in {"VIRTUAL_ENV", "__PYVENV_LAUNCHER__"}
    }
    temp = scratch / "tmp"
    temp.mkdir()
    env.update(
        PIP_CONFIG_FILE=os.devnull,
        PIP_NO_INDEX="1",
        PYTHONNOUSERSITE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        AUTOTEM_RUN_REAL_DIFFSIMS_TESTS="1",
        AUTOTEM_TMP_DIR=str(scratch / "autotem"),
        NUMBA_CACHE_DIR=str(scratch / "numba"),
        MPLCONFIGDIR=str(scratch / "mpl"),
        PYTHONIOENCODING="utf-8",
        TMP=str(temp),
        TEMP=str(temp),
        TMPDIR=str(temp),
    )
    return env


def probe(checkout):
    prefix = Path(sys.prefix).resolve()
    module = Path(importlib.util.find_spec("autotem").origin).resolve()
    require(
        prefix != Path(sys.base_prefix).resolve()
        and sys.flags.isolated
        and not site.ENABLE_USER_SITE,
        "Test interpreter must use an isolated venv without user site-packages",
    )
    require(
        module == checkout / "autotem/__init__.py",
        "Product import did not resolve to the frozen source checkout",
    )
    installed = {}
    for dist in importlib.metadata.distributions():
        name = normal(dist.metadata["Name"])
        require(
            Path(dist.locate_file("")).resolve().is_relative_to(prefix) and name not in installed,
            "Duplicate or external Python distribution in the source-test environment",
        )
        installed[name] = dist.version
    direct = json.loads(importlib.metadata.distribution(PRODUCT).read_text("direct_url.json"))
    require(
        direct["url"].rstrip("/") == checkout.as_uri() and direct["dir_info"]["editable"] is True,
        "AutoTEM must be installed editable from the tested source checkout",
    )
    return {
        "python_version": platform.python_version(),
        "isolated": True,
        "user_site": False,
        "base_executable_sha256": sha(Path(sys._base_executable)),
        "editable_source": "autotem/__init__.py",
        "all_distributions_in_test_venv": True,
        "installed": installed,
    }


def prepare(verifier, bundles, checkout, out, scratch, env, test_pool, explicit_lock):
    require(sys.flags.isolated, "Start the source runner with the exact release Python and -I")
    base = out / "source-regression.json"
    releases = {
        name: verifier.verify_release(path, profile=PROFILES[name])
        for name, path in bundles.items()
    }
    for name, release in releases.items():
        shutil.copyfile(
            bundles[name] / "release-manifest.json", base.with_suffix(f".{name}-manifest.json")
        )
        shutil.copyfile(release.lock, base.with_suffix(f".{name}.lock.txt"))
    sim = releases["simulation"]
    source_lock = checkout / TEST_LOCK
    if source_lock.is_file():
        require(
            explicit_lock is None or sha(explicit_lock) == sha(source_lock),
            "Override differs from archived test lock",
        )
        origin = "archived_source"
    else:
        require(
            explicit_lock and explicit_lock.is_file(),
            "Older source archive requires an explicit --test-lock",
        )
        source_lock, origin = explicit_lock, "explicit_legacy_lock"
    extra = base.with_suffix(".test.lock.txt")
    shutil.copyfile(source_lock, extra)
    test_pins = pins(extra)
    runtime = pins(sim.lock)
    require(
        not runtime.keys() & test_pins.keys(),
        "Test supplements replace an issued runtime requirement",
    )
    bound = release_records(base, releases["core"].manifest_sha256, sim.manifest_sha256)
    runtime_archive = bundles["core"] / bound["core"]["runtime"]["archive"]
    with zipfile.ZipFile(runtime_archive) as z:
        base_digest = hashlib.sha256(z.read("tools/python.exe")).hexdigest()
    require(
        sha(Path(sys._base_executable)) == base_digest,
        "Source runner differs from the issued PSF base executable",
    )
    runtime_lock = scratch / "simulation-dependencies.txt"
    runtime_lock.write_text(
        "".join(f"{n}=={v} --hash=sha256:{h}\n" for n, (v, h) in runtime.items() if n != PRODUCT),
        encoding="ascii",
    )
    venv = scratch / "venv"
    python = venv / "Scripts/python.exe"
    wheels = scratch / "test-wheels"
    wheels.mkdir()
    with (out / "environment.log").open("xb") as log:

        def run(cmd):
            subprocess.run(
                cmd, cwd=scratch, env=env, check=True, stdout=log, stderr=subprocess.STDOUT
            )

        run([sys.executable, "-I", "-m", "venv", str(venv)])
        pip = [
            str(python),
            "-I",
            "-m",
            "pip",
            "--isolated",
            "--disable-pip-version-check",
            "--no-cache-dir",
        ]
        origin_args = (
            ["--no-index", "--find-links", str(test_pool)]
            if test_pool
            else ["--index-url", "https://pypi.org/simple"]
        )
        run(
            [
                *pip,
                "download",
                "--no-deps",
                "--only-binary=:all:",
                "--require-hashes",
                *origin_args,
                "-r",
                str(extra),
                "-d",
                str(wheels),
            ]
        )
        downloaded = {}
        for path in wheels.iterdir():
            name, version = verifier.wheel_identity(path)
            require(
                name not in downloaded and (version, sha(path)) == test_pins.get(name),
                "Test wheel differs from its pin",
            )
            downloaded[name] = {"filename": path.name, "sha256": test_pins[name][1]}
        require(downloaded.keys() == test_pins.keys(), "Incomplete test wheel set")
        run(
            [
                *pip,
                "install",
                "--no-index",
                "--only-binary=:all:",
                "--require-hashes",
                "--find-links",
                str(wheels),
                "--find-links",
                str(sim.wheelhouse),
                "-r",
                str(runtime_lock),
                "-r",
                str(extra),
            ]
        )
        run(
            [
                *pip,
                "install",
                "--no-index",
                "--no-build-isolation",
                "--no-deps",
                "-e",
                str(checkout),
            ]
        )
        run([*pip, "check"])
    env["PATH"] = str(python.parent) + os.pathsep + env.get("PATH", "")
    observed = json.loads(
        subprocess.check_output(
            [str(python), "-I", "-B", str(Path(__file__).resolve()), str(checkout)],
            cwd=scratch,
            env=env,
        )
    )
    result = {
        "schema_version": "1.0.0",
        "execution_layer": "isolated_source_test_venv",
        "probe": observed,
        "simulation_manifest_sha256": sim.manifest_sha256,
        "core_manifest_sha256": releases["core"].manifest_sha256,
        "test_lock_sha256": sha(extra),
        "test_lock_origin": origin,
        "test_wheels": downloaded,
        "psf_executable_sha256": base_digest,
        "pip_check": "passed",
    }
    base.with_suffix(".environment.json").write_text(
        json.dumps(result, sort_keys=True, indent=2), encoding="utf-8"
    )
    return python


def release_records(report, core_identity, simulation_identity):
    manifests = {}
    for name, identity in (("core", core_identity), ("simulation", simulation_identity)):
        raw = report.with_suffix(f".{name}-manifest.json")
        lock = report.with_suffix(f".{name}.lock.txt")
        require(sha(raw) == identity, f"Wrong source-test {name} release manifest")
        manifest = json.loads(raw.read_bytes())
        profile = PROFILES[name]
        require(set(manifest["profiles"]) == {profile}, "Wrong source-test profile")
        selected = manifest["profiles"][profile]
        records = {i["path"]: i["sha256"] for i in manifest["artifacts"]}
        require(
            records[selected["lock"]] == selected["lock_sha256"] == sha(lock),
            "Source-test release lock changed",
        )
        wheel_hashes = {
            p: h
            for p, h in records.items()
            if p.startswith(selected["wheelhouse"] + "/") and p.endswith(".whl")
        }
        require(
            {h for _, h in pins(lock).values()} == set(wheel_hashes.values()),
            "Release lock disagrees with the issued wheels",
        )
        require(
            records[manifest["runtime"]["archive"]] == manifest["runtime"]["sha256"],
            "PSF archive differs from its manifest",
        )
        manifests[name] = manifest
    require(
        all(
            manifests["core"][key] == manifests["simulation"][key]
            for key in ("source", "runtime", "target", "pip_version")
        )
        and pins(report.with_suffix(".core.lock.txt"))[PRODUCT]
        == pins(report.with_suffix(".simulation.lock.txt"))[PRODUCT],
        "Source-test editions have different source, product wheel or interpreter",
    )
    return manifests


def recheck(report, data, identity, source_hash, *, bundle=None, simulation_identity=None):
    path = report.with_suffix(".environment.json")
    require(sha(path) == data["environment_sha256"], "Source-test environment receipt changed")
    env = json.loads(path.read_bytes())
    sim = data["simulation_manifest_sha256"]
    require(
        (simulation_identity is None or sim == simulation_identity)
        and env["core_manifest_sha256"] == identity
        and env["simulation_manifest_sha256"] == sim,
        "Source-test simulation or core release differs",
    )
    manifests = release_records(report, identity, sim)
    core = manifests["core"]
    require(
        core["source"]["archive_sha256"] == source_hash
        and core["source"]["git_commit"] == data["support_git_commit"]
        and core["source"]["file_count"] == data["source_files_byte_verified"],
        "Source-test checkout identity differs",
    )
    extra = report.with_suffix(".test.lock.txt")
    supplements = pins(extra)
    runtime = pins(report.with_suffix(".simulation.lock.txt"))
    require(
        env["schema_version"] == "1.0.0"
        and env["execution_layer"] == "isolated_source_test_venv"
        and not supplements.keys() & runtime.keys()
        and env["test_lock_sha256"] == sha(extra)
        and set(env["test_wheels"]) == set(supplements),
        "Source-test supplement lock differs",
    )
    for name, wheel in env["test_wheels"].items():
        filename = wheel["filename"]
        parts = filename.split("-")
        require(
            wheel["sha256"] == supplements[name][1]
            and Path(filename).name == filename
            and filename.endswith(".whl")
            and len(parts) in {5, 6}
            and normal(parts[0]) == name
            and parts[1] == supplements[name][0],
            "Source-test wheel identity differs",
        )
    expected = {n: v for n, (v, _) in {**runtime, **supplements}.items()}
    expected["pip"] = core["pip_version"]
    obs = env["probe"]
    require(
        obs["installed"] == expected
        and obs["isolated"] is True
        and obs["user_site"] is False
        and obs["all_distributions_in_test_venv"] is True
        and obs["editable_source"] == "autotem/__init__.py"
        and obs["python_version"] == data["python_version"] == core["target"]["python_version"]
        and re.fullmatch("[a-f0-9]{64}", obs["base_executable_sha256"])
        and obs["base_executable_sha256"] == env["psf_executable_sha256"]
        and env["pip_check"] == "passed",
        "Installed source-test environment differs from its locks or source",
    )
    origin = env["test_lock_origin"]
    require(
        origin in {"archived_source", "explicit_legacy_lock"}, "Unknown source-test lock origin"
    )
    if bundle is not None:
        require(
            sha(bundle / "release-manifest.json") == identity, "Source-test core bundle changed"
        )
        archive = bundle / core["source"]["archive"]
        require(sha(archive) == source_hash, "Source-test source archive changed")
        with zipfile.ZipFile(archive) as z:
            archived = TEST_LOCK in z.namelist()
            require(
                archived == (origin == "archived_source")
                and (not archived or z.read(TEST_LOCK) == extra.read_bytes()),
                "Source-test lock differs from source archive",
            )
        runtime_archive = bundle / core["runtime"]["archive"]
        require(
            sha(runtime_archive) == core["runtime"]["sha256"], "Source-test PSF archive changed"
        )
        with zipfile.ZipFile(runtime_archive) as z:
            require(
                hashlib.sha256(z.read("tools/python.exe")).hexdigest()
                == env["psf_executable_sha256"],
                "Source-test base interpreter differs from PSF",
            )
    return {
        "verdict": "passed",
        "sha256": data["environment_sha256"],
        "simulation_manifest_sha256": sim,
        "test_lock_sha256": sha(extra),
        "test_lock_origin": origin,
        "installed_distributions": len(expected),
    }


if __name__ == "__main__":
    print(json.dumps(probe(Path(sys.argv[1]).resolve())))
