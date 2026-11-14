"""Project-local OpenClaw registration shares the nine public dispatch schemas."""
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from autotem.harness.cli import TOOL_DESCRIPTIONS, TOOL_SCHEMAS
from autotem.harness.hosts import install_host, rollback_host
from autotem.harness.openclaw import PLUGIN_DIR, PLUGIN_ID


def test_installed_openclaw_gateway_files_are_exact_opt_in_and_rollback(tmp_path):
    project = tmp_path / "新工程"
    project.mkdir()
    result = install_host(project, "openclaw", dry_run=False)
    assert result["ok"]
    directory = project / PLUGIN_DIR
    manifest = json.loads((directory / "openclaw.plugin.json").read_bytes())
    binding = json.loads((directory / "project.json").read_bytes())
    spec = json.loads((project / "autotem-host/openclaw/adapter.json").read_bytes())
    example = json.loads((project / spec["gateway_plugin"]["config_example"]).read_bytes())
    tools = binding["tools"]
    assert [tool["hostName"] for tool in tools] == manifest["contracts"]["tools"]
    assert {tool["name"]: tool["parameters"] for tool in tools} == TOOL_SCHEMAS
    assert {tool["name"]: tool["description"] for tool in tools} == TOOL_DESCRIPTIONS
    assert all(tool["hostName"] == "autotem__" + tool["name"] for tool in tools)
    assert manifest["toolMetadata"] == {name: {"optional": True} for name in manifest["contracts"]["tools"]}
    assert example["tools"]["alsoAllow"] == manifest["contracts"]["tools"]
    assert example["plugins"]["entries"] == {PLUGIN_ID: {"enabled": True}}
    assert example["plugins"]["load"]["paths"] == [str(directory)]
    assert binding["python"] == spec["jsonl_command"][0] and Path(binding["project"]) == project
    assert spec["gateway_plugin"]["activation"] == "operator_opt_in" and not (project / ".openclaw").exists()
    assert rollback_host(project, result["installation_id"], dry_run=False)["ok"]
    assert not (directory / "index.cjs").exists()


def test_native_factories_require_matching_unrestricted_workspace(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("This test needs the OpenClaw Node runtime.")
    project = tmp_path / "测量 工程"
    project.mkdir()
    assert install_host(project, "openclaw", dry_run=False)["ok"]
    other = tmp_path / "其他工程"
    other.mkdir()
    binding = json.loads((project / PLUGIN_DIR / "project.json").read_bytes())
    store = Path(binding["store"])
    store.mkdir(parents=True)
    with sqlite3.connect(store / "harness.sqlite3") as db:
        db.execute("CREATE TABLE runs (run_id TEXT, input_path TEXT, request TEXT)")
        db.executemany("INSERT INTO runs VALUES (?,?,?)", [
            ("a" * 32, str(other / "outside.npy"), '{"steps":[]}'),
            ("b" * 32, str(project / "local.npy"),
             json.dumps({"steps": [{"parameters": {"cif_candidate_paths": [str(other / "other.cif")]}}]})),
        ])
    script = r"""
const assert = require("node:assert/strict");
const [entry, project, other] = process.argv.slice(1);
const registered = new Map();
require(entry).register({registerTool(factory, opt) {
  assert.equal(opt.optional, true);
  assert(!registered.has(opt.name));
  registered.set(opt.name, factory);
}});
assert.equal(registered.size, 9);
for (const [name, factory] of registered) {
  const tool = factory({workspaceDir: project});
  assert.equal(tool.name, name);
  assert.equal(tool.parameters.type, "object");
  assert.equal(typeof tool.execute, "function");
  assert(factory({workspaceDir: project, fsPolicy:{workspaceOnly:true}}));
  for (const ctx of [{}, {workspaceDir: other}, {workspaceDir: project, sandboxed:true},
                     {workspaceDir: project, config:{agents:{defaults:{sandbox:{mode:"all"}}}}}]) {
    assert.equal(factory(ctx), null);
  }
}
const scoped = {workspaceDir:project, config:{tools:{fs:{workspaceOnly:true}}}};
const inspect = registered.get("autotem__inspect_input")(scoped);
assert.throws(() => inspect.execute("test", {path:other}), /outside/);
const submit = registered.get("autotem__submit_analysis")(scoped);
assert.throws(() => submit.execute("test", {plan:{input_path:"file.npy", steps:[
  {parameters:{cif_candidate_paths:[other]}}]}}), /outside/);
for (const key of ["autotem__submit_analysis", "autotem__validate_plan"]) {
  const tool = registered.get(key)(scoped);
  const answered = (file) => ({plan:{input_path:"file.npy", steps:[], user_input:{
    response:{answers:{cif_candidate_paths:{value:[file]}}}}}});
  assert.throws(() => tool.execute("test", answered(other)), /outside/);
  const noSpawn = new AbortController();
  noSpawn.abort();
  assert.throws(() => tool.execute("test", answered(project+"/permitted.cif"),noSpawn.signal), /abort/i);
  const homeSpellings = ["~/outside.npy","~someone/else.npy","./~/outside.npy","././~someone/else.npy"];
  if (process.platform === "win32") homeSpellings.push("~\\outside.npy","~someone\\else.npy",".\\.\\~\\outside.npy");
  for (const value of homeSpellings) {
    for (const caller of [{}, {run_id:"a".repeat(32)}, {parent_run_id:"a".repeat(32)}]) {
      assert.throws(() => tool.execute("test", {...caller,plan:{input_path:value}}), /home expansion/);
    }
  }
  if (process.platform === "win32") for (const alias of ["hop./x","hop /x","hop../x",
      "hop. /x","hop::$INDEX_ALLOCATION/x","sample.npy:data","NUL.npy"]) {
    assert.throws(() => tool.execute("test",{plan:{input_path:alias}}), /Windows device/);
    assert.throws(() => tool.execute("test",answered(alias)), /Windows device/);
    assert.throws(() => tool.execute("test",{plan:{input_path:"normal.npy",steps:[
      {parameters:{cif_candidate_paths:[alias]}}]}}), /Windows device/);
  }
  assert.throws(() => tool.execute("test", {plan:{input_path:project+"/~folder/inside.npy"}}, noSpawn.signal), /abort/i);
}
for (const key of ["run_id","parent_run_id"]) {
  for (const fill of ["a","b"]) assert.throws(() => submit.execute("test",
    {[key]:fill.repeat(32),plan:{input_path:"file.npy",steps:[]}}), /outside/);
}
const factory = registered.get("autotem__inspect_input");
const context = {workspaceDir:project,agentId:"researcher",sessionKey:"agent:researcher:main"};
const cfg = {tools:{fs:{workspaceOnly:false}},
             agents:{list:[{id:"Researcher",default:true,tools:{fs:{workspaceOnly:true}},sandbox:{mode:"non-main"}}]}};
for (const agentId of ["Researcher", "researcher", undefined]) {
  assert.throws(() => factory({...context,agentId,config:cfg}).execute("test",{path:other}), /outside/);
}
assert.equal(factory({...context,sessionKey:"agent:researcher:other",config:cfg}),null);
cfg.agents.list[0].sandbox.mode = "all";
assert.equal(factory({...context,config:cfg}),null);
"""
    subprocess.run([node, "-e", script, project / PLUGIN_DIR / "index.cjs", project, other],
                   capture_output=True, text=True, check=True, timeout=20)


def test_cancel_and_diagnostics_in_a_real_child_process(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("This test needs the OpenClaw Node runtime.")
    project = tmp_path / "中断测试"
    project.mkdir()
    assert install_host(project, "openclaw", dry_run=False)["ok"]
    directory = project / PLUGIN_DIR
    binding = json.loads((directory / "project.json").read_bytes())
    binding["python"] = node
    (directory / "project.json").write_text(json.dumps(binding), encoding="utf-8")
    # Keep the production spawn options and callback, replacing only the checked CLI's
    # arguments with controlled Node source to exercise an actual OS process.
    driver = r"""
const assert = require("node:assert/strict");
const fs = require("node:fs");
const cp = require("node:child_process");
const {setTimeout:wait} = require("node:timers/promises");
const [entry, workspace] = process.argv.slice(1);
const original = cp.execFile;
let owned;
let behavior = 'setTimeout(()=>{}, 60000)';
cp.execFile = (exe, argv, options, callback) => {
  assert.deepEqual(argv.slice(0,4), ["-I","-m","autotem.harness.cli","--root"]);
  owned = original(exe, ["-e", behavior], options, callback);
  return owned;
};
const tools = {};
const scoped = {};
require(entry).register({registerTool(factory, options) {
  tools[options.name]=factory({workspaceDir:workspace});
  scoped[options.name]=factory({workspaceDir:workspace,fsPolicy:{workspaceOnly:true}});
}});
(async () => {
  const ctrl = new AbortController();
  const call = tools.autotem__list_capabilities.execute("cancel",{},ctrl.signal);
  const pid = owned.pid;
  assert(pid > 0);
  ctrl.abort();
  await assert.rejects(call,/ABORTED/);
  for (let i=0; owned.exitCode === null && owned.signalCode === null && i<80; ++i) await wait(25);
  assert.throws(() => process.kill(pid,0), (err) => err.code === "ESRCH");
  const submission = new AbortController();
  const pending = tools.autotem__submit_analysis.execute("submit",{plan:{}},submission.signal);
  submission.abort();
  await assert.rejects(pending, (err) => /ABORTED/.test(err.message)
    && err.message.includes("Submission status is unknown") && err.message.includes("existing idempotency key"));
  behavior = 'console.error("SYNTHETIC_PRIVATE_DIAGNOSTIC");process.exit(19)';
  await assert.rejects(tools.autotem__list_capabilities.execute("error",{}), (err) =>
    err.message.includes("EXIT_19") && !err.message.includes("SYNTHETIC_PRIVATE_DIAGNOSTIC"));
  behavior = 'let s="";process.stdin.on("data",d=>s+=d).on("end",()=>{const r=JSON.parse(s);' +
    'console.log(JSON.stringify({tool:r.tool,request_id:r.request_id,ok:true,result:r.arguments,error:null}));});';
  const cif = "candidate/候选.cif";
  const expected = require("node:path").join(workspace,cif);
  for (const name of ["autotem__submit_analysis","autotem__validate_plan"]) {
    const plan = {input_path:"./raw.npy",steps:[{parameters:{cif_candidate_paths:[cif]}}],
        user_input:{response:{answers:{cif_candidate_paths:{value:[cif]}}}}};
    const data = await scoped[name].execute("forwarded",{plan});
    const sent = JSON.parse(data.content[0].text).result.plan;
    assert.deepEqual(sent.steps[0].parameters.cif_candidate_paths,[expected]);
    assert.deepEqual(sent.user_input.response.answers.cif_candidate_paths.value,[expected]);
    assert.equal(sent.input_path,plan.input_path);
    assert.deepEqual(plan.steps[0].parameters.cif_candidate_paths,[cif]);
    assert.throws(() => scoped[name].execute("plugin",{plan:{...plan,enabled_plugins:["external"]}}),/built-in/);
  }
  assert.throws(() => scoped.autotem__list_capabilities.execute("plugin",{enabled_plugins:["external"]}),/built-in/);
})().catch((err) => { console.error(err); process.exitCode=1; });
"""
    subprocess.run([node, "-e", driver, directory / "index.cjs", project],
                   capture_output=True, text=True, check=True, timeout=20)


def test_workspace_container_reads_enforce_selected_hdf_storage_and_ser_companion(tmp_path):
    import h5py
    import numpy as np
    from PIL import Image
    from ncempy.io import ser

    from autotem.harness.contracts import HarnessError
    from autotem.harness.workspace import workspace_file
    from autotem.services.data_quality import InputParameters, _read_selected

    project, sibling = tmp_path / "工程", tmp_path / "外部"
    project.mkdir()
    sibling.mkdir()
    pixels = np.array([[77, 101], [97, 115]], dtype="u1")
    raw = sibling / "outside.bin"
    raw.write_bytes(pixels.tobytes())
    other = sibling / "outside.h5"
    with h5py.File(other, "w") as f:
        f.create_dataset("image", data=pixels)
    mixed = project / "mixed.emd"
    with h5py.File(mixed, "w") as f:
        f.create_dataset("local", data=pixels)
        f.create_dataset("raw", (2, 2), dtype="u1", external=[(str(raw), 0, 4)])
        layout = h5py.VirtualLayout((2, 2), dtype="u1")
        layout[:] = h5py.VirtualSource(str(other), "image", shape=(2, 2))
        f.create_virtual_dataset("virtual", layout)
        f["outside"] = h5py.ExternalLink(str(other), "/")
        f["indirect"] = h5py.SoftLink("/outside")
    for key in ("raw", "virtual", "outside/image", "indirect/image"):
        selection = InputParameters(dataset_path=key)
        with pytest.raises(HarnessError, match="self-contained") as exc:
            _read_selected(mixed, selection, workspace=project)
        assert exc.value.code == "WORKSPACE_INPUT_RESTRICTED"
    for key in ("raw", "virtual"):
        received, _, _ = _read_selected(mixed, InputParameters(dataset_path=key))
        np.testing.assert_array_equal(received, pixels)
    for path, key in [(mixed, "local"), (other, "image")]:
        received, _, _ = _read_selected(path, InputParameters(dataset_path=key), workspace=path.parent)
        np.testing.assert_array_equal(received, pixels)
    tiff = project / "take.tif"
    Image.fromarray(pixels).save(tiff)
    np.testing.assert_array_equal(_read_selected(tiff, InputParameters(), workspace=project)[0], pixels)
    # Use a normal SER header read fixture; the assertion concerns ncempy's next I/O.
    from unittest.mock import patch

    fake = project / "capture_1.ser"
    fake.write_bytes(b"SER")
    with patch.object(ser.fileSER, "readHeader", return_value={"ValidNumberElements": 1}), \
         patch.object(ser.fileSER, "getDataset", return_value=(pixels, {})), \
         patch.object(ser, "read_emi", return_value={}) as emi:
        local = project / "capture.emi"
        local.write_bytes(b"local metadata")
        np.testing.assert_array_equal(_read_selected(fake, InputParameters(), workspace=project)[0], pixels)
        emi.assert_called_once_with(local)
        local.unlink()
        alias = project / "outside"
        if __import__("os").name == "nt":
            subprocess.run(["cmd", "/c", "mklink", "/J", str(alias), str(sibling)],
                           check=True, capture_output=True)
        else:
            alias.symlink_to(sibling, target_is_directory=True)
        # child-of-junction also works for a missing companion: no external I/O occurs.
        with pytest.raises(HarnessError, match="outside"):
            workspace_file(alias / "absent.emi", project)
        try:
            local.symlink_to(raw)
        except OSError:
            # Windows users without the symlink privilege can create a junction to a
            # directory under this exact EMI filename. ncempy would try to read it.
            subprocess.run(["cmd", "/c", "mklink", "/J", str(local), str(sibling)],
                           check=True, capture_output=True)
        with pytest.raises(HarnessError, match="outside"):
            _read_selected(fake, InputParameters(), workspace=project)
        assert emi.call_count == 1


def test_queued_workspace_policy_is_specific_to_each_job(tmp_path, monkeypatch):
    import numpy as np
    from autotem.harness import Harness
    from autotem.harness.workspace import job_workspace

    project = tmp_path / "project"
    project.mkdir()
    source = project / "sample.npy"
    np.save(source, np.arange(16).reshape(4, 4))
    bounded = Harness(project / "store", workspace=project)
    normal = Harness(project / "store")
    plan = {"input_path": str(source), "steps": [{"id": "load", "operation_id": "input.load"}]}
    for harness, expected in ((bounded, project), (normal, None)):
        monkeypatch.setattr(harness, "_launch_scheduler", lambda _: None)
        run = harness.submit_analysis(plan)
        assert job_workspace(normal.store, run["run_id"], run["job_id"]) == expected
