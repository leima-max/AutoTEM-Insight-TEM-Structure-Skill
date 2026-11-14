"use strict";

const { execFile } = require("node:child_process");
const { readFileSync, realpathSync } = require("node:fs");
const path = require("node:path");

const binding = JSON.parse(readFileSync(path.join(__dirname, "project.json"), "utf8"));
const project = realpathSync.native(binding.project);

function samePath(left, right) {
  return path.relative(left, right) === "";
}

function inProject(filename) {
  const relative = path.relative(project, filename);
  return relative !== ".." && !relative.startsWith(".." + path.sep) && !path.isAbsolute(relative);
}

function normalizeAgent(value) {
  const id = (value || "").trim().toLowerCase();
  if (/^[a-z0-9][a-z0-9_-]{0,63}$/.test(id)) return id;
  return id.replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 64) || "main";
}

function ruleFor(ctx) {
  if (!ctx.workspaceDir || ctx.sandboxed) return null;
  try {
    if (!samePath(project, realpathSync.native(ctx.workspaceDir))) return null;
  } catch {
    return null;
  }
  const cfg = ctx.getRuntimeConfig?.() ?? ctx.runtimeConfig ?? ctx.config;
  const entries = cfg?.agents?.list ?? [];
  const key = (ctx.sessionKey || "").trim().toLowerCase();
  const fallback = entries.find((entry) => entry.default === true) ?? entries[0];
  const agentId = normalizeAgent(ctx.agentId || /^agent:([^:]+):/.exec(key)?.[1] || fallback?.id);
  const agent = entries.find((entry) => normalizeAgent(entry.id) === agentId);
  // Operator HTTP on 2026.5.18 supplies the agent/session/config but omits the
  // turn's sandboxed and fsPolicy flags. Respect the corresponding agent settings.
  const sandbox = agent?.sandbox?.mode ?? cfg?.agents?.defaults?.sandbox?.mode ?? "off";
  const main = (cfg?.session?.mainKey || "main").trim().toLowerCase();
  const mainAliases = ["main", main, `agent:${agentId}:${main}`, `agent:${agentId}:main`,
                       `agent:main:${main}`, "agent:main:main"];
  const primary = mainAliases.includes(key) || (cfg?.session?.scope === "global" && key === "global");
  if (sandbox === "all" || (sandbox === "non-main" && key && !primary)) return null;
  const workspaceOnly = ctx.fsPolicy?.workspaceOnly === true ||
    (agent?.tools?.fs?.workspaceOnly ?? cfg?.tools?.fs?.workspaceOnly) === true;
  return { workspaceOnly };
}

function projectInput(value, expandsHome = false, directory = project) {
  if (typeof value !== "string") return value; // the published schema handles type errors
  // pathlib drops empty and "." components before expanduser, including ".\\~\\x".
  // A literal tilde-named folder can still be addressed by its absolute project path.
  const win = process.platform === "win32";
  const parts = (win ? value.replace(/^[A-Za-z]:/, "") : value).split(win ? /[/\\]+/ : /\/+/);
  const first = parts.find((part) => part && part !== ".");
  if (expandsHome && !path.isAbsolute(value) && first?.startsWith("~")) {
    throw new Error("Use a workspace path without home expansion.");
  }
  if (win && parts.some((part) => part && part !== "." && part !== ".." &&
      (/[ .]$|:/.test(part) || /^(?:con|prn|aux|nul|conin\$|conout\$|(?:com|lpt)[0-9¹²³])(?:\.|$)/i.test(part)))) {
    throw new Error("Windows device, stream or aliased name is not an ordinary workspace path.");
  }
  const filename = path.resolve(directory, value);
  if (!inProject(filename)) throw new Error("AutoTEM input is outside this agent's workspace.");
  // Resolve the closest existing ancestor, including a junction. Preserve a genuinely
  // missing contained input so the Harness can return its normal INPUT_NOT_FOUND.
  let parent = filename;
  let resolved;
  for (;;) {
    try {
      resolved = realpathSync.native(parent);
      break;
    } catch (err) {
      if (err.code !== "ENOENT" || samePath(parent, project)) {
        throw new Error("AutoTEM cannot verify this workspace input.");
      }
      parent = path.dirname(parent);
    }
  }
  if (!inProject(resolved)) throw new Error("AutoTEM input resolves outside this agent's workspace.");
  return path.join(resolved, path.relative(parent, filename));
}

function checkRunInputs(runId) {
  if (typeof runId !== "string" || !/^[a-f0-9]{32}$/.test(runId)) return;
  // For existing jobs the Harness may adopt the parent's stored source, or a pending
  // step from the registered plan. Read only this store and this run, before dispatch.
  const database = projectInput(path.join(binding.store, "harness.sqlite3"));
  const { DatabaseSync } = require("node:sqlite");
  const db = new DatabaseSync(database, { readOnly: true });
  try {
    const row = db.prepare("SELECT input_path, request FROM runs WHERE run_id = ?").get(runId);
    if (!row) return; // let the Harness report its normal unknown-ID diagnostic
    if (row.input_path) projectInput(row.input_path);
    // The worker starts at the Harness store; older runs may contain relative CIFs.
    const request = JSON.parse(row.request);
    builtinsOnly(request.enabled_plugins);
    bindCandidates(request.steps, binding.store);
  } finally {
    db.close();
  }
}

function bindCandidates(steps, directory = project) {
  if (!Array.isArray(steps)) return steps;
  return steps.map((step) => {
    const candidates = step?.parameters?.cif_candidate_paths;
    if (!Array.isArray(candidates)) return step;
    const scoped = candidates.map((filename) => projectInput(filename, false, directory));
    return { ...step, parameters: { ...step.parameters, cif_candidate_paths: scoped } };
  });
}

function builtinsOnly(plugins) {
  if (Array.isArray(plugins) && plugins.length) {
    throw new Error("Workspace-limited tools support built-in operations. Additional operation plugins require a reviewed service.");
  }
}

function scopedArguments(name, args, workspaceOnly) {
  if (!workspaceOnly || !args) return args;
  if (name === "list_capabilities") builtinsOnly(args.enabled_plugins);
  if (name === "inspect_input") return { ...args, path: projectInput(args.path, true) };
  if (name !== "submit_analysis" && name !== "validate_plan") return args;
  if (!args.plan || typeof args.plan !== "object") return args;
  builtinsOnly(args.plan.enabled_plugins);
  projectInput(args.plan.input_path, true);
  const plan = { ...args.plan };
  if (Array.isArray(plan.steps)) plan.steps = bindCandidates(plan.steps);
  const response = plan.user_input?.response;
  const answer = response?.answers?.cif_candidate_paths;
  if (Array.isArray(answer?.value)) {
    const answers = { ...response.answers,
      cif_candidate_paths: { ...answer, value: answer.value.map((filename) => projectInput(filename)) } };
    plan.user_input = { ...plan.user_input, response: { ...response, answers } };
  }
  if (args.parent_run_id) checkRunInputs(args.parent_run_id);
  if (args.run_id) checkRunInputs(args.run_id);
  // Keep the source spelling for immutable parent/child comparison and idempotency.
  return { ...args, plan };
}

function interpreter() {
  if (!binding.runtime_id) return { exe: binding.python };
  try {
    const control = path.join(project, ".autotem");
    const active = JSON.parse(readFileSync(path.join(control, "active-runtime.json"), "utf8"));
    const marker = JSON.parse(readFileSync(path.join(control, "runtimes", binding.runtime_id, ".autotem-runtime.json"), "utf8"));
    if (!/^[a-f0-9]{64}$/.test(active.manifest_sha256) || active.runtime_id !== binding.runtime_id ||
        active.runtime_id !== "rt-" + active.manifest_sha256.slice(0, 20) ||
        marker.managed_by !== "autotem.bootstrap" || marker.state !== "ready" ||
        marker.manifest_sha256 !== active.manifest_sha256 || !samePath(marker.project, project) ||
        !samePath(marker.runtime, path.dirname(path.dirname(binding.python))) ||
        !samePath(realpathSync.native(binding.python), binding.python)) throw new Error();
    // Avoid the Windows venv redirector's extra process. CPython uses this marker
    // to load the registered venv while the owned process handle points at the actual runner.
    if (process.platform === "win32") {
      return { exe: binding.windows_base_python, env: { __PYVENV_LAUNCHER__: binding.python } };
    }
    return { exe: binding.python };
  } catch {
    throw new Error("AutoTEM activation changed. Refresh the project installation and reload this plugin.");
  }
}

function callHarness(name, id, args, signal, workspaceOnly) {
  signal?.throwIfAborted();
  const request = JSON.stringify({ tool: name, arguments: args, request_id: id }) + "\n";
  if (Buffer.byteLength(request) > 1024 * 1024) throw new Error("AutoTEM JSONL request exceeds 1 MiB.");
  const requested = args?.plan?.budgets?.run_seconds;
  const budget = Number.isFinite(requested) && requested > 0 ? Math.min(86400, requested) : 900;
  const seconds = name === "submit_analysis" || name === "inspect_input" ? 60 + budget : 150;
  const runner = interpreter();
  const env = { ...process.env, ...runner.env };
  delete env.OPENCLAW_GATEWAY_TOKEN;
  delete env.OPENCLAW_GATEWAY_PASSWORD;
  const argv = ["-I", "-m", "autotem.harness.cli", "--root", binding.store, "stdio"];
  if (workspaceOnly) argv.push("--workspace-only", project);
  return new Promise((resolve, reject) => {
    const child = execFile(runner.exe, argv, {
      cwd: project, shell: false, windowsHide: true, encoding: "utf8", env, signal,
      timeout: seconds * 1000, maxBuffer: 32 * 1024 * 1024,
    }, (err, stdout) => {
      if (err) {
        // Host diagnostics may contain sensitive process arguments or Python stderr.
        const code = err.name === "AbortError" ? "ABORTED" :
          err.killed ? "TIMEOUT" : typeof err.code === "number" ? `EXIT_${err.code}` : "PROCESS_ERROR";
        let message = `AutoTEM JSONL bridge: ${code}.`;
        if (name === "submit_analysis" && (code === "ABORTED" || code === "TIMEOUT")) {
          message += " Submission status is unknown. If a run ID was received, check it or explicitly cancel it." +
            " For a retry, use the original arguments and existing idempotency key.";
        }
        reject(new Error(message));
        return;
      }
      let envelope;
      try {
        envelope = JSON.parse(stdout);
      } catch {
        reject(new Error("AutoTEM JSONL bridge returned unreadable output."));
        return;
      }
      if (envelope?.tool !== name || envelope.request_id !== id || typeof envelope.ok !== "boolean") {
        reject(new Error("AutoTEM JSONL response does not match the requested call."));
        return;
      }
      resolve({ content: [{ type: "text", text: JSON.stringify(envelope) }], isError: !envelope.ok });
    });
    child.stdin.on("error", (err) => {
      // EPIPE commonly races with a normal child exit or abort. execFile reports that result.
      if (err.code !== "EPIPE" && err.code !== "ERR_STREAM_DESTROYED") {
        reject(new Error("AutoTEM input stream closed unexpectedly."));
        child.kill();
      }
    });
    child.stdin.end(request, "utf8");
  });
}

module.exports = {
  id: "autotem-harness",
  name: "AutoTEM Harness",
  register(api) {
    for (const spec of binding.tools) {
      api.registerTool((ctx) => {
        const rule = ruleFor(ctx);
        if (!rule) return null;
        return {
          name: spec.hostName, label: spec.hostName, description: spec.description, parameters: spec.parameters,
          execute(id, args, signal) {
            return callHarness(spec.name, id, scopedArguments(spec.name, args, rule.workspaceOnly), signal, rule.workspaceOnly);
          },
        };
      }, { name: spec.hostName, optional: true });
    }
  },
};
