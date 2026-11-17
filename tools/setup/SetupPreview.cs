using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;

internal static partial class Setup {
    static long ExpandedBytes(string archive, string prefix) {
        long size = 0;
        using (var stream = File.OpenRead(archive))
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Read)) {
            foreach (var entry in zip.Entries) {
                if (entry.Name.Length != 0 && entry.FullName.StartsWith(prefix, StringComparison.Ordinal))
                    size = checked(size + entry.Length);
            }
        }
        return size;
    }

    static Dictionary<string, object> ProjectRecord(string path) {
        CheckPath(path);
        return File.Exists(path)
            ? Json.Deserialize<Dictionary<string, object>>(File.ReadAllText(path))
            : null;
    }

    static List<string> ProjectTargets(string host, string profile, string runtimeId,
            Dictionary<string, object> active, bool rollback) {
        var hosts = new SortedSet<string>(StringComparer.Ordinal);
        if (active != null && active.ContainsKey("host_installations")) {
            var owners = active["host_installations"] as Dictionary<string, object>;
            if (owners == null) throw new IOException("项目中的宿主安装记录无法读取。");
            foreach (var owner in owners.Keys) {
                if (owner != "codex" && owner != "openclaw" && owner != "workbuddy")
                    throw new IOException("项目宿主记录包含未知名称。");
                hosts.Add(owner);
            }
        }
        if (!rollback) hosts.Add(host);
        var paths = new List<string> {
            ".autotem/active-runtime.json", ".autotem/host-entry.py",
            ".autotem/host-installations/", ".autotem/installations/", ".autotem/runs/harness.sqlite3"
        };
        if (!rollback) {
            paths.Add(".autotem/python/");
            paths.Add(".autotem/runtimes/" + runtimeId + "/");
        }
        foreach (var name in hosts) {
            paths.Add("autotem-host/" + name + "/");
            if (name == "codex") {
                paths.Add(".agents/skills/autotem/");
                if (profile != "traditional-cli" || (active != null && active.ContainsKey("profile")
                        && (string)active["profile"] != "traditional-cli") || rollback)
                    paths.Add(".codex/config.toml");
            } else if (name == "openclaw") {
                paths.Add("skills/autotem/");
            }
        }
        return paths;
    }

    static Dictionary<string, object> InstallPreview(string root, string host, string profile,
            Dictionary<string, object> manifest, string python, bool rollback) {
        string manifestHash = FileHash(Path.Combine(Bundle, "release-manifest.json"));
        string runtimeId = "rt-" + manifestHash.Substring(0, 20);
        string control = Path.Combine(root, ".autotem");
        var active = ProjectRecord(Path.Combine(control, "active-runtime.json"));
        Dictionary<string, object> marker = null;
        if (!rollback) {
            // A partial marker makes the estimate conservative. The installer diagnoses or repairs it.
            try { marker = ProjectRecord(Path.Combine(control, "runtimes", runtimeId, ".autotem-runtime.json")); }
            catch (ArgumentException) { }
            catch (InvalidOperationException) { }
        }
        bool reusable = marker != null && marker.ContainsKey("manifest_sha256")
            && manifestHash.Equals(marker["manifest_sha256"]) && marker.ContainsKey("state")
            && "ready".Equals(marker["state"]);
        long wheels = 0, expanded = 0;
        if (!rollback) {
            foreach (Dictionary<string, object> item in (IEnumerable)manifest["artifacts"]) {
                string member = (string)item["path"];
                if (member.StartsWith("wheelhouse/", StringComparison.Ordinal)
                        && member.EndsWith(".whl", StringComparison.OrdinalIgnoreCase)) {
                    wheels = checked(wheels + Convert.ToInt64(item["size"]));
                    if (!reusable) expanded = checked(expanded + ExpandedBytes(Path.Combine(Bundle, member), ""));
                }
            }
        }
        long baseBytes = File.Exists(python) ? 0 : ExpandedBytes(Archive, "tools/");
        // managed.manage requires three times the compressed wheel size after Python is deployed.
        long minimum = checked(baseBytes + 3 * wheels);
        long available = new DriveInfo(Path.GetPathRoot(root)).AvailableFreeSpace;
        bool fits = available >= minimum;
        string authorization = rollback ? "回滚后重新连接恢复的项目入口。"
            : profile == "traditional-cli" ? "使用项目 CLI 或 JSONL。" : "首次连接由所选宿主提示信任、导入或授权。";
        var plan = new Dictionary<string, object> {
            {"ok", fits}, {"state", fits ? "planned" : "blocked"}, {"project", root}, {"profile", profile},
            {"host", host}, {"python", python}, {"action", rollback ? "rollback" : "install"}, {"release_ready", false},
            {"target_runtime_id", runtimeId}, {"active_runtime_id", active != null && active.ContainsKey("runtime_id") ? active["runtime_id"] : null},
            {"existing_project_control", Directory.Exists(control)}, {"available_bytes", available},
            {"minimum_free_bytes", minimum}, {"estimated_new_payload_bytes", checked(baseBytes + expanded)},
            {"affected_project_paths", ProjectTargets(host, profile, runtimeId, active, rollback)},
            {"host_authorization", authorization},
            {"next_action", fits ? "请核对项目、磁盘空间和项目内的写入范围。" : "项目盘可用空间不足。请释放空间或选择其他本地项目后重试。"}
        };
        if (!fits) plan["error"] = new {code="DISK_SPACE_LOW", message=plan["next_action"]};
        return plan;
    }

    static string PlanText(Dictionary<string, object> plan) {
        if (!plan.ContainsKey("available_bytes")) return Convert.ToString(plan["next_action"]);
        var paths = new List<string>();
        foreach (string target in (IEnumerable)plan["affected_project_paths"]) paths.Add(target);
        return String.Format("{0}\r\n项目盘可用 {1:N0} MiB，最低预留 {2:N0} MiB，预计新增载荷 {3:N0} MiB。\r\n当前活动版本 {4}\r\n可能写入的项目路径\r\n{5}\r\n{6}",
                plan["project"], Convert.ToInt64(plan["available_bytes"]) / 1048576.0,
                Convert.ToInt64(plan["minimum_free_bytes"]) / 1048576.0,
                Convert.ToInt64(plan["estimated_new_payload_bytes"]) / 1048576.0, plan["active_runtime_id"] ?? "无",
                String.Join("\r\n", paths), plan["host_authorization"]);
    }

    static string ConflictText(Dictionary<string, object> report) {
        if (!report.ContainsKey("conflict_resolution")) return "";
        var details = report["conflict_resolution"] as Dictionary<string, object>;
        if (details == null || !details.ContainsKey("files")) return "";
        var lines = new List<string>();
        foreach (Dictionary<string, object> file in (IEnumerable)details["files"]) {
            lines.Add(Convert.ToString(file["path"]));
            object state = "directory".Equals(file["current_state"]) ? "同名目录" : "文件缺失";
            lines.Add("当前 SHA256  " + (file["current_sha256"] ?? state));
            if (file["expected_sha256"] != null) lines.Add("受管版本 SHA256  " + file["expected_sha256"]);
            if (file["comparison_path"] != null) lines.Add("项目内对照文件  " + file["comparison_path"]);
        }
        return lines.Count == 0 ? "" : "\r\n\r\n" + String.Join("\r\n", lines);
    }
}
