// Windows .NET Framework launcher for the pinned PSF distribution.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.IO.Compression;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows.Forms;

internal static partial class Setup {
    static readonly string Bundle = AppDomain.CurrentDomain.BaseDirectory;
    static readonly string App = Path.Combine(Bundle, "AutoTEM-install.pyz");
    static readonly string Archive = Path.Combine(Bundle, "runtime", "python." + BundlePins.Version + ".nupkg");
    static readonly JavaScriptSerializer Json = new JavaScriptSerializer();
    static readonly Encoding WireEncoding = new UTF8Encoding(false, true);

    [DllImport("kernel32.dll")]
    static extern IntPtr GetStdHandle(int handle);
    [DllImport("kernel32.dll", CharSet=CharSet.Unicode, ExactSpelling=true, SetLastError=true)]
    static extern bool WriteConsoleW(IntPtr handle, string text, int length, out int written, IntPtr reserved);

    static void WriteReply(string json) {
        if (Console.IsOutputRedirected) {
            // CLI clients and redirected files receive one BOM-free UTF-8 JSON line.
            using (var writer = new StreamWriter(Console.OpenStandardOutput(), WireEncoding))
                writer.WriteLine(json);
            return;
        }
        // An attached legacy Windows console accepts UTF-16 directly. Leave the host code page alone.
        string line = json + Environment.NewLine;
        IntPtr stdout = GetStdHandle(-11);
        for (int offset=0; offset<line.Length;) {
            int count = Math.Min(8192, line.Length-offset);
            if (char.IsHighSurrogate(line[offset+count-1]) && offset+count<line.Length) count--;
            int written;
            if (!WriteConsoleW(stdout,line.Substring(offset,count),count,out written,IntPtr.Zero) || written==0)
                throw new IOException("Could not write the installer response to the console.");
            offset+=written;
        }
    }

    static string Hash(Stream stream) {
        using (var sha = SHA256.Create()) {
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
        }
    }
    static string FileHash(string path) {
        using (var stream = File.OpenRead(path)) return Hash(stream);
    }
    static void CheckPath(string path) {
        for (var item = new FileInfo(Path.GetFullPath(path)); item != null; ) {
            FileAttributes attributes;
            try { attributes = File.GetAttributes(item.FullName); }
            catch (FileNotFoundException) { attributes = 0; }
            catch (DirectoryNotFoundException) { attributes = 0; }
            if ((attributes & FileAttributes.ReparsePoint) != 0)
                throw new IOException("Project path contains a reparse point.");
            var parent = Path.GetDirectoryName(item.FullName);
            item = String.IsNullOrEmpty(parent) ? null : new FileInfo(parent);
        }
    }
    static string ProjectPath(string input) {
        var root = Path.GetFullPath(input).TrimEnd('\\');
        if (!Directory.Exists(root) || root.Length <= 3 || root.StartsWith(@"\\"))
            throw new IOException("Choose an existing local project folder.");
        CheckPath(root);
        var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        foreach (var blocked in new [] {home, Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    Environment.GetFolderPath(Environment.SpecialFolder.CommonApplicationData),
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData)})
            if (String.Equals(root, blocked, StringComparison.OrdinalIgnoreCase))
                throw new IOException("Choose a project folder for this installation.");
        foreach (var blocked in new [] {Environment.SystemDirectory, Environment.GetEnvironmentVariable("SystemRoot"),
                    Path.Combine(home, ".codex"), Environment.GetEnvironmentVariable("CODEX_HOME"),
                    Path.Combine(home, ".openclaw"), Path.Combine(home, ".workbuddy")})
            if (!String.IsNullOrEmpty(blocked) && (root + "\\").StartsWith(blocked.TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase))
                throw new IOException("Select a folder outside system and host configuration directories.");
        return root;
    }
    static Dictionary<string,object> Manifest() {
        return Json.Deserialize<Dictionary<string,object>>(File.ReadAllText(Path.Combine(Bundle, "release-manifest.json")));
    }
    static string Profile(Dictionary<string,object> manifest, string requested) {
        var profiles = manifest["profiles"] as Dictionary<string,object>;
        if (profiles == null || profiles.Count != 1) throw new IOException("Expected a single release profile.");
        foreach (var name in profiles.Keys) {
            if (Array.IndexOf(BundlePins.Profiles, name) < 0 || (requested != null && name != requested))
                throw new IOException("The requested profile is not available in this release directory.");
            return name;
        }
        throw new IOException("Missing release profile.");
    }
    static void VerifyBundle(Dictionary<string,object> manifest) {
        if (!Environment.Is64BitOperatingSystem) throw new IOException("Windows x64 is required.");
        if (FileHash(Archive) != BundlePins.Python || FileHash(App) != BundlePins.App)
            throw new IOException("The installer or private Python differs from this launcher's pinned bundle.");
        var artifacts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (Dictionary<string,object> item in (System.Collections.IEnumerable)manifest["artifacts"]) {
            var path = Path.GetFullPath(Path.Combine(Bundle, (string)item["path"]));
            if (!path.StartsWith(Bundle.TrimEnd('\\') + "\\", StringComparison.OrdinalIgnoreCase)
                || !artifacts.Add(path)) throw new IOException("Invalid or duplicate manifest path.");
            CheckPath(path);
            if (!File.Exists(path) || new FileInfo(path).Length != Convert.ToInt64(item["size"])
                || FileHash(path) != (string)item["sha256"]) throw new IOException("Missing or damaged release file: " + item["path"]);
        }
        if (!artifacts.Contains(Archive) || !artifacts.Contains(App))
            throw new IOException("The release manifest omits the installation bootstrap.");
    }
    static string DeployPython(string root, bool apply) {
        var baseParent = Path.Combine(root, ".autotem", "python");
        var target = Path.Combine(baseParent, "cpython-" + BundlePins.Version + "-" + BundlePins.Python.Substring(0,12));
        CheckPath(target);
        if (!apply) return Path.Combine(target, "python.exe");
        Directory.CreateDirectory(baseParent);
        string lockPath = Path.Combine(baseParent, "deployment.lock");
        CheckPath(lockPath);
        using (var lease = new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None)) {
        bool exists = Directory.Exists(target);
        var dest = exists ? target : target + ".pending";
        CheckPath(dest);
        string owner = Path.Combine(dest, ".autotem-python");
        if (!Directory.Exists(dest)) {
            Directory.CreateDirectory(dest);
            File.WriteAllText(owner, BundlePins.Python);
        } else if (!File.Exists(owner) || File.ReadAllText(owner) != BundlePins.Python)
            throw new IOException("The Python staging directory has another owner.");
        var known = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        known.Add(owner);
        using (var stream = File.OpenRead(Archive))
        using (var zip = new ZipArchive(stream, ZipArchiveMode.Read)) {
            foreach (var entry in zip.Entries) {
                if (!entry.FullName.StartsWith("tools/", StringComparison.Ordinal) || entry.Name.Length == 0) continue;
                var name = Path.GetFullPath(Path.Combine(dest, entry.FullName.Substring(6)));
                if (!name.StartsWith(dest + "\\", StringComparison.OrdinalIgnoreCase))
                    throw new IOException("Invalid path in Python archive.");
                CheckPath(name);
                known.Add(name);
                using (var data = entry.Open()) {
                    if (exists) {
                        if (!File.Exists(name) || new FileInfo(name).Length != entry.Length || FileHash(name) != Hash(data))
                            throw new IOException("Private Python files changed. Select a fresh project or review the installation.");
                    } else {
                        Directory.CreateDirectory(Path.GetDirectoryName(name));
                        // Only the identified, unactivated staging tree can be repaired.
                        using (var writer = new FileStream(name, FileMode.Create, FileAccess.Write))
                            data.CopyTo(writer);
                    }
                }
            }
        }
        var dirs = new Stack<string>(); dirs.Push(dest);
        while (dirs.Count>0) {
            var dir=dirs.Pop();
            foreach (var sub in Directory.GetDirectories(dir)) { CheckPath(sub); dirs.Push(sub); }
            foreach (var file in Directory.GetFiles(dir)) {
                CheckPath(file);
                if (!known.Contains(file) && !(Path.GetFileName(dir)=="__pycache__" && Path.GetExtension(file)==".pyc"))
                    throw new IOException("The private Python directory contains unregistered files.");
            }
        }
        if (!exists) Directory.Move(dest, target);
        }
        return Path.Combine(target, "python.exe");
    }
    static string Quote(string value) {
        var result = new StringBuilder("\""); int slashes = 0;
        foreach (char ch in value) {
            if (ch == '\\') { slashes++; continue; }
            result.Append('\\', ch == '"' ? 2 * slashes + 1 : slashes);
            result.Append(ch); slashes = 0;
        }
        result.Append('\\', 2 * slashes); result.Append('"'); return result.ToString();
    }
    // Maintenance can run outside the project even when its private interpreter is damaged.
    // Record each temporary file and remove it only if its bytes still match.
    sealed class TemporaryPython : IDisposable {
        readonly string root = Path.Combine(Path.GetTempPath(), "AutoTEM-remove-" + Guid.NewGuid().ToString("N"));
        readonly Dictionary<string,string> files = new Dictionary<string,string>(StringComparer.OrdinalIgnoreCase);
        readonly HashSet<string> dirs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        public string Executable { get { return Path.Combine(root, "python.exe"); } }
        public TemporaryPython() {
            CheckPath(root);
            if (Directory.Exists(root) || File.Exists(root)) throw new IOException("Temporary directory is occupied.");
            Directory.CreateDirectory(root); dirs.Add(root);
            try {
                using (var stream=File.OpenRead(Archive)) using (var zip=new ZipArchive(stream, ZipArchiveMode.Read)) {
                    foreach (var entry in zip.Entries) {
                        if (!entry.FullName.StartsWith("tools/", StringComparison.Ordinal) || entry.Name.Length==0) continue;
                        string dest=Path.GetFullPath(Path.Combine(root,entry.FullName.Substring(6)));
                        if (!dest.StartsWith(root+"\\",StringComparison.OrdinalIgnoreCase) || files.ContainsKey(dest))
                            throw new IOException("Invalid path in private Python archive.");
                        string dir=Path.GetDirectoryName(dest);
                        for (string p=dir; p!=root; p=Path.GetDirectoryName(p)) dirs.Add(p);
                        CheckPath(dest); Directory.CreateDirectory(dir);
                        using(var data=entry.Open()) using(var writer=new FileStream(dest,FileMode.CreateNew,FileAccess.Write))
                            data.CopyTo(writer);
                        using(var data=entry.Open()) files.Add(dest,Hash(data));
                        if (FileHash(dest)!=files[dest]) throw new IOException("Private cleanup interpreter failed verification.");
                    }
                }
            } catch { Dispose(); throw; }
        }
        public void Dispose() {
            foreach(var item in files) {
                CheckPath(item.Key);
                if(File.Exists(item.Key) && FileHash(item.Key)==item.Value) File.Delete(item.Key);
            }
            var paths=new List<string>(dirs); paths.Sort((a,b)=>b.Length.CompareTo(a.Length));
            foreach(var dir in paths) {
                CheckPath(dir);
                if(Directory.Exists(dir) && Directory.GetFileSystemEntries(dir).Length==0) Directory.Delete(dir);
            }
        }
    }
    static int Run(string project, string host, string requested, bool apply, bool rollback, bool uninstall,
            out string output, string reconnectFailure=null) {
        var root = ProjectPath(project);
        if (host == null && reconnectFailure == null) host = "codex";
        if (host != null && host != "codex" && host != "workbuddy" && host != "openclaw") throw new IOException("Unknown host.");
        if (reconnectFailure != null) {
            if (apply || rollback || uninstall)
                throw new IOException("请单独使用 --host-reconnect-failure，不能同时选择安装、回滚或卸载。");
            // ProcessStartInfo uses the project as its cwd. Resolve relative reports at the caller first.
            reconnectFailure = Path.GetFullPath(reconnectFailure);
            if (!File.Exists(reconnectFailure) || new FileInfo(reconnectFailure).Length == 0)
                throw new IOException("--host-reconnect-failure 需要一个已存在且非空的 JSON 文件。");
        }
        if (rollback && uninstall) throw new IOException("Choose either rollback or uninstall.");
        var manifest = Manifest();
        var profile = Profile(manifest, requested);
        VerifyBundle(manifest);
        bool temporary = uninstall || reconnectFailure != null;
        string python = null;
        if (!temporary) {
            python = DeployPython(root, false);
            var plan = InstallPreview(root, host, profile, manifest, python, rollback);
            if (!apply || !(bool)plan["ok"]) {
                output = Json.Serialize(plan);
                return (bool)plan["ok"] ? 0 : 2;
            }
            python = DeployPython(root, true);
        }
        using(var cleanup=temporary ? new TemporaryPython() : null) {
        var info = new ProcessStartInfo(cleanup==null ? python : cleanup.Executable,
            "-I -B -X utf8 " + Quote(App) + " --release " + Quote(Bundle) +
            " --project " + Quote(root) + (host == null ? "" : " --host " + host) + " --profile " + Quote(profile)
            + (apply ? " --apply" : "") + (rollback ? " --rollback" : "") + (uninstall ? " --uninstall" : "")
            + (reconnectFailure == null ? "" : " --host-reconnect-failure " + Quote(reconnectFailure)));
        info.WorkingDirectory = root; info.UseShellExecute = false; info.CreateNoWindow = true;
        info.RedirectStandardOutput = true; info.RedirectStandardError = true;
        info.StandardOutputEncoding = WireEncoding; info.StandardErrorEncoding = WireEncoding;
        using (var process = Process.Start(info)) {
            var stdout = process.StandardOutput.ReadToEndAsync(); var stderr = process.StandardError.ReadToEndAsync();
            process.WaitForExit(); Task.WaitAll(stdout, stderr); output = stdout.Result.Trim();
            if (output.Length == 0) throw new IOException("The private installer could not start. Check the bundle and project directory.");
            return process.ExitCode;
        }
        }
    }
    static int Dialog() {
        Application.EnableVisualStyles();
        string profile;
        try { profile=Profile(Manifest(),null); }
        catch (Exception ex) { MessageBox.Show(ex.Message,"AutoTEM"); return 2; }
        using (var ui = new Form {Text="AutoTEM 离线安装", ClientSize=new Size(570, 250), StartPosition=FormStartPosition.CenterScreen}) {
            var prompt = new Label {Text="项目文件夹   ·   " + profile, Left=16, Top=18, AutoSize=true};
            var path = new TextBox {Left=16, Top=42, Width=438};
            var browse = new Button {Text="浏览…", Left=462, Top=40, Width=86};
            browse.Click += (s,e) => { using (var pick=new FolderBrowserDialog())
                if (pick.ShowDialog(ui)==DialogResult.OK) path.Text=pick.SelectedPath; };
            var host = new ComboBox {Left=16, Top=83, Width=122, DropDownStyle=ComboBoxStyle.DropDownList};
            host.Items.AddRange(new object[]{"codex","workbuddy","openclaw"}); host.SelectedIndex=0;
            var go = new Button {Text="安装或修复", Left=150, Top=81, Width=120};
            var undo = new Button {Text="回滚上次安装", Left=281, Top=81, Width=132};
            var remove = new Button {Text="卸载项目环境", Left=424, Top=81, Width=124};
            var status = new TextBox {Left=16, Top=123, Width=532, Height=108, Multiline=true,
                ReadOnly=true, ScrollBars=ScrollBars.Vertical, Text="选择项目和宿主。安装将写入项目的 .autotem 及宿主配置目录。"};
            bool busy = false;
            ui.FormClosing += (s,e) => { if (busy) e.Cancel=true; };
            EventHandler apply = async (s,e) => {
                bool revert = Object.ReferenceEquals(s, undo);
                bool uninstall = Object.ReferenceEquals(s, remove);
                var project = path.Text; var selected = (string)host.SelectedItem;
                try {
                    ProjectPath(project);
                    busy=true; go.Enabled=undo.Enabled=remove.Enabled=browse.Enabled=host.Enabled=path.Enabled=false;
                    status.Text="正在读取项目并计算所需空间…";
                    var preview=await Task.Run(() => { string value; Run(project,selected,profile,false,revert,uninstall,out value); return value; });
                    var plan=Json.Deserialize<Dictionary<string,object>>(preview);
                    if (!(bool)plan["ok"]) { status.Text=Convert.ToString(plan["next_action"]); return; }
                    status.Text=PlanText(plan);
                    string confirm = uninstall ? "断开此项目的受管宿主并卸载未修改的环境文件？原始数据、运行记录与报告将保留。"
                        : revert ? "恢复上一次项目配置？" : "安装本项目的私有 Python、传统分析和宿主资源？";
                    if (MessageBox.Show(ui, confirm+"\r\n"+selected+"\r\n"+status.Text,
                            "AutoTEM",MessageBoxButtons.OKCancel)!=DialogResult.OK) return;
                    status.Text="正在校验发行包和运行环境…";
                    var report=await Task.Run(() => { string result; Run(project,selected,profile,true,revert,uninstall,out result); return result; });
                    var row=Json.Deserialize<Dictionary<string,object>>(report);
                    status.Text=((bool)row["ok"] ? "状态 " : "操作受阻 ")+row["state"]+"\r\n"+row["next_action"];
                    status.AppendText(ConflictText(row));
                    if (row.ContainsKey("configuration_references")) status.AppendText("\r\n"+Json.Serialize(row["configuration_references"]));
                    if (row.ContainsKey("active_pids")) status.AppendText("\r\nPID "+Json.Serialize(row["active_pids"]));
                    if (row.ContainsKey("retained")) status.AppendText("\r\n"+Json.Serialize(row["retained"]));
                } catch (Exception ex) { status.Text=ex.Message; }
                finally { busy=false; go.Enabled=undo.Enabled=remove.Enabled=browse.Enabled=host.Enabled=path.Enabled=true; }
            };
            go.Click+=apply; undo.Click+=apply; remove.Click+=apply;
            ui.Controls.AddRange(new Control[]{prompt,path,browse,host,go,undo,remove,status});
            Application.Run(ui);
        }
        return 0;
    }
    [STAThread] static int Main(string[] args) {
        if (args.Length==0) return Dialog();
        try {
            string project=null, host=null, profile=null, reconnectFailure=null;
            bool apply=false, rollback=false, uninstall=false;
            for (int i=0;i<args.Length;i++) {
                switch (args[i]) {
                    case "--project": project=args[++i]; break;
                    case "--host": host=args[++i]; break;
                    case "--profile": profile=args[++i]; break;
                    case "--apply": apply=true; break;
                    case "--rollback": rollback=true; break;
                    case "--uninstall": uninstall=true; break;
                    case "--host-reconnect-failure":
                        if (reconnectFailure != null || i+1 >= args.Length || String.IsNullOrWhiteSpace(args[i+1])
                                || args[i+1].StartsWith("--", StringComparison.Ordinal))
                            throw new IOException("--host-reconnect-failure 只接受一个 JSON 文件路径。");
                        reconnectFailure=args[++i]; break;
                    default: throw new IOException("Unknown installer argument.");
                }
            }
            if (String.IsNullOrWhiteSpace(project)) throw new IOException("Choose a project with --project.");
            string output; int code=Run(project,host,profile,apply,rollback,uninstall,out output,reconnectFailure); WriteReply(output); return code;
        } catch (Exception ex) {
            WriteReply(Json.Serialize(new {ok=false,state="blocked",error=new {code="SETUP_FAILED",message=ex.Message},next_action=ex.Message})); return 2;
        }
    }
}
