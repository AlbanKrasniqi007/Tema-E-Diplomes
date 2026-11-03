using System;
using System.Diagnostics;
using System.IO;
using System.Collections.Generic;
using System.Threading;

namespace Lc0Berlin {
    static class Program {
        static int Main(string[] args) {
            var exeDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(
                Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            Directory.SetCurrentDirectory(exeDir);

            var python = FindPython(exeDir);
            if (python == null) {
                Console.Error.WriteLine("Lc0 Berlin: could not find python.exe.");
                return 2;
            }

            var script = Path.Combine(exeDir, "scripts", "uci.py");
            var config = Path.Combine(exeDir, "configs", "default.yaml");
            if (!File.Exists(script)) {
                Console.Error.WriteLine("Lc0 Berlin: missing " + script);
                return 2;
            }

            var psi = new ProcessStartInfo();
            psi.FileName = python;
            psi.Arguments = "-u \"" + script + "\" --config \"" + config + "\"";
            psi.WorkingDirectory = exeDir;
            psi.UseShellExecute = false;
            psi.CreateNoWindow = true;
            psi.RedirectStandardInput = true;
            psi.RedirectStandardOutput = true;
            psi.RedirectStandardError = true;
            try {
                psi.EnvironmentVariables["PYTHONUNBUFFERED"] = "1";
                psi.EnvironmentVariables["PYTHONIOENCODING"] = "utf-8";
            } catch {
            }

            Process proc;
            try {
                proc = Process.Start(psi);
            } catch (Exception ex) {
                Console.Error.WriteLine("Lc0 Berlin: failed to start Python: " + ex.Message);
                return 3;
            }
            if (proc == null) return 3;

            var stdin = Console.OpenStandardInput();
            var stdout = Console.OpenStandardOutput();
            var stderr = Console.OpenStandardError();

            var tIn = new Thread(() => Pump(stdin, proc.StandardInput.BaseStream, true));
            var tOut = new Thread(() => Pump(proc.StandardOutput.BaseStream, stdout, false));
            var tErr = new Thread(() => Pump(proc.StandardError.BaseStream, stderr, false));
            tIn.IsBackground = true;
            tOut.IsBackground = true;
            tErr.IsBackground = true;
            tIn.Start();
            tOut.Start();
            tErr.Start();

            proc.WaitForExit();
            try { stdout.Flush(); } catch {}
            try { stderr.Flush(); } catch {}
            return proc.ExitCode;
        }

        static void Pump(Stream from, Stream to, bool closeDest) {
            var buf = new byte[4096];
            try {
                int n;
                while ((n = from.Read(buf, 0, buf.Length)) > 0) {
                    to.Write(buf, 0, n);
                    to.Flush();
                }
            } catch {
            } finally {
                if (closeDest) {
                    try { to.Close(); } catch {}
                }
            }
        }

        static string FindPython(string exeDir) {
            var sidecar = Path.Combine(exeDir, "python.path");
            if (File.Exists(sidecar)) {
                var hinted = File.ReadAllText(sidecar).Trim();
                if (hinted.Length > 0 && File.Exists(hinted)) return hinted;
            }

            var local = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            var candidates = new List<string>();
            if (!string.IsNullOrEmpty(local)) {
                var pyRoot = Path.Combine(local, "Programs", "Python");
                if (Directory.Exists(pyRoot)) {
                    foreach (var dir in Directory.GetDirectories(pyRoot, "Python3*")) {
                        candidates.Add(Path.Combine(dir, "python.exe"));
                    }
                }
            }
            foreach (var c in candidates) {
                if (File.Exists(c)) return c;
            }
            return FindOnPath("python.exe") ?? FindOnPath("python3.exe");
        }

        static string FindOnPath(string name) {
            var path = Environment.GetEnvironmentVariable("PATH") ?? "";
            foreach (var dir in path.Split(Path.PathSeparator)) {
                if (dir.Length == 0) continue;
                try {
                    var full = Path.Combine(dir, name);
                    if (File.Exists(full)) return full;
                } catch {
                }
            }
            return null;
        }
    }
}
