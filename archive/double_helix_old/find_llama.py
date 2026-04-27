import os, subprocess, winreg

k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
mp, _ = winreg.QueryValueEx(k, "Path")
k2 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
up, _ = winreg.QueryValueEx(k2, "Path")
os.environ["PATH"] = mp + ";" + up

for p in mp.split(";"):
    if "llama" in p.lower():
        print(f"Found in machine PATH: {p}")

for p in up.split(";"):
    if "llama" in p.lower():
        print(f"Found in user PATH: {p}")

try:
    r = subprocess.run(["llama-cli", "--version"], capture_output=True, timeout=10, text=True)
    print(f"llama-cli version: {r.stdout.strip()}")
    print("OK: llama-cli is available")
except FileNotFoundError:
    print("ERROR: llama-cli not found in PATH")
    print("Searching common locations...")
    for root in [r"C:\Program Files", r"C:\Program Files (x86)", os.path.expanduser("~")]:
        for dirpath, dirnames, filenames in os.walk(root):
            if "llama-cli" in filenames or "llama-cli.exe" in filenames:
                print(f"  Found: {dirpath}")
                break
            if "llama" in dirpath.lower():
                print(f"  Possible dir: {dirpath}")
            if dirpath.count(os.sep) - root.count(os.sep) > 3:
                dirnames.clear()
except Exception as e:
    print(f"ERROR: {e}")
