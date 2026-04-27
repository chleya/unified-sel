import subprocess, sys, os

script = os.path.join(os.path.dirname(__file__), "topomem", "benchmarks", "isolation_test.py")
result = subprocess.run(
    [sys.executable, script],
    capture_output=True,
    cwd=os.path.dirname(__file__),
)
print(result.stdout.decode("utf-8", errors="replace"))
print(result.stderr.decode("utf-8", errors="replace"), file=sys.stderr)
sys.exit(result.returncode)
