import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

print("Python path:", sys.path)
print("CWD:", __file__)

from topomem.system import TopoMemSystem
print("TopoMemSystem imported OK")
