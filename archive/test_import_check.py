import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
print("Starting import test...")
try:
    from topomem.system import TopoMemSystem
    print("TopoMemSystem import: OK")
except Exception as e:
    print(f"Import error: {e}")
try:
    from topomem.memory import MemoryGraph
    print("MemoryGraph import: OK")
except Exception as e:
    print(f"MemoryGraph import error: {e}")
try:
    from topomem.embedding import EmbeddingManager
    print("EmbeddingManager import: OK")
except Exception as e:
    print(f"EmbeddingManager import error: {e}")
print("All imports done")
