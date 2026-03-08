from pathlib import Path
import sys

# Data-Pipeline/ root — makes `common` and all pipeline scripts importable as top-level modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
