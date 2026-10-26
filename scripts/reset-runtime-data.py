"""Run the packaged maintenance command from a source checkout."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'packages/core'))

from app.core.runtime_reset import main

if __name__ == '__main__':
    main()
