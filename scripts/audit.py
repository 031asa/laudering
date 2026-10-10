"""Read-only full-data Spark audit. Writes reports, never silver data."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.audit import run

if __name__ == '__main__':
    run(ROOT)
