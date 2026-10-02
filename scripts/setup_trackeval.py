#!/usr/bin/env python
"""Clone the official TrackEval (MIT) into third_party/TrackEval for `main.py evaluate --backend trackeval`."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dest = ROOT / "third_party" / "TrackEval"
if (dest / "trackeval").exists():
    print(f"TrackEval already present at {dest}")
    sys.exit(0)
dest.parent.mkdir(exist_ok=True)
try:
    subprocess.check_call(["git", "clone", "--depth", "1", "https://github.com/JonathonLuiten/TrackEval.git", str(dest)])
except Exception as e:  # git missing / offline
    print(f"Could not clone automatically ({e}).\nManual: download https://github.com/JonathonLuiten/TrackEval/archive/refs/heads/master.zip "
          f"and unzip it so that {dest}/trackeval exists.")
    sys.exit(2)
print("TrackEval ready. Use: python main.py evaluate --config configs/rca.yaml --backend trackeval")
