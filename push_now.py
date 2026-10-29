"""
Instant GitHub Push Script
Pushes the pre-committed project to https://github.com/durgesh352/speech-emotion-recognition.git
"""

import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_DIR = Path(__file__).resolve().parent

def main():
    print("Pushing to GitHub: https://github.com/durgesh352/speech-emotion-recognition.git ...")
    cmd = ["git", "push", "-u", "origin", "main"]
    res = subprocess.run(cmd, cwd=PROJECT_DIR, capture_output=True, text=True)
    
    if res.returncode == 0:
        print(" SUCCESS! Project successfully pushed to:")
        print("👉 https://github.com/durgesh352/speech-emotion-recognition")
    else:
        print("Push Result:")
        print(res.stdout)
        print(res.stderr)
        if "Repository not found" in res.stderr:
            print("\n⚠️ The repository does not exist on GitHub yet.")
            print("Please create it at: https://github.com/new?name=speech-emotion-recognition")

if __name__ == "__main__":
    main()
