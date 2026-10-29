"""
GitHub Repository Setup & Push Utility
Automates git staging, initial commit, adding remote origin, and pushing to GitHub.

Usage:
    python setup_git.py
    python setup_git.py --repo https://github.com/your-username/your-repo.git
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_git_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return proc.returncode, proc.stdout.strip() + " " + proc.stderr.strip()
    except Exception as e:
        return 1, str(e)

def main():
    parser = argparse.ArgumentParser(description="Git and GitHub setup helper")
    parser.add_argument("--repo", type=str, default=None, help="Remote GitHub repository URL")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    print("=" * 60)
    print(" 🚀 GITHUB SETUP & DEPLOYMENT UTILITY")
    print(f" Working Directory: {project_dir}")
    print("=" * 60)

    # 1. Check if git is initialized
    git_dir = project_dir / ".git"
    if not git_dir.exists():
        print("Initializing git repository...")
        run_git_cmd(["git", "init"])

    # 2. Add files
    print("\n[1/4] Staging all files...")
    code, out = run_git_cmd(["git", "add", "."])
    if code != 0:
        print(f"Error staging files: {out}")
        return

    # 3. Check status
    code, status_out = run_git_cmd(["git", "status", "--short"])
    if status_out.strip():
        print("[2/4] Creating initial commit...")
        commit_msg = "feat: Bilingual Speech Emotion Recognition System (Hindi & English) with Dashboard & Timeline"
        code, out = run_git_cmd(["git", "commit", "-m", commit_msg])
        print(f"Commit created: {out.splitlines()[0] if out else 'Success'}")
    else:
        print("[2/4] Working tree clean, nothing new to commit.")

    # 4. Set branch to main or master
    run_git_cmd(["git", "branch", "-M", "main"])

    # 5. Remote URL
    repo_url = args.repo
    if not repo_url:
        print("\n" + "=" * 60)
        print(" [3/4] To push this repository to your GitHub account:")
        print(" 1. Go to https://github.com/new and create a new repository")
        print("    (e.g., named 'speech-emotion-recognition')")
        print(" 2. Run either:")
        print("    python setup_git.py --repo https://github.com/<your-username>/speech-emotion-recognition.git")
        print("    OR manually run:")
        print("    git remote add origin https://github.com/<your-username>/speech-emotion-recognition.git")
        print("    git push -u origin main")
        print("=" * 60)
        return

    print(f"\n[3/4] Configuring remote origin: {repo_url}")
    # Remove existing origin if any
    run_git_cmd(["git", "remote", "remove", "origin"])
    code, out = run_git_cmd(["git", "remote", "add", "origin", repo_url])
    if code != 0:
        print(f"Notice: {out}")

    print("\n[4/4] Pushing to GitHub (main branch)...")
    code, out = run_git_cmd(["git", "push", "-u", "origin", "main"])
    if code == 0:
        print("\n🎉 SUCCESS! Project has been pushed to your GitHub repository:")
        print(f"   {repo_url}")
    else:
        print(f"\nPush output: {out}")
        print("If authentication is requested, sign in via your browser or Personal Access Token (PAT).")

if __name__ == "__main__":
    main()
