# setup_training_scripts.py
import os, subprocess, sys, pathlib
root = pathlib.Path(__file__).parent.resolve()
dst = root / "diffusers"
if not dst.exists():
    subprocess.check_call(["git", "clone", "https://github.com/huggingface/diffusers.git", str(dst)])
print("OK:", dst)
