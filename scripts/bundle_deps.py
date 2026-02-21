#!/usr/bin/env python3
# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.
"""
Install trimesh and its dependencies into the plugin's lib/ folder.
Run from the plugin root: python scripts/bundle_deps.py
"""
import os
import subprocess
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lib_dir = os.path.join(root, "lib")
    req_file = os.path.join(root, "requirements-bundle.txt")

    os.makedirs(lib_dir, exist_ok=True)
    if not os.path.isfile(req_file):
        print("requirements-bundle.txt not found", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", req_file,
        "--target", lib_dir,
        "--upgrade",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. lib/ is ready.")

if __name__ == "__main__":
    main()
