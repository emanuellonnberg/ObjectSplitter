#!/usr/bin/env python3
# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.
"""
Bundle the plugin's non-Cura runtime dependencies into lib/.
Run from the plugin root: python scripts/bundle_deps.py

Only packages Cura does not ship are bundled (trimesh 4.x, networkx, rtree).
numpy, scipy and shapely come from Cura, so they are deliberately not bundled.
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

    # --no-deps: bundle only the packages listed in requirements-bundle.txt.
    # Without it, pip would pull numpy/scipy back in as transitive deps of
    # trimesh/scipy, re-introducing the Python-version-locked binaries we are
    # intentionally taking from Cura instead.
    cmd = [
        sys.executable, "-m", "pip", "install",
        "-r", req_file,
        "--target", lib_dir,
        "--no-deps",
        "--upgrade",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print("Done. lib/ is ready (numpy/scipy/shapely come from Cura).")

if __name__ == "__main__":
    main()
