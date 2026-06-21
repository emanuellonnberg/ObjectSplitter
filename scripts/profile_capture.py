# Copyright (c) 2024 Emanuel Lönnberg.
# This tool is released under the terms of the LGPLv3 or higher.

"""Profile a captured ObjectSplitter operation offline.

Usage:
    python scripts/profile_capture.py <capture_dir> [--sort cumulative] [--lines 40]

Loads a capture directory (input_mesh.stl + params.json) produced by the
plugin's Debug Capture, replays it under cProfile, and prints the hottest
functions. Works for any captured mode, including "path" (multi-point).

Note: this runs in whatever Python/numpy/scipy the shell uses, which is not
necessarily Cura's runtime. Absolute timings differ from Cura; the function
ranking is what matters for finding hotspots.
"""

import argparse
import cProfile
import os
import pstats
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
_LIB = os.path.join(_ROOT, "lib")
if os.path.isdir(_LIB):
    sys.path.insert(0, _LIB)


def main():
    parser = argparse.ArgumentParser(description="Profile a captured cut operation.")
    parser.add_argument("capture_dir", help="Path to the capture directory")
    parser.add_argument("--sort", default="cumulative",
                        choices=["cumulative", "tottime", "ncalls"],
                        help="pstats sort key (default: cumulative)")
    parser.add_argument("--lines", type=int, default=40,
                        help="Number of rows to print (default: 40)")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Replay N times (warm caches / average) (default: 1)")
    args = parser.parse_args()

    from core.debug_capture import load_captured_operation, replay_operation

    mesh, params = load_captured_operation(args.capture_dir)
    print("Capture:        %s" % args.capture_dir)
    print("Mode:           %s" % params.cut_mode)
    print("Mesh:           %d verts, %d faces" % (len(mesh.vertices), len(mesh.faces)))
    if params.cut_mode == "path" and params.waypoints:
        print("Waypoints:      %d" % len(params.waypoints))
    print("-" * 60)

    # Wall-clock first (uncontaminated by profiler overhead).
    t0 = time.perf_counter()
    for _ in range(args.repeat):
        replay_operation(args.capture_dir)
    wall = (time.perf_counter() - t0) / max(args.repeat, 1)
    print("Wall-clock replay: %.1f ms (avg of %d)" % (wall * 1000.0, args.repeat))
    print("-" * 60)

    # Profiled run.
    profiler = cProfile.Profile()
    profiler.enable()
    replay_operation(args.capture_dir)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.sort_stats(args.sort)
    stats.print_stats(args.lines)


if __name__ == "__main__":
    main()
