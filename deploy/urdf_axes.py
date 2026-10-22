#!/usr/bin/env python3
"""
Compute the world-frame axis and origin of every joint in a URDF, at the
zero configuration.

A joint's <origin rpy> is relative to its PARENT link, so the rpy values
printed in the raw URDF cannot be compared between joints. This composes
the full chain from the root link down, so every axis below is expressed
in one common frame.

Usage:
    python3 urdf_axes.py /path/to/robot.urdf
"""
import sys
import xml.etree.ElementTree as ET

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required:  pip3 install numpy")


def rpy_to_R(r, p, y):
    """URDF fixed-axis XYZ convention: R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def floats(text, default):
    if text is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in text.split()], dtype=float)


def main(path):
    root = ET.parse(path).getroot()

    joints = []
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        limit = j.find("limit")
        joints.append({
            "name":   j.get("name"),
            "type":   j.get("type"),
            "parent": j.find("parent").get("link"),
            "child":  j.find("child").get("link"),
            "xyz":    floats(origin.get("xyz") if origin is not None else None, [0, 0, 0]),
            "rpy":    floats(origin.get("rpy") if origin is not None else None, [0, 0, 0]),
            "axis":   floats(axis.get("xyz") if axis is not None else None, [1, 0, 0]),
            "lo":     limit.get("lower") if limit is not None else None,
            "hi":     limit.get("upper") if limit is not None else None,
        })

    if not joints:
        sys.exit("No <joint> elements found -- is that really a URDF?")

    children = {j["child"] for j in joints}
    all_links = {j["parent"] for j in joints} | children
    roots = sorted(all_links - children)

    print(f"links: {len(all_links)}   joints: {len(joints)}   root link(s): {roots}")
    print()

    # Walk the tree, composing rotations and translations from each root.
    by_parent = {}
    for j in joints:
        by_parent.setdefault(j["parent"], []).append(j)

    placed = {}          # link name -> (R_world_link, p_world_link)
    results = []         # joints in the order we reach them
    for r in roots:
        placed[r] = (np.eye(3), np.zeros(3))
        stack = [r]
        while stack:
            link = stack.pop()
            R_p, p_p = placed[link]
            for j in by_parent.get(link, []):
                R_c = R_p @ rpy_to_R(*j["rpy"])
                p_c = p_p + R_p @ j["xyz"]
                placed[j["child"]] = (R_c, p_c)
                j["world_axis"] = R_c @ j["axis"]
                j["world_pos"] = p_c
                results.append(j)
                stack.append(j["child"])

    unreached = [j["name"] for j in joints if "world_axis" not in j]
    if unreached:
        print(f"WARNING: not reachable from root, skipped: {unreached}\n")

    def fmt(v):
        # snap values very close to 0 / +-1 so the output is readable
        out = []
        for x in v:
            x = float(x)
            if abs(x) < 1e-9:
                x = 0.0
            out.append(f"{x:+.4f}")
        return "[" + " ".join(out) + "]"

    hdr = f"{'joint':<22}{'world axis (x,y,z)':<28}{'world origin (x,y,z)':<30}limits"
    print(hdr)
    print("-" * len(hdr))
    for j in results:
        if j["type"] not in ("revolute", "continuous", "prismatic"):
            continue
        lim = ""
        if j["lo"] is not None and j["hi"] is not None:
            lim = f"{float(j['lo']):+.4f} .. {float(j['hi']):+.4f}"
        print(f"{j['name']:<22}{fmt(j['world_axis']):<28}{fmt(j['world_pos']):<30}{lim}")

    print()
    print("Reading this:")
    print("  * Joints whose world axes are IDENTICAL rotate the same way in space.")
    print("  * Joints whose world axes are NEGATIVES of each other are mirrored:")
    print("    a positive command turns them in opposite directions in space.")
    print("  * Compare the two hip joints' world origins to see which sign of which")
    print("    column corresponds to the robot's left side.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
