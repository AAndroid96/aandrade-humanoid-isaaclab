#!/usr/bin/env python3
"""Patch AAndrade-Humanoid-Lite URDF for Isaac Lab.

Fixes three export problems from onshape-to-robot:
  1. All links have mass=1e-09 (no material assigned in Onshape).
     -> Assigns ESTIMATED masses + solid-box inertias. Replace with real
        Onshape mass properties when you re-export with materials assigned.
  2. Joint limits say effort=10 N.m / velocity=10 rad/s.
     -> XL330-M288-T: stall torque 0.52 N.m @ 5V, no-load 104 rpm = 10.9 rad/s.
        Sets effort=0.52, velocity=10.5.
  3. Mesh paths use package://assets/... which Isaac Lab's URDF importer
     can't resolve without a ROS package.
     -> Rewrites to relative assets/... (keep the urdf next to the assets/ dir).

Usage:
    python3 patch_urdf.py robot.urdf robot_isaac.urdf
"""
import sys
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# ESTIMATED masses (kg). TODO: replace with Onshape mass properties.
# Basis: XL330-M288-T = 23 g each; 3D-printed brackets/horns ~10-20 g;
# pelvis carries OpenRB-150 + BNO055 + battery + frame.
# ---------------------------------------------------------------------------
MASS_ESTIMATES = {
    "dc15_a01_adaptor_2_1": 0.180,  # pelvis: electronics + battery + 2 hip servos' share
}
DEFAULT_LINK_MASS = 0.040          # servo + bracket per leg link
FOOT_LINKS = {"servobody_4", "servobody_8"}
FOOT_MASS = 0.030

# Approx bounding box per link type (m) for solid-box inertia
PELVIS_DIMS = (0.090, 0.050, 0.040)
LINK_DIMS = (0.024, 0.034, 0.046)   # ~XL330 body + bracket
FOOT_DIMS = (0.060, 0.040, 0.010)

# XL330-M288-T actuator limits
EFFORT_NM = 0.52
VELOCITY_RAD_S = 10.5


def box_inertia(m, dims):
    x, y, z = dims
    return (m / 12.0 * (y * y + z * z),
            m / 12.0 * (x * x + z * z),
            m / 12.0 * (x * x + y * y))


def main(src, dst):
    tree = ET.parse(src)
    root = tree.getroot()

    for link in root.findall("link"):
        name = link.get("name")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        if name in MASS_ESTIMATES:
            m, dims = MASS_ESTIMATES[name], PELVIS_DIMS
        elif name in FOOT_LINKS:
            m, dims = FOOT_MASS, FOOT_DIMS
        else:
            m, dims = DEFAULT_LINK_MASS, LINK_DIMS
        ixx, iyy, izz = box_inertia(m, dims)
        inertial.find("mass").set("value", f"{m:.4f}")
        inertia = inertial.find("inertia")
        inertia.set("ixx", f"{ixx:.3e}")
        inertia.set("iyy", f"{iyy:.3e}")
        inertia.set("izz", f"{izz:.3e}")
        for k in ("ixy", "ixz", "iyz"):
            inertia.set(k, "0")

    for joint in root.findall("joint"):
        lim = joint.find("limit")
        if lim is not None:
            lim.set("effort", str(EFFORT_NM))
            lim.set("velocity", str(VELOCITY_RAD_S))

    for mesh in root.iter("mesh"):
        fn = mesh.get("filename", "")
        if fn.startswith("package://"):
            mesh.set("filename", fn.replace("package://", ""))

    tree.write(dst, xml_declaration=True, encoding="utf-8")
    total = sum(float(l.find("inertial/mass").get("value"))
                for l in root.findall("link") if l.find("inertial/mass") is not None)
    print(f"Wrote {dst} | total mass: {total:.3f} kg")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
