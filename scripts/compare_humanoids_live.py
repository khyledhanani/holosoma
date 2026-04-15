#!/usr/bin/env python3
"""Visualize two humanoid XMLs in one MuJoCo scene.

Usage example:
  conda run -n hsretargeting python scripts/compare_humanoids_live.py \
    --xml-a models/mujoco_models/converted_model_test.xml \
    --xml-b models/mujoco_models/converted_model_test_regen.xml \
    --rotate-a-x-deg 90
"""

from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


DUAL_XML_REF_ATTRS: tuple[str, ...] = (
    "name",
    "class",
    "childclass",
    "joint",
    "joint1",
    "joint2",
    "geom",
    "geom1",
    "geom2",
    "body",
    "body1",
    "body2",
    "site",
    "site1",
    "site2",
    "tendon",
    "tendon1",
    "tendon2",
    "actuator",
    "objname",
    "mesh",
    "material",
    "texture",
    "hfield",
    "heightfield",
    "target",
    "slidersite",
    "cranksite",
)


def _prefix_attributes_in_subtree(elem: ET.Element, prefix: str, attrs: tuple[str, ...]) -> None:
    for node in elem.iter():
        for attr in attrs:
            if attr in node.attrib:
                node.set(attr, prefix + node.attrib[attr])


def _resolve_mesh_base_dir(single_xml_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    if meshdir:
        mesh_base = Path(meshdir)
        if not mesh_base.is_absolute():
            mesh_base = (single_xml_path.parent / mesh_base).resolve()
        return mesh_base
    assets_dir = single_xml_path.parent / "assets"
    meshes_dir = single_xml_path.parent / "meshes"
    if assets_dir.exists():
        return assets_dir.resolve()
    if meshes_dir.exists():
        return meshes_dir.resolve()
    return single_xml_path.parent.resolve()


def _absolutize_file_paths(root: ET.Element, single_xml_path: Path) -> None:
    mesh_base = _resolve_mesh_base_dir(single_xml_path, root)
    for node in root.iter():
        file_attr = node.attrib.get("file")
        if file_attr is None:
            continue
        file_path = Path(file_attr)
        if file_path.is_absolute():
            continue
        if node.tag in {"mesh", "texture", "hfield", "heightfield"}:
            candidate_mesh = (mesh_base / file_path).resolve()
            candidate_local = (single_xml_path.parent / file_path).resolve()
            abs_path = candidate_mesh if candidate_mesh.exists() else candidate_local
        else:
            abs_path = (single_xml_path.parent / file_path).resolve()
        node.set("file", str(abs_path))


def _load_and_prefix_root(single_xml_path: Path, prefix: str) -> ET.Element:
    tree = ET.parse(str(single_xml_path))
    root = tree.getroot()
    _absolutize_file_paths(root, single_xml_path)
    _prefix_attributes_in_subtree(root, prefix, DUAL_XML_REF_ATTRS)
    return root


def _append_section_children(dst_root: ET.Element, src_root: ET.Element, section_tag: str) -> None:
    src_sections = src_root.findall(section_tag)
    if not src_sections:
        return
    if section_tag == "default":
        dst_section = dst_root.find(section_tag)
        if dst_section is None:
            dst_section = ET.SubElement(dst_root, section_tag)
        for src_section in src_sections:
            if src_section.attrib:
                nested_default = ET.SubElement(dst_section, "default", attrib=deepcopy(src_section.attrib))
                for child in list(src_section):
                    nested_default.append(deepcopy(child))
            else:
                for child in list(src_section):
                    dst_section.append(deepcopy(child))
        return

    dst_section = dst_root.find(section_tag)
    if dst_section is None:
        dst_section = ET.SubElement(dst_root, section_tag)
    for src_section in src_sections:
        for child in list(src_section):
            dst_section.append(deepcopy(child))


def build_dual_scene_xml(
    xml_a: Path,
    xml_b: Path,
    out_path: Path,
    prefix_a: str = "A_",
    prefix_b: str = "B_",
) -> Path:
    root_a = _load_and_prefix_root(xml_a, prefix_a)
    root_b = _load_and_prefix_root(xml_b, prefix_b)

    merged_root = ET.Element("mujoco", attrib={"model": f"{prefix_a}{xml_a.stem}__{prefix_b}{xml_b.stem}"})

    for singleton_tag in ("compiler", "option", "size", "visual", "statistic"):
        singleton = root_a.find(singleton_tag)
        if singleton is None:
            singleton = root_b.find(singleton_tag)
        if singleton is not None:
            merged_root.append(deepcopy(singleton))

    for section_tag in ("default", "asset", "worldbody", "sensor", "actuator", "tendon", "equality", "contact"):
        _append_section_children(merged_root, root_a, section_tag)
        _append_section_children(merged_root, root_b, section_tag)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


def _set_root_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    root_joint_name: str,
    pos_xyz: np.ndarray,
    quat_wxyz: np.ndarray,
) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, root_joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found: {root_joint_name}")
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr : qadr + 3] = pos_xyz
    data.qpos[qadr + 3 : qadr + 7] = quat_wxyz


def _quat_wxyz_from_euler_xyz_deg(x_deg: float, y_deg: float, z_deg: float) -> np.ndarray:
    q_xyzw = R.from_euler("xyz", [x_deg, y_deg, z_deg], degrees=True).as_quat()
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two humanoid XMLs in one live MuJoCo scene.")
    parser.add_argument("--xml-a", type=Path, required=True, help="Path to first humanoid xml (e.g., converted_model_test.xml).")
    parser.add_argument("--xml-b", type=Path, required=True, help="Path to second humanoid xml (e.g., regenerated).")
    parser.add_argument(
        "--out-xml",
        type=Path,
        default=Path("DATA/humanoid_compare/dual_humanoid_compare.xml"),
        help="Output merged dual scene xml path.",
    )
    parser.add_argument("--prefix-a", type=str, default="A_")
    parser.add_argument("--prefix-b", type=str, default="B_")
    parser.add_argument("--rotate-a-x-deg", type=float, default=90.0, help="Root X-rotation for model A in degrees.")
    parser.add_argument("--rotate-b-x-deg", type=float, default=0.0, help="Root X-rotation for model B in degrees.")
    parser.add_argument("--offset-a", type=float, nargs=3, default=(-0.7, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--offset-b", type=float, nargs=3, default=(0.7, 0.0, 1.0), metavar=("X", "Y", "Z"))
    parser.add_argument("--simulate", action="store_true", help="Step physics instead of frozen pose.")
    parser.add_argument("--no-viewer", action="store_true", help="Build scene and print info without opening viewer.")
    args = parser.parse_args()

    xml_a = args.xml_a.resolve()
    xml_b = args.xml_b.resolve()
    out_xml = args.out_xml.resolve()

    if not xml_a.exists():
        raise FileNotFoundError(xml_a)
    if not xml_b.exists():
        raise FileNotFoundError(xml_b)

    build_dual_scene_xml(xml_a, xml_b, out_xml, args.prefix_a, args.prefix_b)

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)

    quat_a = _quat_wxyz_from_euler_xyz_deg(args.rotate_a_x_deg, 0.0, 0.0)
    quat_b = _quat_wxyz_from_euler_xyz_deg(args.rotate_b_x_deg, 0.0, 0.0)

    # Prefixing keeps original root free-joint names ("Pelvis") under each prefix.
    _set_root_pose(model, data, f"{args.prefix_a}Pelvis", np.array(args.offset_a, dtype=float), quat_a)
    _set_root_pose(model, data, f"{args.prefix_b}Pelvis", np.array(args.offset_b, dtype=float), quat_b)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    print(f"Merged scene: {out_xml}")
    print(f"A root: {args.prefix_a}Pelvis, rot_x={args.rotate_a_x_deg} deg, pos={tuple(args.offset_a)}")
    print(f"B root: {args.prefix_b}Pelvis, rot_x={args.rotate_b_x_deg} deg, pos={tuple(args.offset_b)}")

    if args.no_viewer:
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(max(model.opt.timestep, 1.0 / 120.0))


if __name__ == "__main__":
    main()
