#!/usr/bin/env python3
"""Generate a persistent SMPL-X humanoid MuJoCo XML from SMPLSim.

By default this exports the primitive-body proxy variant that is better suited
for optimization and collision handling. Mesh-collision export is still
available when needed for comparison.
"""

from __future__ import annotations

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
from lxml import etree
from PIL import Image
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot


ROOT = Path("/Users/khyledhanani/Documents/holosoma")
DEFAULT_MODEL_DIR = ROOT / "models" / "smplx"
DEFAULT_OUT_XML = ROOT / "models" / "mujoco_models" / "generated_smplx_humanoid.xml"
DEFAULT_MESH_ROOT = ROOT / "models" / "mujoco_models" / "assets" / "mujoco_models"
DEFAULT_COMPARE_XML_A = ROOT / "models" / "mujoco_models" / "converted_model_test.xml"


PRESET_BETAS: dict[str, np.ndarray] = {
    # 20-D SMPL-X betas. These are practical presets, not clinically exact body types.
    "neutral": np.zeros(20, dtype=np.float32),
    "fat": np.array([0.2, 2.4, 1.4, 0.6, 0.4, 0.2] + [0.0] * 14, dtype=np.float32),
    "skinny": np.array([0.1, -2.1, -1.1, -0.4, -0.2, -0.1] + [0.0] * 14, dtype=np.float32),
    "tall": np.array([2.0, -0.1, 0.2, 0.1] + [0.0] * 16, dtype=np.float32),
    "short": np.array([-2.0, 0.1, -0.2, -0.1] + [0.0] * 16, dtype=np.float32),
    "athletic": np.array([0.7, -0.6, 1.0, 0.4, 0.2, 0.2] + [0.0] * 14, dtype=np.float32),
    "stocky": np.array([-0.6, 1.8, 1.2, 0.4, 0.2, 0.1] + [0.0] * 14, dtype=np.float32),
}

PRESET_HEIGHT_SCALE: dict[str, float] = {
    "neutral": 1.0,
    "fat": 1.0,
    "skinny": 1.0,
    "tall": 1.1,
    "short": 0.9,
    "athletic": 1.02,
    "stocky": 0.95,
}


def _parse_betas_arg(raw: str | None, preset: str) -> np.ndarray:
    if raw is None:
        return PRESET_BETAS[preset].copy()
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != 20:
        raise ValueError(f"--betas must provide exactly 20 comma-separated floats, got {len(vals)}")
    return np.asarray(vals, dtype=np.float32)


def _gender_code(gender: str) -> list[int]:
    mapping = {"neutral": 0, "male": 1, "female": 2}
    if gender not in mapping:
        raise ValueError(f"Invalid gender: {gender}")
    return [mapping[gender]]


def _scale_vec_attr(elem: etree._Element, attr: str, scale: float) -> None:
    raw = elem.get(attr)
    if not raw:
        return
    vals = [float(x) for x in raw.split()]
    scaled = [x * scale for x in vals]
    elem.set(attr, " ".join(f"{x:.8g}" for x in scaled))


def _scale_xml(root: etree._Element, scale: float) -> None:
    if abs(scale - 1.0) < 1e-8:
        return
    for body in root.xpath(".//body[@pos]"):
        _scale_vec_attr(body, "pos", scale)
    for geom in root.xpath(".//geom[@pos]"):
        _scale_vec_attr(geom, "pos", scale)
    for geom in root.xpath(".//geom[@size]"):
        _scale_vec_attr(geom, "size", scale)
    for site in root.xpath(".//site[@pos]"):
        _scale_vec_attr(site, "pos", scale)
    for site in root.xpath(".//site[@size]"):
        _scale_vec_attr(site, "size", scale)


def _render_preview(xml_path: Path, out_png: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    if model.nq >= 3:
        data.qpos[2] = 1.0
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=480, width=640)
    renderer.update_scene(data)
    rgb = renderer.render()
    renderer.close()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_png)


def _uses_mesh_assets(xml_root: etree._Element) -> bool:
    return bool(xml_root.xpath(".//asset/mesh"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate persistent SMPL-X humanoid XML from SMPLSim.")
    parser.add_argument("--preset", choices=sorted(PRESET_BETAS.keys()), default="neutral")
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument(
        "--betas",
        type=str,
        default=None,
        help="Optional explicit 20-D comma-separated betas override.",
    )
    parser.add_argument("--height-scale", type=float, default=None, help="Optional global XML scale override.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out-xml", type=Path, default=DEFAULT_OUT_XML)
    parser.add_argument(
        "--out-mesh-dir",
        type=Path,
        default=None,
        help="Stable mesh directory. Default: models/mujoco_models/assets/mujoco_models/<out_stem>_geom",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing output files/dirs.")
    parser.add_argument(
        "--replace-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use SMPLSim's foot-replacement proxy geoms.",
    )
    parser.add_argument(
        "--big-ankle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use SMPLSim's larger ankle proxy setting.",
    )
    parser.add_argument(
        "--box-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to use SMPLSim's box/capsule body proxy mode when collision-mode=proxy.",
    )
    parser.add_argument("--render-preview", action="store_true", help="Render one preview PNG next to output XML.")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help=(
            "Launch compare_humanoids_live after generation using defaults: "
            "xml-a=converted_model_test.xml, xml-b=<out-xml>, rotate-a-x-deg=90."
        ),
    )
    parser.add_argument(
        "--visualize-no-viewer",
        action="store_true",
        help="When used with --visualize, build/validate the compare scene without opening the viewer window.",
    )
    parser.add_argument(
        "--visualize-xml-a",
        type=Path,
        default=DEFAULT_COMPARE_XML_A,
        help="Reference XML for compare_humanoids_live (defaults to converted_model_test.xml).",
    )
    parser.add_argument(
        "--visualize-rotate-a-x-deg",
        type=float,
        default=90.0,
        help="Root X rotation (degrees) for visualize xml-a.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=["proxy", "mesh"],
        default="proxy",
        help=(
            "Collision/body representation to export. "
            "'proxy' uses SMPLSim's primitive capsule/box body and is recommended for optimization. "
            "'mesh' keeps per-part mesh geoms."
        ),
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    out_xml = args.out_xml.resolve()
    out_mesh_dir = (
        args.out_mesh_dir.resolve()
        if args.out_mesh_dir is not None
        else (DEFAULT_MESH_ROOT / f"{out_xml.stem}_geom").resolve()
    )
    preview_png = out_xml.with_suffix(".png")

    if not model_dir.exists():
        raise FileNotFoundError(f"SMPL-X model dir not found: {model_dir}")

    if out_xml.exists() and not args.overwrite:
        raise FileExistsError(f"Output XML exists: {out_xml}. Pass --overwrite.")
    if args.collision_mode == "mesh" and out_mesh_dir.exists() and not args.overwrite:
        raise FileExistsError(f"Output mesh dir exists: {out_mesh_dir}. Pass --overwrite.")

    betas = _parse_betas_arg(args.betas, args.preset)
    height_scale = PRESET_HEIGHT_SCALE[args.preset] if args.height_scale is None else float(args.height_scale)

    robot_cfg = {
        "mesh": args.collision_mode == "mesh",
        "rel_joint_lm": True,
        "upright_start": True,
        "remove_toe": False,
        "real_weight": True,
        "real_weight_porpotion_capsules": True,
        "real_weight_porpotion_boxes": True,
        "replace_feet": bool(args.replace_feet),
        "masterfoot": False,
        "big_ankle": bool(args.big_ankle),
        "freeze_hand": False,
        "box_body": bool(args.box_body),
        "sim": "mujoco",
        "create_vel_sensors": False,
        "model": "smplx",
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
        "gender": args.gender,
    }

    smpl_robot = SMPL_Robot(robot_cfg, data_dir=str(model_dir))
    smpl_robot.load_from_skeleton(
        betas=torch.from_numpy(betas[None, :]),
        gender=_gender_code(args.gender),
    )

    xml_root = etree.fromstring(smpl_robot.export_xml_string())

    if _uses_mesh_assets(xml_root):
        if not smpl_robot.model_dirs:
            raise RuntimeError("SMPLSim exported mesh assets but did not record a mesh directory.")
        src_geom_dir = Path(smpl_robot.model_dirs[-1]) / "geom"
        if not src_geom_dir.exists():
            raise FileNotFoundError(f"Generated mesh dir missing: {src_geom_dir}")

        out_mesh_dir.parent.mkdir(parents=True, exist_ok=True)
        if out_mesh_dir.exists():
            shutil.rmtree(out_mesh_dir)
        shutil.copytree(src_geom_dir, out_mesh_dir)

        # Rewrite mesh paths to stable absolute locations.
        for mesh in xml_root.xpath(".//asset/mesh"):
            file_attr = mesh.get("file")
            if not file_attr:
                continue
            fname = Path(file_attr).name
            mesh.set("file", str((out_mesh_dir / fname).resolve()))
    elif out_mesh_dir.exists() and args.overwrite:
        shutil.rmtree(out_mesh_dir)

    _scale_xml(xml_root, height_scale)

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    etree.ElementTree(xml_root).write(str(out_xml), pretty_print=True)

    if args.render_preview:
        _render_preview(out_xml, preview_png)

    print(f"WROTE_XML={out_xml}")
    if _uses_mesh_assets(xml_root):
        print(f"WROTE_MESH_DIR={out_mesh_dir}")
    else:
        print("WROTE_MESH_DIR=<none>")
    print(f"PRESET={args.preset}")
    print(f"GENDER={args.gender}")
    print(f"COLLISION_MODE={args.collision_mode}")
    print(f"HEIGHT_SCALE={height_scale:.4f}")
    print(f"BETAS={','.join(f'{x:.4f}' for x in betas.tolist())}")
    if args.render_preview:
        print(f"WROTE_PREVIEW={preview_png}")

    if args.visualize:
        compare_script = (ROOT / "scripts" / "compare_humanoids_live.py").resolve()
        compare_xml_a = args.visualize_xml_a.resolve()
        runner = sys.executable
        if platform.system() == "Darwin" and not args.visualize_no_viewer:
            mjpython_bin = shutil.which("mjpython")
            if mjpython_bin is not None:
                runner = mjpython_bin
            else:
                # Avoid hard-failing on macOS when mjpython is unavailable.
                print("WARN: mjpython not found; forcing --visualize-no-viewer fallback on macOS.")
                args.visualize_no_viewer = True

        compare_cmd = [
            runner,
            str(compare_script),
            "--xml-a",
            str(compare_xml_a),
            "--xml-b",
            str(out_xml),
            "--rotate-a-x-deg",
            str(args.visualize_rotate_a_x_deg),
        ]
        if args.visualize_no_viewer:
            compare_cmd.append("--no-viewer")
        print(f"VISUALIZE_CMD={' '.join(compare_cmd)}")
        try:
            subprocess.run(compare_cmd, check=True)
        except subprocess.CalledProcessError:
            if args.visualize_no_viewer:
                raise
            # Fallback: still produce merged scene even if live viewer launch fails.
            fallback_cmd = list(compare_cmd) + ["--no-viewer"]
            fallback_cmd[0] = sys.executable
            print("WARN: live viewer launch failed; retrying in --no-viewer mode.")
            print(f"VISUALIZE_FALLBACK_CMD={' '.join(fallback_cmd)}")
            subprocess.run(fallback_cmd, check=True)


if __name__ == "__main__":
    main()
