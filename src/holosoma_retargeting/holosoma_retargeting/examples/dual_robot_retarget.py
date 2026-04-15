"""
Dual-robot retargeting entrypoint (Part 1 MVP).

Part 1 focuses on:
1) Loading Inter-X dual-human data (.npz).
2) Preprocessing each person for robot retargeting.
3) Optionally running independent retargeting for person A and B.

This script intentionally keeps A/B retargeting independent (no robot-robot constraints yet).
"""

from __future__ import annotations

import logging
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
from typing import Literal

import mujoco
import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.robot_retarget import (
    build_retargeter_kwargs_from_config,
    create_task_constants,
    setup_object_data,
)
from holosoma_retargeting.src.interaction_mesh_retargeter import (
    InteractionMeshRetargeter,
)
from holosoma_retargeting.src.dual_interaction_mesh_retargeter import (
    DualInteractionMeshRetargeter,
)
from holosoma_retargeting.src.utils import (
    extract_foot_sticking_sequence_velocity,
    preprocess_motion_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


HUMAN_MODEL_ROBOT_TYPE = "smplx_humanoid"


# Right-handed y-up -> z-up coordinate transform, matching dual_joint_renderer.py.
R_YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _convert_y_up_to_z_up_points(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    converted = (R_YUP_TO_ZUP @ flat.T).T
    return converted.reshape(points.shape).astype(np.float32, copy=False)


def _attach_qpos_coordinate_metadata(npz_path: Path, coordinate_frame: str = "z_up") -> None:
    data = np.load(str(npz_path), allow_pickle=True)
    payload = {key: data[key] for key in data.files}
    payload["qpos_coordinate_frame"] = np.asarray(coordinate_frame)
    np.savez(npz_path, **payload)


def _build_dual_dense_joint_mapping(
    *,
    constants: SimpleNamespace,
    robot_type: str,
) -> dict[str, str | tuple[str, np.ndarray]]:
    mapping = deepcopy(dict(constants.JOINTS_MAPPING))

    if robot_type == "g1":
        mapping.update(
            {
                "Spine1": ("pelvis_contour_link", np.array([0.0, 0.0, 0.06], dtype=float)),
                "Spine2": ("torso_link", np.array([0.0, 0.0, 0.02], dtype=float)),
                "Spine3": ("torso_link", np.array([0.01, 0.0, 0.14], dtype=float)),
                "Neck": ("head_link", np.array([0.0, 0.0, -0.04], dtype=float)),
                "Head": ("head_link", np.array([0.03, 0.0, 0.12], dtype=float)),
                "L_Collar": ("left_shoulder_pitch_link", np.array([0.0, 0.02, 0.02], dtype=float)),
                "R_Collar": ("right_shoulder_pitch_link", np.array([0.0, -0.02, 0.02], dtype=float)),
                "L_Wrist": ("left_rubber_hand_link", np.array([0.03, 0.0, 0.0], dtype=float)),
                "R_Wrist": ("right_rubber_hand_link", np.array([0.03, 0.0, 0.0], dtype=float)),
                "L_Foot": ("left_ankle_roll_sphere_5_link", np.array([0.02, 0.0, 0.0], dtype=float)),
                "R_Foot": ("right_ankle_roll_sphere_5_link", np.array([0.02, 0.0, 0.0], dtype=float)),
            }
        )
        return mapping

    if robot_type == "t1":
        mapping.update(
            {
                "Spine1": ("Waist", np.array([0.0, 0.0, 0.06], dtype=float)),
                "Spine2": ("Trunk", np.array([0.03, 0.0, 0.03], dtype=float)),
                "Spine3": ("Trunk", np.array([0.05, 0.0, 0.16], dtype=float)),
                "Neck": ("H1", np.array([0.0, 0.0, 0.02], dtype=float)),
                "Head": ("H2", np.array([0.02, 0.0, 0.08], dtype=float)),
                "L_Collar": ("AL1", np.array([0.0, 0.02, 0.02], dtype=float)),
                "R_Collar": ("AR1", np.array([0.0, -0.02, 0.02], dtype=float)),
                "L_Wrist": ("left_hand_sphere_link", np.array([0.03, 0.0, 0.0], dtype=float)),
                "R_Wrist": ("right_hand_sphere_link", np.array([0.03, 0.0, 0.0], dtype=float)),
                "L_Foot": ("left_foot_sphere_5_link", np.array([0.03, 0.0, 0.0], dtype=float)),
                "R_Foot": ("right_foot_sphere_5_link", np.array([0.03, 0.0, 0.0], dtype=float)),
            }
        )
        return mapping

    return mapping


def _make_dual_dense_task_constants(constants: SimpleNamespace, robot_type: str) -> SimpleNamespace:
    dense_constants = deepcopy(constants)
    dense_constants.JOINTS_MAPPING = _build_dual_dense_joint_mapping(constants=constants, robot_type=robot_type)
    return dense_constants


@dataclass
class DualRetargetConfig:
    """Config for dual-robot retargeting Part 1."""

    data_dir: Path = Path("DATA/interx_dual_inputs")
    """Input Inter-X directory containing dual-human .npz files."""

    sequence_id: str | None = None
    """Optional sequence id (file stem). If None, the first sorted file is used."""

    output_dir: Path = Path("DATA/interx_dual_retarget_part1")
    """Directory for part-1 outputs."""

    robot: str = "g1"
    """Shared fallback robot type (used for both A/B if agent-specific types are not provided)."""

    robot_a: str | None = None
    """Optional robot type for agent A (overrides `robot`)."""

    robot_b: str | None = None
    """Optional robot type for agent B (overrides `robot`)."""

    target_mode: Literal["robot_to_robot", "human_to_robot", "human_to_human"] = "robot_to_robot"
    """High-level dual mode:
    - robot_to_robot: both sides use robot types
    - human_to_robot: one side uses an articulated human model
    - human_to_human: both sides use articulated human models
    """

    robot_side: Literal["A", "B"] = "A"
    """For human_to_robot mode: which side should remain a robot target."""

    human_model_robot_type: str = HUMAN_MODEL_ROBOT_TYPE
    """Robot-type token used for articulated human constants/mapping."""

    human_model_xml: Path = Path("models/mujoco_models/converted_model_test.xml")
    """MuJoCo XML for the articulated human model used in optimization."""

    human_model_xml_a: Path | None = None
    """Optional per-agent override for human model XML on side A."""

    human_model_xml_b: Path | None = None
    """Optional per-agent override for human model XML on side B."""

    human_model_height: float = 1.78
    """Nominal height for articulated human scaling in preprocessing."""

    augment_human_with_hand_landmarks: bool = True
    """If True, append Inter-X hand landmark tracks to human-target source joints when available."""

    data_format: str = "smplx"
    """Motion data format used for joint naming/mapping."""

    input_y_up_to_z_up: bool = True
    """Apply right-handed y-up -> z-up conversion to dual-human joints before retargeting."""

    run_retarget: bool = False
    """If True, run independent A/B retargeting and save qpos outputs."""

    coupled_dual: bool = False
    """If True, run coupled dual optimization via DualInteractionMeshRetargeter."""

    coupled_warm_start_nominal: bool = True
    """If True, run independent A/B retarget first and use outputs for coupled warm-start + nominal tracking."""

    dual_scene_xml: Path | None = None
    """Optional prebuilt dual scene XML path. If None and coupled_dual=True, auto-generate one."""

    dual_prefix_a: str = "A_"
    """Prefix for robot A body/geom names in dual scene."""

    dual_prefix_b: str = "B_"
    """Prefix for robot B body/geom names in dual scene."""

    max_frames: int | None = None
    """Optional frame cap for faster debugging."""

    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplx", robot_type="g1")
    )
    task_config: TaskConfig = field(default_factory=TaskConfig)
    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)


def _pick_sequence_file(data_dir: Path, sequence_id: str | None) -> Path:
    files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")

    if sequence_id is None:
        return files[0]

    f = data_dir / f"{sequence_id}.npz"
    if not f.exists():
        raise FileNotFoundError(f"Sequence not found: {f}")
    return f


def _find_dual_joint_keys(npz_obj: np.lib.npyio.NpzFile) -> tuple[str, str]:
    key_pairs = [
        ("human_joints_A", "human_joints_B"),
        ("global_joint_positions_A", "global_joint_positions_B"),
    ]
    for key_a, key_b in key_pairs:
        if key_a in npz_obj and key_b in npz_obj:
            return key_a, key_b
    raise KeyError(
        "Could not find dual joint keys in npz. Expected one of: "
        "[human_joints_A/human_joints_B] or [global_joint_positions_A/global_joint_positions_B]"
    )


def _load_dual_sequence(npz_path: Path) -> tuple[str, int, np.ndarray, np.ndarray, float, float]:
    data = np.load(npz_path, allow_pickle=True)
    key_a, key_b = _find_dual_joint_keys(data)

    seq = str(data["sequence_id"]) if "sequence_id" in data else npz_path.stem
    fps = int(data["fps"]) if "fps" in data else 30
    joints_a = np.asarray(data[key_a], dtype=np.float32)
    joints_b = np.asarray(data[key_b], dtype=np.float32)
    height_a = float(data["height_A"]) if "height_A" in data else float(data.get("height", 1.78))
    height_b = float(data["height_B"]) if "height_B" in data else float(data.get("height", 1.78))

    if joints_a.ndim != 3 or joints_a.shape[-1] != 3:
        raise ValueError(f"Expected joints A shape (T, J, 3), got {joints_a.shape}")
    if joints_b.ndim != 3 or joints_b.shape[-1] != 3:
        raise ValueError(f"Expected joints B shape (T, J, 3), got {joints_b.shape}")
    if joints_a.shape[0] != joints_b.shape[0]:
        raise ValueError(f"A/B frame mismatch: {joints_a.shape[0]} vs {joints_b.shape[0]}")

    return seq, fps, joints_a, joints_b, height_a, height_b


def _identity_object_poses(num_frames: int) -> np.ndarray:
    # MuJoCo order: [x, y, z, qw, qx, qy, qz]
    object_poses = np.zeros((num_frames, 7), dtype=np.float32)
    object_poses[:, 3] = 1.0
    return object_poses


def _build_q_init(human_joints: np.ndarray, robot_dof: int) -> np.ndarray:
    q_init = np.zeros(7 + robot_dof, dtype=np.float32)
    q_init[:3] = human_joints[0, 0, :3]  # pelvis/root position
    q_init[3] = 1.0  # identity orientation [qw, qx, qy, qz]
    return q_init


def _load_hand_landmark_tracks(
    sequence_dir: Path,
    *,
    num_frames: int,
) -> tuple[np.ndarray | None, list[str], np.ndarray | None, list[str]]:
    p1 = sequence_dir / "P1.npz"
    p2 = sequence_dir / "P2.npz"
    if not p1.exists() or not p2.exists():
        return None, [], None, []

    def _extract(path: Path) -> tuple[np.ndarray | None, list[str]]:
        data = np.load(path, allow_pickle=True)
        required = {"left_hand_landmarks_world", "right_hand_landmarks_world", "hand_landmark_names"}
        if not required.issubset(set(data.files)):
            return None, []
        landmark_names = [str(x) for x in np.asarray(data["hand_landmark_names"]).tolist()]
        left = np.asarray(data["left_hand_landmarks_world"], dtype=np.float32)[:num_frames]
        right = np.asarray(data["right_hand_landmarks_world"], dtype=np.float32)[:num_frames]
        keep_idx = [i for i, name in enumerate(landmark_names) if name.strip().lower() != "wrist"]
        if not keep_idx:
            return None, []
        left = left[:, keep_idx, :]
        right = right[:, keep_idx, :]
        names_left = [f"L_hand_{landmark_names[i].strip().lower()}" for i in keep_idx]
        names_right = [f"R_hand_{landmark_names[i].strip().lower()}" for i in keep_idx]
        points = np.concatenate([left, right], axis=1)
        return points.astype(np.float32), names_left + names_right

    a_points, a_names = _extract(p1)
    b_points, b_names = _extract(p2)
    return a_points, a_names, b_points, b_names


def _human_hand_landmark_mapping() -> dict[str, str]:
    return {
        "L_hand_index_mcp": "L_Index1",
        "L_hand_middle_mcp": "L_Middle1",
        "L_hand_pinky_mcp": "L_Pinky1",
        "L_hand_index_tip": "L_Index3",
        "L_hand_thumb_tip": "L_Thumb3",
        "R_hand_index_mcp": "R_Index1",
        "R_hand_middle_mcp": "R_Middle1",
        "R_hand_pinky_mcp": "R_Pinky1",
        "R_hand_index_tip": "R_Index3",
        "R_hand_thumb_tip": "R_Thumb3",
    }


def _run_single_agent_retarget(
    human_joints: np.ndarray,
    smpl_scale: float,
    toe_names: list[str],
    constants: SimpleNamespace,
    retargeter_cfg: RetargeterConfig,
    object_local_pts: np.ndarray,
    object_local_pts_demo: np.ndarray,
    out_path: Path,
) -> np.ndarray:
    retargeter_kwargs = build_retargeter_kwargs_from_config(
        retargeter_config=retargeter_cfg,
        constants=constants,
        object_urdf_path=None,
        task_type="robot_only",
    )
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

    human_processed = preprocess_motion_data(
        human_joints=human_joints.copy(),
        retargeter=retargeter,
        foot_names=toe_names,
        scale=smpl_scale,
    )
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
        human_processed,
        retargeter.demo_joints,
        toe_names,
    )
    return _run_single_agent_retarget_processed(
        human_processed=human_processed,
        foot_sticking_sequences=foot_sticking_sequences,
        constants=constants,
        retargeter_cfg=retargeter_cfg,
        object_local_pts=object_local_pts,
        object_local_pts_demo=object_local_pts_demo,
        out_path=out_path,
    )


def _run_single_agent_retarget_processed(
    human_processed: np.ndarray,
    foot_sticking_sequences: list[dict[str, bool]],
    constants: SimpleNamespace,
    retargeter_cfg: RetargeterConfig,
    object_local_pts: np.ndarray,
    object_local_pts_demo: np.ndarray,
    out_path: Path,
) -> np.ndarray:
    retargeter_kwargs = build_retargeter_kwargs_from_config(
        retargeter_config=retargeter_cfg,
        constants=constants,
        object_urdf_path=None,
        task_type="robot_only",
    )
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

    object_poses = _identity_object_poses(human_processed.shape[0])
    q_init = _build_q_init(human_processed, constants.ROBOT_DOF)

    qpos, _, _, _ = retargeter.retarget_motion(
        human_joint_motions=human_processed,
        object_poses=object_poses,
        object_poses_augmented=object_poses,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=None,
        original=True,
        dest_res_path=str(out_path),
    )
    _attach_qpos_coordinate_metadata(out_path)
    return np.asarray(qpos, dtype=np.float32)


def _prefix_attributes_in_subtree(elem: ET.Element, prefix: str, attrs: tuple[str, ...]) -> None:
    for node in elem.iter():
        for attr in attrs:
            if attr in node.attrib:
                node.set(attr, prefix + node.attrib[attr])


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
    "target",
    "slidersite",
    "cranksite",
)


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
            # Some XMLs already encode "assets/..." in file paths even when a
            # meshdir/assets base is present. Prefer whichever candidate exists.
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
        merge_scope_idx = 0

        def _new_merge_scope() -> ET.Element:
            nonlocal merge_scope_idx
            scope = ET.SubElement(dst_section, "default", attrib={"class": f"merged_scope_{merge_scope_idx}"})
            merge_scope_idx += 1
            return scope

        for src_section in src_sections:
            # Preserve named defaults as nested scopes.
            if src_section.attrib:
                nested_default = ET.SubElement(dst_section, "default", attrib=deepcopy(src_section.attrib))
                for child in list(src_section):
                    nested_default.append(deepcopy(child))
                continue

            merge_scope: ET.Element | None = None
            for child in list(src_section):
                child_copy = deepcopy(child)
                if child_copy.tag == "default":
                    dst_section.append(child_copy)
                    continue

                # MuJoCo allows only one direct singleton tag (e.g. <geom>)
                # per <default> scope. Collisions are moved into a classed
                # nested scope.
                if dst_section.find(child_copy.tag) is None:
                    dst_section.append(child_copy)
                else:
                    if merge_scope is None:
                        merge_scope = _new_merge_scope()
                    merge_scope.append(child_copy)
        return

    dst_section = dst_root.find(section_tag)
    if dst_section is None:
        dst_section = ET.SubElement(dst_root, section_tag)
    for src_section in src_sections:
        for child in list(src_section):
            dst_section.append(deepcopy(child))


def _build_dual_scene_xml_from_pair(
    robot_xml_a: Path,
    robot_xml_b: Path,
    out_path: Path,
    prefix_a: str,
    prefix_b: str,
) -> Path:
    root_a = _load_and_prefix_root(robot_xml_a, prefix_a)
    root_b = _load_and_prefix_root(robot_xml_b, prefix_b)

    merged_root = ET.Element("mujoco", attrib={"model": f"{prefix_a}{robot_xml_a.stem}__{prefix_b}{robot_xml_b.stem}"})

    # Keep singleton sections from A first, then B as fallback.
    for singleton_tag in ("compiler", "option", "size", "visual", "statistic"):
        singleton = root_a.find(singleton_tag)
        if singleton is None:
            singleton = root_b.find(singleton_tag)
        if singleton is not None:
            merged_root.append(deepcopy(singleton))

    for section_tag in ("default", "asset", "worldbody", "sensor", "actuator", "tendon", "equality", "contact"):
        _append_section_children(merged_root, root_a, section_tag)
        _append_section_children(merged_root, root_b, section_tag)

    worldbody = merged_root.find("worldbody")
    if worldbody is None:
        raise ValueError("Merged dual scene has no worldbody section.")

    def _is_free_root_body(body: ET.Element) -> bool:
        return body.find("freejoint") is not None or body.find("joint[@type='free']") is not None

    freejoint_roots = [child for child in list(worldbody) if child.tag == "body" and _is_free_root_body(child)]
    if len(freejoint_roots) < 2:
        raise ValueError(
            f"Expected at least two root freejoint bodies in merged dual scene, got {len(freejoint_roots)}."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(merged_root).write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


def _preprocess_dual_humans_for_coupled(
    human_a: np.ndarray,
    human_b: np.ndarray,
    scale_a: float,
    scale_b: float,
    toe_names_a: list[str],
    toe_names_b: list[str],
    demo_joints_a: list[str],
    demo_joints_b: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, bool]], list[dict[str, bool]]]:
    demo_proxy_a = SimpleNamespace(demo_joints=demo_joints_a)
    demo_proxy_b = SimpleNamespace(demo_joints=demo_joints_b)
    human_a_processed = preprocess_motion_data(
        human_joints=human_a.copy(),
        retargeter=demo_proxy_a,
        foot_names=toe_names_a,
        scale=scale_a,
    )
    human_b_processed = preprocess_motion_data(
        human_joints=human_b.copy(),
        retargeter=demo_proxy_b,
        foot_names=toe_names_b,
        scale=scale_b,
    )
    foot_a = extract_foot_sticking_sequence_velocity(human_a_processed, demo_joints_a, toe_names_a)
    foot_b = extract_foot_sticking_sequence_velocity(human_b_processed, demo_joints_b, toe_names_b)
    return human_a_processed, human_b_processed, foot_a, foot_b


def _resolve_coupled_source_scales(
    *,
    scale_a: float,
    scale_b: float,
    target_mode: str,
) -> tuple[float, float]:
    if target_mode != "human_to_robot":
        return scale_a, scale_b

    shared_scale = 0.5 * (float(scale_a) + float(scale_b))
    return shared_scale, shared_scale


def _resolve_asset_paths(constants: SimpleNamespace, asset_root: Path) -> None:
    project_root = Path(__file__).resolve().parents[4]

    def _to_abs(path_value: str | None) -> str | None:
        if path_value is None:
            return None
        p = Path(path_value)
        if p.is_absolute():
            return str(p)
        candidate_asset = (asset_root / p).resolve()
        if candidate_asset.exists():
            return str(candidate_asset)
        candidate_project = (project_root / p).resolve()
        if candidate_project.exists():
            return str(candidate_project)
        return str(candidate_asset)

    constants.ROBOT_URDF_FILE = _to_abs(getattr(constants, "ROBOT_URDF_FILE", None))
    constants.OBJECT_URDF_FILE = _to_abs(getattr(constants, "OBJECT_URDF_FILE", None))
    constants.OBJECT_MESH_FILE = _to_abs(getattr(constants, "OBJECT_MESH_FILE", None))
    if hasattr(constants, "SCENE_XML_FILE"):
        constants.SCENE_XML_FILE = _to_abs(getattr(constants, "SCENE_XML_FILE", None))


def _resolve_agent_robot_types(cfg: object) -> tuple[str, str]:
    fallback_robot = str(getattr(cfg, "robot", "g1"))
    robot_type_a_cfg = getattr(cfg, "robot_a", None)
    robot_type_b_cfg = getattr(cfg, "robot_b", None)
    target_mode = str(getattr(cfg, "target_mode", "robot_to_robot")).strip().lower()
    human_type = str(getattr(cfg, "human_model_robot_type", HUMAN_MODEL_ROBOT_TYPE))

    if target_mode == "human_to_human":
        return human_type, human_type

    if target_mode == "human_to_robot":
        robot_side = str(getattr(cfg, "robot_side", "A")).strip().upper()
        if robot_side not in {"A", "B"}:
            raise ValueError(f"robot_side must be 'A' or 'B', got {robot_side!r}")
        if robot_side == "A":
            return str(robot_type_a_cfg or fallback_robot), human_type
        return human_type, str(robot_type_b_cfg or fallback_robot)

    # Default: explicit per-agent overrides if provided, else shared robot.
    return str(robot_type_a_cfg or fallback_robot), str(robot_type_b_cfg or fallback_robot)


def _resolve_human_model_xml(cfg: object, agent: str | None = None) -> Path:
    raw = None
    if agent == "A":
        raw = getattr(cfg, "human_model_xml_a", None)
    elif agent == "B":
        raw = getattr(cfg, "human_model_xml_b", None)
    if raw is None:
        raw = getattr(cfg, "human_model_xml", Path("models/mujoco_models/converted_model_test.xml"))
    p = Path(raw)
    if p.is_absolute():
        return p
    project_root = Path(__file__).resolve().parents[4]
    candidate_project = (project_root / p).resolve()
    if candidate_project.exists():
        return candidate_project
    return (Path(__file__).resolve().parents[1] / p).resolve()


def _infer_robot_dof_from_model_xml(model_xml: Path) -> int:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    free_joint_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joint_ids:
        raise ValueError(f"Model has no free joint root: {model_xml}")
    free_qadr = sorted(int(model.jnt_qposadr[j]) for j in free_joint_ids)
    q_start = free_qadr[0]
    q_end = free_qadr[1] if len(free_qadr) > 1 else int(model.nq)
    width = q_end - q_start
    if width < 7:
        raise ValueError(f"Expected free-joint block width >= 7 for {model_xml}, got {width}")
    return int(width - 7)


def _apply_human_model_overrides(robot_config: RobotConfig, cfg: object, agent: str | None = None) -> RobotConfig:
    human_type = str(getattr(cfg, "human_model_robot_type", HUMAN_MODEL_ROBOT_TYPE))
    if robot_config.robot_type != human_type:
        return robot_config

    human_xml = _resolve_human_model_xml(cfg, agent=agent)
    if not human_xml.exists():
        raise FileNotFoundError(f"Human model XML does not exist: {human_xml}")
    human_dof = _infer_robot_dof_from_model_xml(human_xml)
    human_height = float(getattr(cfg, "human_model_height", 1.78))

    return replace(
        robot_config,
        robot_urdf_file=str(human_xml),
        robot_dof=human_dof,
        robot_height=human_height,
        foot_sticking_links=["L_Toe", "R_Toe"],
    )


def _resolve_robot_and_motion_config(
    cfg: object,
) -> tuple[RobotConfig, RobotConfig, MotionDataConfig, MotionDataConfig]:
    """Resolve per-agent robot/motion configs from shared settings + optional A/B overrides."""
    robot_type_a, robot_type_b = _resolve_agent_robot_types(cfg)
    data_format = str(getattr(cfg, "data_format", "smplx"))
    robot_config_seed = getattr(cfg, "robot_config")
    motion_data_config_seed = getattr(cfg, "motion_data_config")

    robot_config_a = robot_config_seed
    if robot_config_a.robot_type != robot_type_a:
        robot_config_a = replace(robot_config_a, robot_type=robot_type_a)
    robot_config_a = _apply_human_model_overrides(robot_config_a, cfg, agent="A")

    robot_config_b = robot_config_seed
    if robot_config_b.robot_type != robot_type_b:
        robot_config_b = replace(robot_config_b, robot_type=robot_type_b)
    robot_config_b = _apply_human_model_overrides(robot_config_b, cfg, agent="B")

    motion_data_config_a = motion_data_config_seed
    if motion_data_config_a.robot_type != robot_type_a or motion_data_config_a.data_format != data_format:
        motion_data_config_a = replace(motion_data_config_a, robot_type=robot_type_a, data_format=data_format)

    motion_data_config_b = motion_data_config_seed
    if motion_data_config_b.robot_type != robot_type_b or motion_data_config_b.data_format != data_format:
        motion_data_config_b = replace(motion_data_config_b, robot_type=robot_type_b, data_format=data_format)

    return robot_config_a, robot_config_b, motion_data_config_a, motion_data_config_b


def _build_robot_only_task_context(
    *,
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    cfg: DualRetargetConfig,
    asset_root: Path,
) -> tuple[SimpleNamespace, np.ndarray, np.ndarray]:
    task_constants = create_task_constants(
        robot_config=robot_config,
        motion_data_config=motion_data_config,
        task_config=cfg.task_config,
        task_type="robot_only",
    )
    _resolve_asset_paths(task_constants, asset_root)
    object_local_pts, object_local_pts_demo, _ = setup_object_data(
        task_type="robot_only",
        constants=task_constants,
        object_dir=None,
        smpl_scale=1.0,
        task_config=cfg.task_config,
        augmentation=False,
    )
    if object_local_pts is None or object_local_pts_demo is None:
        raise RuntimeError("Failed to create ground points for robot_only task.")
    return task_constants, object_local_pts, object_local_pts_demo


def main(cfg: DualRetargetConfig) -> None:
    robot_config_a, robot_config_b, motion_data_config_a, motion_data_config_b = _resolve_robot_and_motion_config(cfg)

    sequence_file = _pick_sequence_file(cfg.data_dir, cfg.sequence_id)
    sequence_id, fps, human_a, human_b, height_a, height_b = _load_dual_sequence(sequence_file)

    if cfg.max_frames is not None:
        n = min(cfg.max_frames, human_a.shape[0])
        human_a = human_a[:n]
        human_b = human_b[:n]

    logger.info("Sequence: %s", sequence_id)
    logger.info("Frames: %d, Joints: %d, FPS: %d", human_a.shape[0], human_a.shape[1], fps)
    logger.info("Heights (A/B): %.3f / %.3f", height_a, height_b)
    logger.info("Robots (A/B): %s / %s", robot_config_a.robot_type, robot_config_b.robot_type)
    logger.info("Input source: %s", sequence_file)

    human_a_retarget = human_a.copy()
    human_b_retarget = human_b.copy()
    if cfg.input_y_up_to_z_up:
        human_a_retarget = _convert_y_up_to_z_up_points(human_a_retarget)
        human_b_retarget = _convert_y_up_to_z_up_points(human_b_retarget)
        logger.info("Applying right-handed y-up -> z-up conversion before retargeting.")

    hand_extra_a: np.ndarray | None = None
    hand_extra_b: np.ndarray | None = None
    hand_extra_names_a: list[str] = []
    hand_extra_names_b: list[str] = []
    if cfg.augment_human_with_hand_landmarks:
        sequence_dir_candidates = [
            cfg.data_dir,
            cfg.data_dir / sequence_id,
            sequence_file.parent / sequence_id,
        ]
        sequence_dir = next((d for d in sequence_dir_candidates if d.exists() and d.is_dir()), None)
        if sequence_dir is not None:
            hand_extra_a, hand_extra_names_a, hand_extra_b, hand_extra_names_b = _load_hand_landmark_tracks(
                sequence_dir,
                num_frames=human_a_retarget.shape[0],
            )
            if cfg.input_y_up_to_z_up:
                if hand_extra_a is not None:
                    hand_extra_a = _convert_y_up_to_z_up_points(hand_extra_a)
                if hand_extra_b is not None:
                    hand_extra_b = _convert_y_up_to_z_up_points(hand_extra_b)

    os.makedirs(cfg.output_dir, exist_ok=True)

    asset_root = Path(__file__).resolve().parents[1]
    task_constants_a, object_local_pts_a, object_local_pts_demo_a = _build_robot_only_task_context(
        robot_config=robot_config_a,
        motion_data_config=motion_data_config_a,
        cfg=cfg,
        asset_root=asset_root,
    )
    task_constants_b, object_local_pts_b, object_local_pts_demo_b = _build_robot_only_task_context(
        robot_config=robot_config_b,
        motion_data_config=motion_data_config_b,
        cfg=cfg,
        asset_root=asset_root,
    )

    is_human_a = robot_config_a.robot_type == cfg.human_model_robot_type
    is_human_b = robot_config_b.robot_type == cfg.human_model_robot_type

    human_a_solver = human_a_retarget
    human_b_solver = human_b_retarget
    human_names_a_rich: list[str] | None = None
    human_names_b_rich: list[str] | None = None

    if is_human_a and hand_extra_a is not None and hand_extra_names_a:
        human_a_solver = np.concatenate([human_a_solver, hand_extra_a], axis=1)
        task_constants_a.DEMO_JOINTS = list(task_constants_a.DEMO_JOINTS) + list(hand_extra_names_a)
        task_constants_a.JOINTS_MAPPING = {
            **dict(task_constants_a.JOINTS_MAPPING),
            **_human_hand_landmark_mapping(),
        }
        human_names_a_rich = list(task_constants_a.DEMO_JOINTS)
        logger.info(
            "Augmented human A source with hand landmarks: base=%d + hand=%d",
            human_a_retarget.shape[1],
            hand_extra_a.shape[1],
        )
    if is_human_b and hand_extra_b is not None and hand_extra_names_b:
        human_b_solver = np.concatenate([human_b_solver, hand_extra_b], axis=1)
        task_constants_b.DEMO_JOINTS = list(task_constants_b.DEMO_JOINTS) + list(hand_extra_names_b)
        task_constants_b.JOINTS_MAPPING = {
            **dict(task_constants_b.JOINTS_MAPPING),
            **_human_hand_landmark_mapping(),
        }
        human_names_b_rich = list(task_constants_b.DEMO_JOINTS)
        logger.info(
            "Augmented human B source with hand landmarks: base=%d + hand=%d",
            human_b_retarget.shape[1],
            hand_extra_b.shape[1],
        )

    # Persist inspected/prepared inputs for Part 1 debugging.
    prepared_out = cfg.output_dir / f"{sequence_id}_part1_inputs.npz"
    np.savez(
        prepared_out,
        sequence_id=sequence_id,
        fps=fps,
        human_joints_A=human_a,
        human_joints_B=human_b,
        height_A=height_a,
        height_B=height_b,
        robot_A=robot_config_a.robot_type,
        robot_B=robot_config_b.robot_type,
        data_format_A=motion_data_config_a.data_format,
        data_format_B=motion_data_config_b.data_format,
    )
    logger.info("Saved Part-1 prepared inputs: %s", prepared_out)

    if not cfg.run_retarget:
        logger.info("run_retarget=False, stopping after data investigation + part-1 input packaging.")
        return

    if height_a <= 0 or height_b <= 0:
        raise ValueError(f"Invalid height(s): A={height_a}, B={height_b}")

    scale_a = task_constants_a.ROBOT_HEIGHT / height_a
    scale_b = task_constants_b.ROBOT_HEIGHT / height_b
    target_a = "human" if robot_config_a.robot_type == getattr(cfg, "human_model_robot_type", HUMAN_MODEL_ROBOT_TYPE) else "robot"
    target_b = "human" if robot_config_b.robot_type == getattr(cfg, "human_model_robot_type", HUMAN_MODEL_ROBOT_TYPE) else "robot"
    toe_names_a = motion_data_config_a.toe_names
    toe_names_b = motion_data_config_b.toe_names

    if cfg.coupled_dual:
        robot_xml_a = Path(str(task_constants_a.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
        robot_xml_b = Path(str(task_constants_b.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
        dual_scene_xml = (
            cfg.dual_scene_xml
            if cfg.dual_scene_xml is not None
            else cfg.output_dir / f"{sequence_id}_dual_scene.xml"
        )
        dual_scene_xml = _build_dual_scene_xml_from_pair(
            robot_xml_a=robot_xml_a,
            robot_xml_b=robot_xml_b,
            out_path=dual_scene_xml,
            prefix_a=cfg.dual_prefix_a,
            prefix_b=cfg.dual_prefix_b,
        )
        logger.info("Using dual scene XML: %s", dual_scene_xml)

        coupled_scale_a, coupled_scale_b = _resolve_coupled_source_scales(
            scale_a=scale_a,
            scale_b=scale_b,
            target_mode=str(getattr(cfg, "target_mode", "robot_to_robot")).strip().lower(),
        )
        if (coupled_scale_a != scale_a) or (coupled_scale_b != scale_b):
            logger.info(
                "Using shared coupled source scale for interaction graph: %.4f (agent scales were %.4f / %.4f)",
                coupled_scale_a,
                scale_a,
                scale_b,
            )

        human_a_processed, human_b_processed, foot_a, foot_b = _preprocess_dual_humans_for_coupled(
            human_a=human_a_solver,
            human_b=human_b_solver,
            scale_a=coupled_scale_a,
            scale_b=coupled_scale_b,
            toe_names_a=toe_names_a,
            toe_names_b=toe_names_b,
            demo_joints_a=list(task_constants_a.DEMO_JOINTS),
            demo_joints_b=list(task_constants_b.DEMO_JOINTS),
        )

        q_nominal_a: np.ndarray | None = None
        q_nominal_b: np.ndarray | None = None
        if cfg.coupled_warm_start_nominal:
            warm_a_path = cfg.output_dir / f"{sequence_id}_A_coupled_warmstart.npz"
            warm_b_path = cfg.output_dir / f"{sequence_id}_B_coupled_warmstart.npz"
            human_a_warmstart, human_b_warmstart, foot_a_warmstart, foot_b_warmstart = _preprocess_dual_humans_for_coupled(
                human_a=human_a_solver,
                human_b=human_b_solver,
                scale_a=scale_a,
                scale_b=scale_b,
                toe_names_a=toe_names_a,
                toe_names_b=toe_names_b,
                demo_joints_a=list(task_constants_a.DEMO_JOINTS),
                demo_joints_b=list(task_constants_b.DEMO_JOINTS),
            )
            logger.info("Running independent warm-start retargeting for A...")
            q_nominal_a = _run_single_agent_retarget_processed(
                human_processed=human_a_warmstart,
                foot_sticking_sequences=foot_a_warmstart,
                constants=task_constants_a,
                retargeter_cfg=cfg.retargeter,
                object_local_pts=object_local_pts_a,
                object_local_pts_demo=object_local_pts_demo_a,
                out_path=warm_a_path,
            )
            logger.info("Running independent warm-start retargeting for B...")
            q_nominal_b = _run_single_agent_retarget_processed(
                human_processed=human_b_warmstart,
                foot_sticking_sequences=foot_b_warmstart,
                constants=task_constants_b,
                retargeter_cfg=cfg.retargeter,
                object_local_pts=object_local_pts_b,
                object_local_pts_demo=object_local_pts_demo_b,
                out_path=warm_b_path,
            )

            n_shared = min(
                human_a_processed.shape[0],
                human_b_processed.shape[0],
                q_nominal_a.shape[0],
                q_nominal_b.shape[0],
            )
            if n_shared < human_a_processed.shape[0]:
                logger.warning(
                    "Truncating coupled solve to %d frames due to warm-start length mismatch.",
                    n_shared,
                )
            human_a_processed = human_a_processed[:n_shared]
            human_b_processed = human_b_processed[:n_shared]
            foot_a = foot_a[:n_shared]
            foot_b = foot_b[:n_shared]
            q_nominal_a = q_nominal_a[:n_shared]
            q_nominal_b = q_nominal_b[:n_shared]

            q_init_a = np.asarray(q_nominal_a[0], dtype=np.float32)
            q_init_b = np.asarray(q_nominal_b[0], dtype=np.float32)
        else:
            q_init_a = _build_q_init(human_a_processed, task_constants_a.ROBOT_DOF)
            q_init_b = _build_q_init(human_b_processed, task_constants_b.ROBOT_DOF)

        dual_task_constants_a = _make_dual_dense_task_constants(task_constants_a, robot_config_a.robot_type)
        dual_task_constants_b = _make_dual_dense_task_constants(task_constants_b, robot_config_b.robot_type)
        logger.info(
            "Using dual-dense Laplacian mappings: A=%d targets, B=%d targets",
            len(dual_task_constants_a.JOINTS_MAPPING),
            len(dual_task_constants_b.JOINTS_MAPPING),
        )

        dual_retargeter = DualInteractionMeshRetargeter(
            task_constants_a=dual_task_constants_a,
            task_constants_b=dual_task_constants_b,
            dual_scene_xml_path=str(dual_scene_xml),
            robot_a_prefix=cfg.dual_prefix_a,
            robot_b_prefix=cfg.dual_prefix_b,
            q_a_init_idx=cfg.retargeter.q_a_init_idx,
            q_b_init_idx=cfg.retargeter.q_a_init_idx,
            activate_foot_sticking=cfg.retargeter.activate_foot_sticking,
            activate_joint_limits=cfg.retargeter.activate_joint_limits,
            step_size=0.2,
            n_iter_first=40,
            n_iter_other=10,
            collision_detection_threshold=0.08,
            penetration_tolerance=1e-3,
            foot_sticking_tolerance=cfg.retargeter.foot_sticking_tolerance,
            whitelist_margin=-0.015,
            ab_top_k_pairs=24,
            contact_slack_weight=300.0,
            w_nominal_tracking_init=cfg.retargeter.w_nominal_tracking_init,
            nominal_tracking_tau=cfg.retargeter.nominal_tracking_tau,
            debug=cfg.retargeter.debug,
        )
        logger.info("Running coupled dual retargeting for A+B...")
        qpos_a, qpos_b = dual_retargeter.retarget_motion(
            human_joint_motions_a=human_a_processed,
            human_joint_motions_b=human_b_processed,
            foot_sticking_sequences_a=foot_a,
            foot_sticking_sequences_b=foot_b,
            q_nominal_motions_a=q_nominal_a,
            q_nominal_motions_b=q_nominal_b,
            q_init_a=q_init_a,
            q_init_b=q_init_b,
            dest_res_path=str(cfg.output_dir / f"{sequence_id}_dual_coupled_raw.npz"),
        )
        _attach_qpos_coordinate_metadata(cfg.output_dir / f"{sequence_id}_dual_coupled_raw.npz")

        final_path = cfg.output_dir / f"{sequence_id}.npz"
        n = min(qpos_a.shape[0], qpos_b.shape[0], human_a.shape[0], human_b.shape[0])
        np.savez(
            final_path,
            sequence_id=sequence_id,
            fps=fps,
            qpos_A=qpos_a[:n],
            qpos_B=qpos_b[:n],
            human_joints_A=human_a_retarget[:n],
            human_joints_B=human_b_retarget[:n],
            height_A=height_a,
            height_B=height_b,
            scale_A=scale_a,
            scale_B=scale_b,
            mode="coupled_dual",
            target_A=target_a,
            target_B=target_b,
            robot_A=robot_config_a.robot_type,
            robot_B=robot_config_b.robot_type,
            data_format_A=motion_data_config_a.data_format,
            data_format_B=motion_data_config_b.data_format,
            dual_scene_xml=str(dual_scene_xml),
            dual_prefix_A=cfg.dual_prefix_a,
            dual_prefix_B=cfg.dual_prefix_b,
            qpos_coordinate_frame="z_up",
            **(
                {
                    "human_joints_A_rich": human_a_solver[:n],
                    "human_joint_names_A": np.asarray(human_names_a_rich),
                }
                if human_names_a_rich is not None
                else {}
            ),
            **(
                {
                    "human_joints_B_rich": human_b_solver[:n],
                    "human_joint_names_B": np.asarray(human_names_b_rich),
                }
                if human_names_b_rich is not None
                else {}
            ),
        )
        logger.info("Saved coupled dual retarget output: %s", final_path)
        return

    tmp_a = cfg.output_dir / f"{sequence_id}_A_robot_only.npz"
    tmp_b = cfg.output_dir / f"{sequence_id}_B_robot_only.npz"

    logger.info("Running independent retargeting for A (scale=%.4f)...", scale_a)
    qpos_a = _run_single_agent_retarget(
        human_joints=human_a_solver,
        smpl_scale=scale_a,
        toe_names=toe_names_a,
        constants=task_constants_a,
        retargeter_cfg=cfg.retargeter,
        object_local_pts=object_local_pts_a,
        object_local_pts_demo=object_local_pts_demo_a,
        out_path=tmp_a,
    )

    logger.info("Running independent retargeting for B (scale=%.4f)...", scale_b)
    qpos_b = _run_single_agent_retarget(
        human_joints=human_b_solver,
        smpl_scale=scale_b,
        toe_names=toe_names_b,
        constants=task_constants_b,
        retargeter_cfg=cfg.retargeter,
        object_local_pts=object_local_pts_b,
        object_local_pts_demo=object_local_pts_demo_b,
        out_path=tmp_b,
    )

    # Part 1 output format for dual setup (independent A/B retarget only).
    final_path = cfg.output_dir / f"{sequence_id}.npz"
    n = min(qpos_a.shape[0], qpos_b.shape[0])
    np.savez(
        final_path,
        sequence_id=sequence_id,
        fps=fps,
        qpos_A=qpos_a[:n],
        qpos_B=qpos_b[:n],
        human_joints_A=human_a_retarget[:n],
        human_joints_B=human_b_retarget[:n],
        height_A=height_a,
        height_B=height_b,
        scale_A=scale_a,
        scale_B=scale_b,
        mode="independent_dual",
        target_A=target_a,
        target_B=target_b,
        robot_A=robot_config_a.robot_type,
        robot_B=robot_config_b.robot_type,
        data_format_A=motion_data_config_a.data_format,
        data_format_B=motion_data_config_b.data_format,
        qpos_coordinate_frame="z_up",
        **(
            {
                "human_joints_A_rich": human_a_solver[:n],
                "human_joint_names_A": np.asarray(human_names_a_rich),
            }
            if human_names_a_rich is not None
            else {}
        ),
        **(
            {
                "human_joints_B_rich": human_b_solver[:n],
                "human_joint_names_B": np.asarray(human_names_b_rich),
            }
            if human_names_b_rich is not None
            else {}
        ),
    )
    logger.info("Saved dual Part-1 retarget output: %s", final_path)


if __name__ == "__main__":
    main(tyro.cli(DualRetargetConfig))
