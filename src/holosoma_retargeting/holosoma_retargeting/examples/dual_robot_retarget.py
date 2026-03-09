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
    """Robot type."""

    data_format: str = "smplx"
    """Motion data format used for joint naming/mapping."""

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
    return np.asarray(qpos, dtype=np.float32)


def _prefix_attributes_in_subtree(elem: ET.Element, prefix: str, attrs: tuple[str, ...]) -> None:
    for node in elem.iter():
        for attr in attrs:
            if attr in node.attrib:
                node.set(attr, prefix + node.attrib[attr])


def _prefix_names_in_subtree(elem: ET.Element, prefix: str) -> None:
    _prefix_attributes_in_subtree(elem, prefix, ("name", "joint", "body", "geom", "site", "tendon"))


def _duplicate_and_prefix_section(
    root: ET.Element,
    section_tag: str,
    prefix_a: str,
    prefix_b: str,
    attrs_to_prefix: tuple[str, ...],
) -> None:
    section = root.find(section_tag)
    if section is None:
        return

    original_children = list(section)
    if not original_children:
        return

    for child in original_children:
        section.remove(child)

    for prefix in (prefix_a, prefix_b):
        for child in original_children:
            child_copy = deepcopy(child)
            _prefix_attributes_in_subtree(child_copy, prefix, attrs_to_prefix)
            section.append(child_copy)


def _build_dual_scene_xml_from_single(single_xml_path: Path, out_path: Path, prefix_a: str, prefix_b: str) -> Path:
    tree = ET.parse(str(single_xml_path))
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir")
        if meshdir:
            meshdir_abs = Path(meshdir)
            if not meshdir_abs.is_absolute():
                meshdir_abs = (single_xml_path.parent / meshdir_abs).resolve()
        else:
            assets_dir = single_xml_path.parent / "assets"
            meshes_dir = single_xml_path.parent / "meshes"
            if assets_dir.exists():
                meshdir_abs = assets_dir.resolve()
            elif meshes_dir.exists():
                meshdir_abs = meshes_dir.resolve()
            else:
                meshdir_abs = single_xml_path.parent.resolve()
        compiler.set("meshdir", str(meshdir_abs) + "/")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"worldbody not found in XML: {single_xml_path}")

    robot_bodies = [child for child in list(worldbody) if child.tag == "body" and child.find("freejoint") is not None]
    if not robot_bodies:
        raise ValueError(f"No root robot body with freejoint found in XML: {single_xml_path}")
    if len(robot_bodies) > 1:
        logger.warning("Found %d root freejoint bodies; using first: %s", len(robot_bodies), robot_bodies[0].get("name"))

    template_body = robot_bodies[0]
    worldbody.remove(template_body)

    body_a = deepcopy(template_body)
    body_b = deepcopy(template_body)
    _prefix_names_in_subtree(body_a, prefix_a)
    _prefix_names_in_subtree(body_b, prefix_b)

    worldbody.append(body_a)
    worldbody.append(body_b)

    dual_ref_attrs = (
        "name",
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
        "slidersite",
        "cranksite",
    )
    _duplicate_and_prefix_section(root, "sensor", prefix_a, prefix_b, dual_ref_attrs)
    _duplicate_and_prefix_section(root, "actuator", prefix_a, prefix_b, dual_ref_attrs)
    _duplicate_and_prefix_section(root, "tendon", prefix_a, prefix_b, dual_ref_attrs)
    _duplicate_and_prefix_section(root, "equality", prefix_a, prefix_b, dual_ref_attrs)
    _duplicate_and_prefix_section(root, "contact", prefix_a, prefix_b, dual_ref_attrs)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


def _preprocess_dual_humans_for_coupled(
    human_a: np.ndarray,
    human_b: np.ndarray,
    scale_a: float,
    scale_b: float,
    toe_names: list[str],
    demo_joints: list[str],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, bool]], list[dict[str, bool]]]:
    demo_proxy = SimpleNamespace(demo_joints=demo_joints)
    human_a_processed = preprocess_motion_data(
        human_joints=human_a.copy(),
        retargeter=demo_proxy,
        foot_names=toe_names,
        scale=scale_a,
    )
    human_b_processed = preprocess_motion_data(
        human_joints=human_b.copy(),
        retargeter=demo_proxy,
        foot_names=toe_names,
        scale=scale_b,
    )
    foot_a = extract_foot_sticking_sequence_velocity(human_a_processed, demo_joints, toe_names)
    foot_b = extract_foot_sticking_sequence_velocity(human_b_processed, demo_joints, toe_names)
    return human_a_processed, human_b_processed, foot_a, foot_b


def _resolve_asset_paths(constants: SimpleNamespace, asset_root: Path) -> None:
    def _to_abs(path_value: str | None) -> str | None:
        if path_value is None:
            return None
        p = Path(path_value)
        if p.is_absolute():
            return str(p)
        return str(asset_root / p)

    constants.ROBOT_URDF_FILE = _to_abs(getattr(constants, "ROBOT_URDF_FILE", None))
    constants.OBJECT_URDF_FILE = _to_abs(getattr(constants, "OBJECT_URDF_FILE", None))
    constants.OBJECT_MESH_FILE = _to_abs(getattr(constants, "OBJECT_MESH_FILE", None))
    if hasattr(constants, "SCENE_XML_FILE"):
        constants.SCENE_XML_FILE = _to_abs(getattr(constants, "SCENE_XML_FILE", None))


def _resolve_robot_and_motion_config(
    cfg: DualRetargetConfig,
) -> tuple[RobotConfig, MotionDataConfig]:
    """Align nested configs with top-level robot/data format while preserving overrides."""
    robot_config = cfg.robot_config
    if robot_config.robot_type != cfg.robot:
        robot_config = replace(robot_config, robot_type=cfg.robot)

    motion_data_config = cfg.motion_data_config
    if motion_data_config.robot_type != cfg.robot or motion_data_config.data_format != cfg.data_format:
        motion_data_config = replace(motion_data_config, robot_type=cfg.robot, data_format=cfg.data_format)

    return robot_config, motion_data_config


def main(cfg: DualRetargetConfig) -> None:
    robot_config, motion_data_config = _resolve_robot_and_motion_config(cfg)

    sequence_file = _pick_sequence_file(cfg.data_dir, cfg.sequence_id)
    sequence_id, fps, human_a, human_b, height_a, height_b = _load_dual_sequence(sequence_file)

    if cfg.max_frames is not None:
        n = min(cfg.max_frames, human_a.shape[0])
        human_a = human_a[:n]
        human_b = human_b[:n]

    logger.info("Sequence: %s", sequence_id)
    logger.info("Frames: %d, Joints: %d, FPS: %d", human_a.shape[0], human_a.shape[1], fps)
    logger.info("Heights (A/B): %.3f / %.3f", height_a, height_b)
    logger.info("Input source: %s", sequence_file)

    os.makedirs(cfg.output_dir, exist_ok=True)

    task_constants = create_task_constants(
        robot_config=robot_config,
        motion_data_config=motion_data_config,
        task_config=cfg.task_config,
        task_type="robot_only",
    )
    asset_root = Path(__file__).resolve().parents[1]
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
        robot=robot_config.robot_type,
        data_format=motion_data_config.data_format,
    )
    logger.info("Saved Part-1 prepared inputs: %s", prepared_out)

    if not cfg.run_retarget:
        logger.info("run_retarget=False, stopping after data investigation + part-1 input packaging.")
        return

    if height_a <= 0 or height_b <= 0:
        raise ValueError(f"Invalid height(s): A={height_a}, B={height_b}")

    scale_a = task_constants.ROBOT_HEIGHT / height_a
    scale_b = task_constants.ROBOT_HEIGHT / height_b
    toe_names = motion_data_config.toe_names

    if cfg.coupled_dual:
        robot_xml_single = Path(str(task_constants.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
        dual_scene_xml = (
            cfg.dual_scene_xml
            if cfg.dual_scene_xml is not None
            else cfg.output_dir / f"{sequence_id}_dual_scene.xml"
        )
        dual_scene_xml = _build_dual_scene_xml_from_single(
            single_xml_path=robot_xml_single,
            out_path=dual_scene_xml,
            prefix_a=cfg.dual_prefix_a,
            prefix_b=cfg.dual_prefix_b,
        )
        logger.info("Using dual scene XML: %s", dual_scene_xml)

        human_a_processed, human_b_processed, foot_a, foot_b = _preprocess_dual_humans_for_coupled(
            human_a=human_a,
            human_b=human_b,
            scale_a=scale_a,
            scale_b=scale_b,
            toe_names=toe_names,
            demo_joints=list(task_constants.DEMO_JOINTS),
        )

        q_nominal_a: np.ndarray | None = None
        q_nominal_b: np.ndarray | None = None
        if cfg.coupled_warm_start_nominal:
            warm_a_path = cfg.output_dir / f"{sequence_id}_A_coupled_warmstart.npz"
            warm_b_path = cfg.output_dir / f"{sequence_id}_B_coupled_warmstart.npz"
            logger.info("Running independent warm-start retargeting for A...")
            q_nominal_a = _run_single_agent_retarget_processed(
                human_processed=human_a_processed,
                foot_sticking_sequences=foot_a,
                constants=task_constants,
                retargeter_cfg=cfg.retargeter,
                object_local_pts=object_local_pts,
                object_local_pts_demo=object_local_pts_demo,
                out_path=warm_a_path,
            )
            logger.info("Running independent warm-start retargeting for B...")
            q_nominal_b = _run_single_agent_retarget_processed(
                human_processed=human_b_processed,
                foot_sticking_sequences=foot_b,
                constants=task_constants,
                retargeter_cfg=cfg.retargeter,
                object_local_pts=object_local_pts,
                object_local_pts_demo=object_local_pts_demo,
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
            q_init_a = _build_q_init(human_a_processed, task_constants.ROBOT_DOF)
            q_init_b = _build_q_init(human_b_processed, task_constants.ROBOT_DOF)

        dual_retargeter = DualInteractionMeshRetargeter(
            task_constants=task_constants,
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

        final_path = cfg.output_dir / f"{sequence_id}.npz"
        n = min(qpos_a.shape[0], qpos_b.shape[0], human_a.shape[0], human_b.shape[0])
        np.savez(
            final_path,
            sequence_id=sequence_id,
            fps=fps,
            qpos_A=qpos_a[:n],
            qpos_B=qpos_b[:n],
            height_A=height_a,
            height_B=height_b,
            scale_A=scale_a,
            scale_B=scale_b,
            mode="coupled_dual",
            dual_scene_xml=str(dual_scene_xml),
            dual_prefix_A=cfg.dual_prefix_a,
            dual_prefix_B=cfg.dual_prefix_b,
        )
        logger.info("Saved coupled dual retarget output: %s", final_path)
        return

    tmp_a = cfg.output_dir / f"{sequence_id}_A_robot_only.npz"
    tmp_b = cfg.output_dir / f"{sequence_id}_B_robot_only.npz"

    logger.info("Running independent retargeting for A (scale=%.4f)...", scale_a)
    qpos_a = _run_single_agent_retarget(
        human_joints=human_a,
        smpl_scale=scale_a,
        toe_names=toe_names,
        constants=task_constants,
        retargeter_cfg=cfg.retargeter,
        object_local_pts=object_local_pts,
        object_local_pts_demo=object_local_pts_demo,
        out_path=tmp_a,
    )

    logger.info("Running independent retargeting for B (scale=%.4f)...", scale_b)
    qpos_b = _run_single_agent_retarget(
        human_joints=human_b,
        smpl_scale=scale_b,
        toe_names=toe_names,
        constants=task_constants,
        retargeter_cfg=cfg.retargeter,
        object_local_pts=object_local_pts,
        object_local_pts_demo=object_local_pts_demo,
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
        height_A=height_a,
        height_B=height_b,
        scale_A=scale_a,
        scale_B=scale_b,
    )
    logger.info("Saved dual Part-1 retarget output: %s", final_path)


if __name__ == "__main__":
    main(tyro.cli(DualRetargetConfig))
