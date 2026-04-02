from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.examples.dual_robot_retarget import (
    DualRetargetConfig,
    _build_robot_only_task_context,
    _convert_y_up_to_z_up_points,
    _load_dual_sequence,
    _make_dual_dense_task_constants,
    _pick_sequence_file,
    _resolve_robot_and_motion_config,
    _run_single_agent_retarget_processed,
)
from holosoma_retargeting.src.mixed_human_robot_retargeter import (
    MixedHumanRobotRetargeter,
    build_human_proxy_scene_xml,
)
from holosoma_retargeting.src.utils import (
    extract_foot_sticking_sequence_velocity,
    preprocess_motion_data,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _height_normalise(
    joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    mat_height: float = 0.1,
) -> np.ndarray:
    joints = joints.copy()
    z_shift = _compute_height_shift(joints, demo_joints, toe_names, mat_height=mat_height)
    joints[:, :, 2] -= z_shift
    return joints


def _compute_height_shift(
    joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    mat_height: float = 0.1,
) -> float:
    toe_indices = [demo_joints.index(name) for name in toe_names if name in demo_joints]
    z_shift = 0.0
    if toe_indices:
        z_min = float(joints[:, toe_indices, 2].min())
        if z_min >= mat_height:
            z_min -= mat_height
        z_shift = z_min
    return float(z_shift)


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


@dataclass
class MixedRetargetConfig:
    data_dir: Path = Path("DATA/interx_dual_inputs")
    """Input directory containing dual-human Inter-X npz files."""

    sequence_id: str | None = None
    """Specific sequence id to process. If omitted, uses the first sorted file."""

    output_dir: Path = Path("DATA/interx_mixed_outputs")
    """Output directory for mixed retarget files."""

    robot: str = "g1"
    """Robot type used for the retargeted participant."""

    robot_side: str = "A"
    """Original participant to retarget to a robot: 'A' or 'B'."""

    data_format: str = "smplx"
    """Joint naming convention for the input motion."""

    input_y_up_to_z_up: bool = True
    """Apply right-handed y-up -> z-up conversion before retargeting."""

    max_frames: int | None = None
    """Optional frame cap for debugging."""

    human_prefix: str = "H_"
    """Prefix for generated MuJoCo human proxy bodies/geoms."""

    warm_start_nominal: bool = True
    """Run single-agent retarget first and use it as mixed warm start + nominal."""

    human_height_mode: Literal["source", "robot"] = "robot"
    """Scale the fixed human to source height or to the robot nominal height."""

    mixed_step_size: float = 0.12
    """SQP step-size trust region radius; lower is more conservative for contact handling."""

    collision_detection_threshold: float = 0.12
    """Distance threshold for candidate robot-human contact linearization."""

    penetration_tolerance: float = 2e-4
    """Allowed penetration slack before constraints are violated."""

    human_contact_slack_weight: float = 10000.0
    """Penalty on robot-human contact slack; higher discourages penetration."""

    allow_human_contact_slack: bool = True
    """If False, enforce hard robot-human non-penetration constraints without slack."""

    human_clearance_margin: float = 0.01
    """Target positive robot-human distance margin in meters."""

    n_iter_initial: int = 80
    """Number of SQP iterations on the first frame."""

    n_iter_per_frame: int = 20
    """Number of SQP iterations on subsequent frames."""

    collision_polish_passes: int = 8
    """Post-solve per-frame contact polish passes to reduce residual human penetration."""

    collision_polish_step_size: float = 0.02
    """Step-size limit for each collision polish pass."""

    collision_polish_slack_weight: float = 200000.0
    """Penalty on polish slack; higher enforces stronger non-penetration correction."""

    collision_polish_target_margin: float = 0.0
    """Target minimum signed distance (meters) for robot-human contact after polish."""

    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplx", robot_type="g1")
    )
    task_config: TaskConfig = field(default_factory=TaskConfig)
    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)


def main(cfg: MixedRetargetConfig) -> None:
    side = cfg.robot_side.upper()
    if side not in {"A", "B"}:
        raise ValueError(f"robot_side must be 'A' or 'B', got {cfg.robot_side!r}")

    shared_cfg = DualRetargetConfig(
        data_dir=cfg.data_dir,
        sequence_id=cfg.sequence_id,
        output_dir=cfg.output_dir,
        robot=cfg.robot,
        robot_a=cfg.robot,
        robot_b=cfg.robot,
        data_format=cfg.data_format,
        input_y_up_to_z_up=cfg.input_y_up_to_z_up,
        max_frames=cfg.max_frames,
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        retargeter=cfg.retargeter,
    )
    robot_cfg_a, _, motion_cfg_a, motion_cfg_b = _resolve_robot_and_motion_config(shared_cfg)

    sequence_file = _pick_sequence_file(cfg.data_dir, cfg.sequence_id)
    seq_id, fps, human_a, human_b, height_a, height_b = _load_dual_sequence(sequence_file)
    if cfg.max_frames is not None:
        n = min(int(cfg.max_frames), human_a.shape[0])
        human_a = human_a[:n]
        human_b = human_b[:n]
    n_frames = min(human_a.shape[0], human_b.shape[0])
    hand_extra_a, hand_extra_names_a, hand_extra_b, hand_extra_names_b = _load_hand_landmark_tracks(
        cfg.data_dir / seq_id,
        num_frames=n_frames,
    )

    if cfg.input_y_up_to_z_up:
        human_a = _convert_y_up_to_z_up_points(human_a)
        human_b = _convert_y_up_to_z_up_points(human_b)
        if hand_extra_a is not None:
            hand_extra_a = _convert_y_up_to_z_up_points(hand_extra_a)
        if hand_extra_b is not None:
            hand_extra_b = _convert_y_up_to_z_up_points(hand_extra_b)
        logger.info("Applied y-up -> z-up coordinate conversion.")

    if side == "A":
        robot_source = human_a
        fixed_human = human_b
        robot_height_source = height_a
        fixed_height = height_b
        motion_cfg_robot = motion_cfg_a
        motion_cfg_human = motion_cfg_b
        fixed_human_extra = hand_extra_b
        fixed_human_extra_names = hand_extra_names_b
    else:
        robot_source = human_b
        fixed_human = human_a
        robot_height_source = height_b
        fixed_height = height_a
        motion_cfg_robot = motion_cfg_b
        motion_cfg_human = motion_cfg_a
        fixed_human_extra = hand_extra_a
        fixed_human_extra_names = hand_extra_names_a

    os.makedirs(cfg.output_dir, exist_ok=True)
    asset_root = Path(__file__).resolve().parents[1]

    task_constants, object_local_pts, object_local_pts_demo = _build_robot_only_task_context(
        robot_config=robot_cfg_a,
        motion_data_config=motion_cfg_robot,
        cfg=shared_cfg,
        asset_root=asset_root,
    )

    demo_joints_robot = list(task_constants.DEMO_JOINTS)
    demo_joints_human = list(motion_cfg_human.resolved_demo_joints)
    robot_source_norm = _height_normalise(robot_source, demo_joints_robot, motion_cfg_robot.toe_names)
    fixed_human_norm = _height_normalise(fixed_human, demo_joints_human, motion_cfg_human.toe_names)
    fixed_human_shift = _compute_height_shift(fixed_human, demo_joints_human, motion_cfg_human.toe_names)

    robot_height = float(getattr(task_constants, "ROBOT_HEIGHT", 1.78))
    robot_scale = robot_height / float(robot_height_source)
    human_scale = 1.0 if cfg.human_height_mode == "source" else robot_height / float(fixed_height)
    fixed_human_scaled = fixed_human_norm * float(human_scale)
    robot_source_scaled = robot_source_norm * robot_scale
    foot_seq = extract_foot_sticking_sequence_velocity(
        robot_source_scaled,
        demo_joints_robot,
        motion_cfg_robot.toe_names,
    )

    q_nominal = None
    q_init = None
    if cfg.warm_start_nominal:
        warmstart_path = cfg.output_dir / f"{seq_id}_{side}_warmstart.npz"
        logger.info("Running single-agent warm start for robot side %s...", side)
        robot_processed = preprocess_motion_data(
            human_joints=robot_source.copy(),
            retargeter=type("DemoProxy", (), {"demo_joints": demo_joints_robot})(),
            foot_names=motion_cfg_robot.toe_names,
            scale=robot_scale,
        )
        q_nominal = _run_single_agent_retarget_processed(
            human_processed=robot_processed,
            foot_sticking_sequences=foot_seq,
            constants=task_constants,
            retargeter_cfg=cfg.retargeter,
            object_local_pts=object_local_pts,
            object_local_pts_demo=object_local_pts_demo,
            out_path=warmstart_path,
        )
        q_init = np.asarray(q_nominal[0], dtype=np.float32)

    fixed_human_rich = fixed_human_scaled[:, :22]
    fixed_human_rich_names = list(demo_joints_human[:22])
    if fixed_human_extra is not None and fixed_human_extra_names:
        fixed_human_extra_norm = fixed_human_extra.copy()
        fixed_human_extra_norm[:, :, 2] -= fixed_human_shift
        fixed_human_extra_scaled = fixed_human_extra_norm * float(human_scale)
        fixed_human_rich = np.concatenate([fixed_human_rich, fixed_human_extra_scaled], axis=1)
        fixed_human_rich_names.extend(fixed_human_extra_names)
        logger.info(
            "Using richer human proxy joints: base=%d + hand=%d",
            22,
            fixed_human_extra_scaled.shape[1],
        )

    robot_xml = Path(str(task_constants.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
    mixed_scene_xml = build_human_proxy_scene_xml(
        robot_xml_path=robot_xml,
        out_path=cfg.output_dir / f"{seq_id}_{side}_mixed_scene.xml",
        human_joint_trajectory=fixed_human_rich,
        joint_names=fixed_human_rich_names,
        human_prefix=cfg.human_prefix,
    )
    logger.info("Mixed scene XML: %s", mixed_scene_xml)

    dense_constants = _make_dual_dense_task_constants(task_constants, robot_cfg_a.robot_type)
    retargeter = MixedHumanRobotRetargeter(
        task_constants=dense_constants,
        mixed_scene_xml_path=str(mixed_scene_xml),
        human_joint_names=fixed_human_rich_names,
        human_prefix=cfg.human_prefix,
        q_a_init_idx=cfg.retargeter.q_a_init_idx,
        activate_foot_sticking=cfg.retargeter.activate_foot_sticking,
        activate_joint_limits=cfg.retargeter.activate_joint_limits,
        step_size=cfg.mixed_step_size,
        collision_detection_threshold=cfg.collision_detection_threshold,
        penetration_tolerance=cfg.penetration_tolerance,
        foot_sticking_tolerance=cfg.retargeter.foot_sticking_tolerance,
        human_contact_slack_weight=cfg.human_contact_slack_weight,
        allow_human_contact_slack=cfg.allow_human_contact_slack,
        human_clearance_margin=cfg.human_clearance_margin,
        collision_polish_passes=cfg.collision_polish_passes,
        collision_polish_step_size=cfg.collision_polish_step_size,
        collision_polish_slack_weight=cfg.collision_polish_slack_weight,
        collision_polish_target_margin=cfg.collision_polish_target_margin,
        w_nominal_tracking_init=cfg.retargeter.w_nominal_tracking_init,
        nominal_tracking_tau=cfg.retargeter.nominal_tracking_tau,
        debug=cfg.retargeter.debug,
    )

    logger.info(
        "Running mixed robot-human retargeting for %d frames. Robot side=%s | Human side=%s",
        robot_source_norm.shape[0],
        side,
        "B" if side == "A" else "A",
    )
    raw_path = cfg.output_dir / f"{seq_id}_{side}_mixed_raw.npz"
    qpos_robot = retargeter.retarget_motion(
        robot_human_joints_raw=robot_source_norm[:, :22],
        human_joints_fixed=fixed_human_rich,
        robot_height_source=robot_height_source,
        human_scale_fixed=human_scale,
        foot_sticking_sequences=foot_seq,
        q_init_a=q_init,
        q_nominal_a=q_nominal,
        dest_res_path=str(raw_path),
        n_iter_initial=cfg.n_iter_initial,
        n_iter_per_frame=cfg.n_iter_per_frame,
    )

    final_path = cfg.output_dir / f"{seq_id}.npz"
    if side == "A":
        payload = dict(
            qpos_A=qpos_robot,
            human_joints_A=robot_source_norm[:, :22],
            human_joints_B=fixed_human_scaled[:, :22],
            human_joints_B_rich=fixed_human_rich,
            human_joint_names_B=np.asarray(fixed_human_rich_names),
            target_A=np.asarray("robot"),
            target_B=np.asarray("human"),
            robot_A=np.asarray(robot_cfg_a.robot_type),
            robot_B=np.asarray("human"),
            scale_A=np.asarray(robot_scale, dtype=np.float32),
            scale_B=np.asarray(human_scale, dtype=np.float32),
        )
    else:
        payload = dict(
            qpos_B=qpos_robot,
            human_joints_A=fixed_human_scaled[:, :22],
            human_joints_B=robot_source_norm[:, :22],
            human_joints_A_rich=fixed_human_rich,
            human_joint_names_A=np.asarray(fixed_human_rich_names),
            target_A=np.asarray("human"),
            target_B=np.asarray("robot"),
            robot_A=np.asarray("human"),
            robot_B=np.asarray(robot_cfg_a.robot_type),
            scale_A=np.asarray(human_scale, dtype=np.float32),
            scale_B=np.asarray(robot_scale, dtype=np.float32),
        )

    np.savez(
        final_path,
        sequence_id=seq_id,
        fps=fps,
        mode=np.asarray("mixed_robot_human"),
        robot_side=np.asarray(side),
        mixed_scene_xml=np.asarray(str(mixed_scene_xml)),
        human_collision_mode=np.asarray("mujoco_proxy"),
        human_proxy_prefix=np.asarray(cfg.human_prefix),
        human_scale_mode=np.asarray(cfg.human_height_mode),
        qpos_coordinate_frame=np.asarray("z_up"),
        data_format_A=np.asarray(motion_cfg_a.data_format),
        data_format_B=np.asarray(motion_cfg_b.data_format),
        height_A=np.asarray(height_a, dtype=np.float32),
        height_B=np.asarray(height_b, dtype=np.float32),
        **payload,
    )
    logger.info("Saved mixed robot-human output: %s", final_path)


if __name__ == "__main__":
    main(tyro.cli(MixedRetargetConfig))
