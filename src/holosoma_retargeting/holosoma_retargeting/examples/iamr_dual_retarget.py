"""IAMR dual-robot retargeting entrypoint.

Implements the full IAMR pipeline from:
    "Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids"
    arXiv:2603.02856

Pipeline:
    1. Load Inter-X dual-human .npz (raw, unscaled joints + heights).
    2. Load SMPL-X body pose parameters (pose_body) from raw Inter-X P1/P2 files.
    3. Height-normalise each agent (shift z so feet touch ground) — NO scaling.
    4. Run IAMRRetargeter (dual manifolds + topological partitioning + asymmetric SQP).
       J_rot uses θ̂_k^src directly from SMPL-X pose parameters, as per the paper.
    5. Save qpos_A, qpos_B, interaction_graph, contact_graph to .npz.

Usage:
    python iamr_dual_retarget.py \\
        --data_dir DATA/interx_dual_inputs \\
        --raw_interx_dir Inter-X/visualize/smplx_viewer_tool/data/motions \\
        --output_dir DATA/interx_iamr_outputs \\
        --sequence_id G001T000A000R000
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.path_utils import resolve_portable_path
from holosoma_retargeting.examples.dual_robot_retarget import (
    _build_dual_scene_xml_from_pair,
    _build_robot_only_task_context,
    _convert_y_up_to_z_up_points,
    _load_dual_sequence,
    _make_dual_dense_task_constants,
    _pick_sequence_file,
    _resolve_robot_and_motion_config,
)
from holosoma_retargeting.src.iamr_retargeter import IAMRRetargeter
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# y-up → z-up rotation matrix (right-handed, matches dual_joint_renderer.py)
_R_YUP_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float32
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class IAMRRetargetConfig:
    """Configuration for the IAMR dual-robot retargeting pipeline."""

    data_dir: Path = Path("DATA/interx_dual_inputs")
    """Input directory containing dual-human .npz files (Inter-X format)."""

    sequence_id: str | None = None
    """Specific sequence stem to process; if None, picks the first sorted file."""

    output_dir: Path = Path("DATA/interx_iamr_outputs")
    """Output directory for retargeted .npz files."""

    robot: str = "g1"
    """Shared robot type (used for both A/B unless overridden)."""

    robot_a: str | None = None
    """Optional robot type override for agent A."""

    robot_b: str | None = None
    """Optional robot type override for agent B."""

    data_format: str = "smplx"
    """Motion data joint naming convention."""

    input_y_up_to_z_up: bool = True
    """Apply right-handed y-up → z-up conversion to raw joints before retargeting."""

    raw_interx_dir: Path | None = None
    """Directory containing raw Inter-X per-sequence folders (e.g.
    ``Inter-X/visualize/smplx_viewer_tool/data/motions``).  Each subfolder
    must contain ``P1.npz`` and ``P2.npz`` with SMPL-X ``pose_body`` arrays.
    When provided, J_rot uses the paper-correct θ̂_k^src from source SMPL-X
    rotations.  When None, J_rot is disabled."""

    run_iamr: bool = True
    """Whether to run the IAMR retargeter (set False for dry-run / data inspection)."""

    max_frames: int | None = None
    """Cap on frames to process (useful for fast debugging)."""

    dual_prefix_a: str = "A_"
    """MuJoCo body/geom name prefix for robot A in the dual scene XML."""

    dual_prefix_b: str = "B_"
    """MuJoCo body/geom name prefix for robot B in the dual scene XML."""

    dual_scene_xml: Path | None = None
    """Pre-built dual scene XML; auto-generated when None."""

    # IAMR hyper-parameters (paper defaults — arXiv:2603.02856)
    w_self: float = 2.0
    w_inter: float = 10.0
    w_reg: float = 0.1
    lambda_rot: float = 0.1
    omega_max: float = 1.0
    gamma_decay: float = 5.0

    # SQP / constraint settings
    step_size: float = 0.2
    n_iter_first: int = 40
    n_iter_other: int = 10
    activate_foot_sticking: bool = True
    activate_joint_limits: bool = True
    penetration_tolerance: float = 1e-3
    foot_sticking_tolerance: float = 1e-3
    collision_detection_threshold: float = 0.08

    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplx", robot_type="g1")
    )
    task_config: TaskConfig = field(default_factory=TaskConfig)
    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)


# ---------------------------------------------------------------------------
# Height normalisation (applied BEFORE IAMR — no scaling, just z-shift)
# ---------------------------------------------------------------------------


def _height_normalise(
    joints: np.ndarray,
    demo_joints: list[str],
    toe_names: list[str],
    mat_height: float = 0.1,
) -> np.ndarray:
    """Shift joints vertically so the lowest toe just touches z=0.

    This is Step 1 of preprocess_motion_data without the scale step.
    IAMR computes its own per-agent and unified scales internally.
    """
    joints = joints.copy()
    toe_indices = [demo_joints.index(n) for n in toe_names if n in demo_joints]
    if not toe_indices:
        return joints
    z_min = joints[:, toe_indices, 2].min()
    if z_min >= mat_height:
        z_min -= mat_height
    joints[:, :, 2] -= z_min
    return joints


# ---------------------------------------------------------------------------
# SMPL-X pose loader
# ---------------------------------------------------------------------------


def _load_smplx_poses(
    seq_id: str,
    raw_interx_dir: Path,
    n_frames_proc: int,
    fps_proc: int = 30,
    fps_raw: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and subsample SMPL-X ``pose_body`` arrays for one Inter-X sequence.

    The raw Inter-X data is stored at 120 fps (or inferred from fps_raw);
    the preprocessed positions are at fps_proc (default 30).  We subsample
    with stride = fps_raw // fps_proc and trim to n_frames_proc.

    Returns:
        poses_a: (n_frames_proc, 21, 3) float32 — SMPL-X body pose for P1.
        poses_b: (n_frames_proc, 21, 3) float32 — SMPL-X body pose for P2.
    """
    p1_path = raw_interx_dir / seq_id / "P1.npz"
    p2_path = raw_interx_dir / seq_id / "P2.npz"
    if not p1_path.exists() or not p2_path.exists():
        raise FileNotFoundError(
            f"Raw Inter-X files not found: {p1_path}, {p2_path}\n"
            "Pass --raw_interx_dir pointing to the motions directory."
        )
    d1 = np.load(str(p1_path), allow_pickle=True)
    d2 = np.load(str(p2_path), allow_pickle=True)

    stride = max(1, fps_raw // fps_proc)
    poses_a = d1["pose_body"][::stride].astype(np.float32)
    poses_b = d2["pose_body"][::stride].astype(np.float32)

    # Trim to n_frames_proc (or pad with last frame if short)
    def _align(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] >= n_frames_proc:
            return arr[:n_frames_proc]
        pad = np.repeat(arr[-1:], n_frames_proc - arr.shape[0], axis=0)
        return np.concatenate([arr, pad], axis=0)

    return _align(poses_a), _align(poses_b)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: IAMRRetargetConfig) -> None:
    cfg.data_dir = resolve_portable_path(cfg.data_dir, prefer_bundle=True, must_exist=True)
    cfg.output_dir = resolve_portable_path(cfg.output_dir, prefer_bundle=True)
    if cfg.raw_interx_dir is not None:
        cfg.raw_interx_dir = resolve_portable_path(cfg.raw_interx_dir, prefer_bundle=True, must_exist=True)
    if cfg.dual_scene_xml is not None:
        cfg.dual_scene_xml = resolve_portable_path(cfg.dual_scene_xml, prefer_bundle=True)

    robot_cfg_a, robot_cfg_b, motion_cfg_a, motion_cfg_b = (
        _resolve_robot_and_motion_config(cfg)
    )

    sequence_file = _pick_sequence_file(cfg.data_dir, cfg.sequence_id)
    seq_id, fps, human_a, human_b, height_a, height_b = _load_dual_sequence(sequence_file)

    if cfg.max_frames is not None:
        n = min(cfg.max_frames, human_a.shape[0])
        human_a = human_a[:n]
        human_b = human_b[:n]

    logger.info("Sequence: %s | Frames: %d | FPS: %d", seq_id, human_a.shape[0], fps)
    logger.info(
        "Heights A/B: %.3f / %.3f | Robots A/B: %s / %s",
        height_a, height_b,
        robot_cfg_a.robot_type, robot_cfg_b.robot_type,
    )

    if height_a <= 0 or height_b <= 0:
        raise ValueError(f"Invalid heights: A={height_a}, B={height_b}")

    # ---- y-up → z-up conversion -------------------------------------------
    if cfg.input_y_up_to_z_up:
        human_a = _convert_y_up_to_z_up_points(human_a)
        human_b = _convert_y_up_to_z_up_points(human_b)
        logger.info("Applied y-up → z-up coordinate conversion.")

    os.makedirs(cfg.output_dir, exist_ok=True)

    asset_root = Path(__file__).resolve().parents[1]

    task_constants_a, object_pts_a, object_pts_demo_a = _build_robot_only_task_context(
        robot_config=robot_cfg_a,
        motion_data_config=motion_cfg_a,
        cfg=cfg,
        asset_root=asset_root,
    )
    task_constants_b, object_pts_b, object_pts_demo_b = _build_robot_only_task_context(
        robot_config=robot_cfg_b,
        motion_data_config=motion_cfg_b,
        cfg=cfg,
        asset_root=asset_root,
    )

    if not cfg.run_iamr:
        logger.info("run_iamr=False; stopping after data preparation.")
        return

    # ---- Height normalisation (NO scaling — IAMR handles that internally) -
    toe_names_a = motion_cfg_a.toe_names
    toe_names_b = motion_cfg_b.toe_names
    demo_joints_a = list(task_constants_a.DEMO_JOINTS)
    demo_joints_b = list(task_constants_b.DEMO_JOINTS)

    human_a_norm = _height_normalise(human_a, demo_joints_a, toe_names_a)
    human_b_norm = _height_normalise(human_b, demo_joints_b, toe_names_b)

    # ---- Foot-sticking sequences (computed from individually-scaled joints) -
    robot_height_a = float(getattr(task_constants_a, "ROBOT_HEIGHT", 1.78))
    robot_height_b = float(getattr(task_constants_b, "ROBOT_HEIGHT", 1.78))
    s_a, s_b, _ = IAMRRetargeter.compute_scale_factors(
        height_a, height_b, robot_height_a, robot_height_b
    )
    human_a_scaled = human_a_norm * s_a
    human_b_scaled = human_b_norm * s_b

    foot_a = extract_foot_sticking_sequence_velocity(
        human_a_scaled, demo_joints_a, toe_names_a
    )
    foot_b = extract_foot_sticking_sequence_velocity(
        human_b_scaled, demo_joints_b, toe_names_b
    )

    # ---- Load SMPL-X source rotations for J_rot (paper §3.2) --------------
    smplx_poses_a: np.ndarray | None = None
    smplx_poses_b: np.ndarray | None = None

    if cfg.raw_interx_dir is not None:
        logger.info(
            "Loading SMPL-X pose_body from %s/%s ...", cfg.raw_interx_dir, seq_id
        )
        smplx_poses_a, smplx_poses_b = _load_smplx_poses(
            seq_id=seq_id,
            raw_interx_dir=cfg.raw_interx_dir,
            n_frames_proc=human_a_norm.shape[0],
            fps_proc=int(fps),
        )
        logger.info(
            "SMPL-X poses loaded: A=%s  B=%s", smplx_poses_a.shape, smplx_poses_b.shape
        )
    else:
        logger.info(
            "raw_interx_dir not set — J_rot disabled (running without source rotations)."
        )

    # ---- Build dual scene XML ---------------------------------------------
    robot_xml_a = Path(str(task_constants_a.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
    robot_xml_b = Path(str(task_constants_b.ROBOT_URDF_FILE).replace(".urdf", ".xml"))
    dual_scene_xml = (
        cfg.dual_scene_xml
        if cfg.dual_scene_xml is not None
        else cfg.output_dir / f"{seq_id}_dual_scene.xml"
    )
    dual_scene_xml = _build_dual_scene_xml_from_pair(
        robot_xml_a=robot_xml_a,
        robot_xml_b=robot_xml_b,
        out_path=dual_scene_xml,
        prefix_a=cfg.dual_prefix_a,
        prefix_b=cfg.dual_prefix_b,
    )
    logger.info("Dual scene XML: %s", dual_scene_xml)

    # ---- Dense joint mapping (adds more Laplacian targets) ----------------
    dense_constants_a = _make_dual_dense_task_constants(
        task_constants_a, robot_cfg_a.robot_type
    )
    dense_constants_b = _make_dual_dense_task_constants(
        task_constants_b, robot_cfg_b.robot_type
    )
    logger.info(
        "Dense Laplacian targets — A: %d, B: %d",
        len(dense_constants_a.JOINTS_MAPPING),
        len(dense_constants_b.JOINTS_MAPPING),
    )

    # ---- Construct IAMR retargeter ----------------------------------------
    iamr = IAMRRetargeter(
        task_constants_a=dense_constants_a,
        task_constants_b=dense_constants_b,
        dual_scene_xml_path=str(dual_scene_xml),
        robot_a_prefix=cfg.dual_prefix_a,
        robot_b_prefix=cfg.dual_prefix_b,
        q_a_init_idx=cfg.retargeter.q_a_init_idx,
        q_b_init_idx=cfg.retargeter.q_a_init_idx,
        # Paper hyper-parameters
        w_self=cfg.w_self,
        w_inter=cfg.w_inter,
        w_reg=cfg.w_reg,
        lambda_rot=cfg.lambda_rot,
        omega_max=cfg.omega_max,
        gamma_decay=cfg.gamma_decay,
        # Constraint settings
        activate_foot_sticking=cfg.activate_foot_sticking,
        activate_joint_limits=cfg.activate_joint_limits,
        step_size=cfg.step_size,
        n_iter_first=cfg.n_iter_first,
        n_iter_other=cfg.n_iter_other,
        collision_detection_threshold=cfg.collision_detection_threshold,
        penetration_tolerance=cfg.penetration_tolerance,
        foot_sticking_tolerance=cfg.foot_sticking_tolerance,
    )

    # ---- Run IAMR ---------------------------------------------------------
    logger.info("Running IAMR retargeting ...")
    dest_raw = cfg.output_dir / f"{seq_id}_iamr_raw.npz"
    qpos_a, qpos_b, igraph, cgraph = iamr.retarget_motion(
        human_joints_a_raw=human_a_norm,
        human_joints_b_raw=human_b_norm,
        height_a=height_a,
        height_b=height_b,
        foot_sticking_sequences_a=foot_a,
        foot_sticking_sequences_b=foot_b,
        smplx_poses_a=smplx_poses_a,
        smplx_poses_b=smplx_poses_b,
        dest_res_path=str(dest_raw),
    )

    # ---- Save final output ------------------------------------------------
    final_path = cfg.output_dir / f"{seq_id}.npz"
    n = min(qpos_a.shape[0], qpos_b.shape[0])
    np.savez(
        final_path,
        sequence_id=seq_id,
        fps=fps,
        qpos_A=qpos_a[:n],
        qpos_B=qpos_b[:n],
        interaction_graph=igraph[:n],
        contact_graph=cgraph[:n],
        height_A=height_a,
        height_B=height_b,
        robot_A=robot_cfg_a.robot_type,
        robot_B=robot_cfg_b.robot_type,
        data_format_A=motion_cfg_a.data_format,
        data_format_B=motion_cfg_b.data_format,
        dual_scene_xml=str(dual_scene_xml),
        dual_prefix_A=cfg.dual_prefix_a,
        dual_prefix_B=cfg.dual_prefix_b,
        mode="iamr_dual",
        qpos_coordinate_frame="z_up",
        # IAMR hyper-parameters for reproducibility
        w_self=cfg.w_self,
        w_inter=cfg.w_inter,
        w_reg=cfg.w_reg,
        lambda_rot=cfg.lambda_rot,
        omega_max=cfg.omega_max,
        gamma_decay=cfg.gamma_decay,
    )
    logger.info("Saved IAMR output: %s", final_path)
    logger.info(
        "interaction_graph shape: %s | contact_graph shape: %s",
        igraph.shape, cgraph.shape,
    )


if __name__ == "__main__":
    main(tyro.cli(IAMRRetargetConfig))
