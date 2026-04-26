#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]

try:
    from scipy.spatial import Delaunay  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency at runtime
    Delaunay = None

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.path_utils import resolve_portable_path


# SMPL-X 22-joint body skeleton connectivity.
SMPLX22_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (12, 13),
    (12, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
]

SMPLX22_JOINT_NAMES: list[str] = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
]

OPTIMIZER_CROSS_CRITICAL_NAMES = {
    "L_Wrist",
    "R_Wrist",
    "L_Elbow",
    "R_Elbow",
    "L_Shoulder",
    "R_Shoulder",
}

OPTIMIZER_CROSS_CONTACT_NAMES = {
    "Pelvis",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "L_Collar",
    "R_Collar",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hip",
    "R_Hip",
}

# Right-handed y-up -> z-up coordinate transform.
# p_new = R_YUP_TO_ZUP @ p_old
R_YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)

# Quaternion for +90 deg rotation around x-axis in wxyz.
Q_YUP_TO_ZUP = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)

DEFAULT_URDF_BY_ROBOT: dict[str, str] = {
    "g1": "models/g1/g1_29dof.urdf",
    "t1": "models/t1/t1_23dof.urdf",
}


@dataclass
class Config:
    data_npz: str
    """Path to an Inter-X joint npz file (dual or single)."""

    port: int = 8080
    """Port for the Viser server."""

    prefer_optimization_joints: bool = True
    """Prefer the resized/preprocessed joints used by the optimizer when available."""

    joint_scale_mode: Literal["individual", "unified"] = "individual"
    """Choose whether to render individually-scaled optimizer joints (P_ind) or unified-scale joints (P_uni)."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from npz (default 30 if absent)."""

    frame_stride: int = 1
    """Subsample frames for faster playback."""

    point_size: float = 0.02
    """Joint marker size."""

    line_width: float = 2.0
    """Skeleton line width."""

    show_human_joints: bool = True
    """Render human joint point clouds."""

    show_skeleton: bool = True
    """Render skeleton edges in addition to joints."""

    show_optimizer_graph: bool = False
    """Render the dual retargeter's source graph edges on the first 22 SMPL-X joints."""

    optimizer_graph_line_width: float = 1.0
    """Line width for optimizer graph edges."""

    optimizer_max_source_edge_len: float = 0.45
    """Maximum intra-agent edge length, matching the dual optimizer default."""

    optimizer_cross_agent_rescue_k: int = 3
    """Fallback nearest-neighbor rescue count for cross-agent edges."""

    optimizer_cross_agent_contact_threshold: float = 0.4
    """Distance threshold for creating cross-agent contact edges."""

    optimizer_cross_agent_contact_persist_threshold: float | None = None
    """Persistence threshold for retaining cross-agent edges between frames."""

    optimizer_cross_agent_contact_k: int = 2
    """Maximum cross-agent neighbors to keep per source joint."""

    loop: bool = True
    """Loop playback."""

    y_up_to_z_up: bool = True
    """Convert incoming joints from y-up to z-up before rendering."""

    show_robots: bool = False
    """Render robot overlays if qpos is available."""

    robot_urdf: str | None = None
    """Deprecated shared URDF path used for both A and B when per-agent paths are not provided."""

    robot_urdf_a: str | None = None
    """URDF path for robot A overlay. If None, infer from npz metadata (robot_A) or fallback defaults."""

    robot_urdf_b: str | None = None
    """URDF path for robot B overlay. If None, infer from npz metadata (robot_B) or fallback defaults."""

    qpos_npz: str | None = None
    """Optional npz containing qpos_A/qpos_B. If None, try reading from data_npz."""


def _load_joint_trajectories(npz_path: Path, frame_stride: int) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    data = np.load(str(npz_path), allow_pickle=True)
    return _extract_joint_trajectories(data, npz_path.stem, frame_stride)


def _extract_joint_trajectories(
    data: np.lib.npyio.NpzFile,
    fallback_sequence_id: str,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    # Support both raw and processed dual naming.
    if "human_joints_A" in data and "human_joints_B" in data:
        joints_a = np.asarray(data["human_joints_A"], dtype=np.float32)
        joints_b = np.asarray(data["human_joints_B"], dtype=np.float32)
    elif "global_joint_positions_A" in data and "global_joint_positions_B" in data:
        joints_a = np.asarray(data["global_joint_positions_A"], dtype=np.float32)
        joints_b = np.asarray(data["global_joint_positions_B"], dtype=np.float32)
    # Support single-track files too.
    elif "human_joints" in data:
        joints_a = np.asarray(data["human_joints"], dtype=np.float32)
        joints_b = None
    elif "global_joint_positions" in data:
        joints_a = np.asarray(data["global_joint_positions"], dtype=np.float32)
        joints_b = None
    else:
        raise KeyError(
            "Unsupported npz format. Expected one of: "
            "human_joints_A/B, global_joint_positions_A/B, human_joints, global_joint_positions"
        )

    if joints_a.ndim != 3 or joints_a.shape[-1] != 3:
        raise ValueError(f"joints_a must have shape (T, J, 3), got {joints_a.shape}")
    if joints_b is not None and (joints_b.ndim != 3 or joints_b.shape[-1] != 3):
        raise ValueError(f"joints_b must have shape (T, J, 3), got {joints_b.shape}")

    stride = max(1, int(frame_stride))
    joints_a = joints_a[::stride]
    if joints_b is not None:
        joints_b = joints_b[::stride]
        n = min(joints_a.shape[0], joints_b.shape[0])
        joints_a = joints_a[:n]
        joints_b = joints_b[:n]

    fps = float(data["fps"]) if "fps" in data else 30.0
    if stride > 1:
        fps = fps / stride
    sequence_id = str(data["sequence_id"]) if "sequence_id" in data else fallback_sequence_id
    return joints_a, joints_b, fps, sequence_id


def _coerce_scalar_string(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    text = str(value).strip()
    return text or None


def _coerce_scalar_float(data: np.lib.npyio.NpzFile | None, key: str) -> float | None:
    if data is None or key not in data:
        return None
    return float(np.asarray(data[key]).item())


def _resolve_joint_scales(
    *,
    data_npz: np.lib.npyio.NpzFile,
    qpos_data: np.lib.npyio.NpzFile | None,
    mode: Literal["individual", "unified"],
    has_b: bool,
) -> tuple[float, float | None, str]:
    if mode == "individual":
        scale_a = _coerce_scalar_float(qpos_data, "scale_A")
        if scale_a is None:
            scale_a = _coerce_scalar_float(data_npz, "scale_A")
        scale_b = _coerce_scalar_float(qpos_data, "scale_B")
        if scale_b is None:
            scale_b = _coerce_scalar_float(data_npz, "scale_B")
        if scale_a is None or (has_b and scale_b is None):
            raise ValueError(
                "joint_scale_mode='individual' requires scale_A/scale_B in either --data_npz or --qpos_npz."
            )
        return scale_a, scale_b, "P_ind"

    scale_uni = _coerce_scalar_float(qpos_data, "scale_uni")
    if scale_uni is None:
        scale_uni = _coerce_scalar_float(data_npz, "scale_uni")
    if scale_uni is None:
        raise ValueError(
            "joint_scale_mode='unified' requires scale_uni in either --data_npz or --qpos_npz."
        )
    return scale_uni, scale_uni if has_b else None, "P_uni"


def _resolve_motion_metadata(
    *,
    data_npz: np.lib.npyio.NpzFile,
    qpos_data: np.lib.npyio.NpzFile | None,
    suffix: str,
) -> tuple[str, str]:
    data_format = _coerce_scalar_string(qpos_data[f"data_format_{suffix}"]) if qpos_data is not None and f"data_format_{suffix}" in qpos_data else None
    if data_format is None and f"data_format_{suffix}" in data_npz:
        data_format = _coerce_scalar_string(data_npz[f"data_format_{suffix}"])
    robot = _coerce_scalar_string(qpos_data[f"robot_{suffix}"]) if qpos_data is not None and f"robot_{suffix}" in qpos_data else None
    if robot is None and f"robot_{suffix}" in data_npz:
        robot = _coerce_scalar_string(data_npz[f"robot_{suffix}"])
    return data_format or "smplx", robot or "g1"


def _normalize_height_and_scale(
    joints: np.ndarray,
    *,
    demo_joints: list[str],
    toe_names: list[str],
    scale: float,
    mat_height: float = 0.1,
) -> np.ndarray:
    joints_scaled = np.array(joints, dtype=np.float32, copy=True)
    toe_indices = [demo_joints.index(name) for name in toe_names if name in demo_joints]
    if toe_indices:
        z_min = float(joints_scaled[:, toe_indices, 2].min())
        if z_min >= mat_height:
            z_min -= mat_height
        joints_scaled[:, :, 2] -= z_min
    joints_scaled *= float(scale)
    return joints_scaled


def _reconstruct_optimizer_joints_from_scales(
    data_joints_a: np.ndarray,
    data_joints_b: np.ndarray | None,
    data_npz: np.lib.npyio.NpzFile,
    qpos_data: np.lib.npyio.NpzFile | None,
    *,
    raw_y_up_to_z_up: bool,
    joint_scale_mode: Literal["individual", "unified"],
) -> tuple[np.ndarray, np.ndarray | None, str]:
    scale_a, scale_b, scale_label = _resolve_joint_scales(
        data_npz=data_npz,
        qpos_data=qpos_data,
        mode=joint_scale_mode,
        has_b=data_joints_b is not None,
    )

    joints_a_input = _convert_y_up_to_z_up_points(data_joints_a) if raw_y_up_to_z_up else data_joints_a
    data_format_a, robot_a = _resolve_motion_metadata(data_npz=data_npz, qpos_data=qpos_data, suffix="A")
    motion_cfg_a = MotionDataConfig(data_format=data_format_a or "smplx", robot_type=robot_a or "g1")
    joints_a = _normalize_height_and_scale(
        joints_a_input,
        demo_joints=motion_cfg_a.resolved_demo_joints,
        toe_names=motion_cfg_a.toe_names,
        scale=scale_a,
    )

    joints_b: np.ndarray | None = None
    if data_joints_b is not None:
        if scale_b is None:
            raise ValueError(f"Missing scale for track B in joint_scale_mode='{joint_scale_mode}'.")
        joints_b_input = _convert_y_up_to_z_up_points(data_joints_b) if raw_y_up_to_z_up else data_joints_b
        data_format_b, robot_b = _resolve_motion_metadata(data_npz=data_npz, qpos_data=qpos_data, suffix="B")
        motion_cfg_b = MotionDataConfig(data_format=data_format_b or "smplx", robot_type=robot_b or "g1")
        joints_b = _normalize_height_and_scale(
            joints_b_input,
            demo_joints=motion_cfg_b.resolved_demo_joints,
            toe_names=motion_cfg_b.toe_names,
            scale=scale_b,
        )

    return joints_a, joints_b, scale_label


def _load_render_joint_trajectories(
    data_npz_path: Path,
    qpos_npz_path: Path | None,
    frame_stride: int,
    prefer_optimization_joints: bool,
    raw_y_up_to_z_up: bool,
    joint_scale_mode: Literal["individual", "unified"],
) -> tuple[np.ndarray, np.ndarray | None, float, str, str | None, str]:
    data_npz = np.load(str(data_npz_path), allow_pickle=True)
    qpos_data = np.load(str(qpos_npz_path), allow_pickle=True) if qpos_npz_path is not None else None
    data_joint_coordinate_frame = (
        _parse_coordinate_frame(data_npz["qpos_coordinate_frame"])
        if "qpos_coordinate_frame" in data_npz
        else ("z_up" if "scale_uni" in data_npz else None)
    )

    if prefer_optimization_joints and qpos_data is not None:
        try:
            joints_a, joints_b, fps, sequence_id = _extract_joint_trajectories(qpos_data, qpos_npz_path.stem, frame_stride)
            try:
                scale_a, scale_b, scale_label = _resolve_joint_scales(
                    data_npz=data_npz,
                    qpos_data=qpos_data,
                    mode=joint_scale_mode,
                    has_b=joints_b is not None,
                )
                joints_a = joints_a * scale_a
                if joints_b is not None and scale_b is not None:
                    joints_b = joints_b * scale_b
                source = f"qpos_npz:{scale_label}"
            except ValueError:
                if joint_scale_mode != "individual":
                    raise
                source = "qpos_npz:processed_joints"
            joint_coordinate_frame = (
                _parse_coordinate_frame(qpos_data["qpos_coordinate_frame"])
                if "qpos_coordinate_frame" in qpos_data
                else None
            )
            return joints_a, joints_b, fps, sequence_id, joint_coordinate_frame, source
        except KeyError:
            pass

    data_joints_a, data_joints_b, fps, sequence_id = _extract_joint_trajectories(data_npz, data_npz_path.stem, frame_stride)

    if prefer_optimization_joints and qpos_data is not None:
        joints_a, joints_b, scale_label = _reconstruct_optimizer_joints_from_scales(
            data_joints_a,
            data_joints_b,
            data_npz,
            qpos_data,
            raw_y_up_to_z_up=raw_y_up_to_z_up and data_joint_coordinate_frame != "z_up",
            joint_scale_mode=joint_scale_mode,
        )
        joint_coordinate_frame = (
            _parse_coordinate_frame(qpos_data["qpos_coordinate_frame"])
            if "qpos_coordinate_frame" in qpos_data
            else ("z_up" if raw_y_up_to_z_up else None)
        )
        return joints_a, joints_b, fps, sequence_id, joint_coordinate_frame, f"data_npz+qpos_scales:{scale_label}"

    return data_joints_a, data_joints_b, fps, sequence_id, data_joint_coordinate_frame, "data_npz:raw_joints"


def _edge_segments(joints: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[joints[i], joints[j]] for i, j in edges], dtype=np.float32)


def _empty_line_segments() -> np.ndarray:
    return np.zeros((0, 2, 3), dtype=np.float32)


def _line_segment_colors(num_segments: int, color: tuple[int, int, int]) -> np.ndarray:
    if num_segments <= 0:
        return np.zeros((0, 2, 3), dtype=np.uint8)
    return np.tile(np.array([[color, color]], dtype=np.uint8), (num_segments, 1, 1))


def _combined_edge_segments(
    joints_a: np.ndarray,
    joints_b: np.ndarray | None,
    edges: list[tuple[int, int]],
    *,
    num_source_a: int,
) -> np.ndarray:
    if joints_b is None or not edges:
        return _empty_line_segments()
    combined = np.vstack([joints_a[:num_source_a], joints_b[:num_source_a]])
    return np.asarray([[combined[i], combined[j]] for i, j in edges], dtype=np.float32)


def _delaunay_tetrahedra(vertices: np.ndarray) -> np.ndarray:
    if Delaunay is not None:
        try:
            return Delaunay(vertices).simplices
        except Exception:
            pass

    n = vertices.shape[0]
    if n < 4:
        raise ValueError(f"Need at least 4 vertices to build optimizer graph tetrahedra, got {n}")
    distances = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    np.fill_diagonal(distances, np.inf)
    simplices: list[list[int]] = []
    for i in range(n):
        nbrs = np.argsort(distances[i])[:3]
        simplices.append([i, int(nbrs[0]), int(nbrs[1]), int(nbrs[2])])
    return np.asarray(simplices, dtype=int)


def _tetrahedra_to_edges(tetrahedra: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for tet in tetrahedra:
        a, b, c, d = [int(x) for x in tet]
        for i, j in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
            edges.add((i, j) if i < j else (j, i))
    return edges


def _prune_long_edges(vertices: np.ndarray, edges: set[tuple[int, int]], max_len: float) -> set[tuple[int, int]]:
    kept: set[tuple[int, int]] = set()
    for i, j in edges:
        if np.linalg.norm(vertices[i] - vertices[j]) <= max_len:
            kept.add((i, j))
    return kept


def _build_intra_agent_optimizer_edges(vertices: np.ndarray, *, max_len: float, offset: int = 0) -> set[tuple[int, int]]:
    tetrahedra = _delaunay_tetrahedra(vertices)
    edges = _prune_long_edges(vertices, _tetrahedra_to_edges(tetrahedra), max_len)
    if offset == 0:
        return edges
    return {(i + offset, j + offset) for i, j in edges}


def _optimizer_cross_indices(num_source_a: int) -> tuple[list[int], list[int], list[int], list[int]]:
    source_joint_names_a = SMPLX22_JOINT_NAMES[:num_source_a]
    source_joint_names_b = SMPLX22_JOINT_NAMES[:num_source_a]
    source_name_to_idx_a = {name: i for i, name in enumerate(source_joint_names_a)}
    source_name_to_idx_b = {name: i for i, name in enumerate(source_joint_names_b)}

    cross_rescue_a = [source_name_to_idx_a[name] for name in OPTIMIZER_CROSS_CRITICAL_NAMES if name in source_name_to_idx_a]
    cross_rescue_b = [
        num_source_a + source_name_to_idx_b[name] for name in OPTIMIZER_CROSS_CRITICAL_NAMES if name in source_name_to_idx_b
    ]
    cross_contact_a = [source_name_to_idx_a[name] for name in OPTIMIZER_CROSS_CONTACT_NAMES if name in source_name_to_idx_a]
    cross_contact_b = [
        num_source_a + source_name_to_idx_b[name] for name in OPTIMIZER_CROSS_CONTACT_NAMES if name in source_name_to_idx_b
    ]
    if not cross_contact_a:
        cross_contact_a = list(range(num_source_a))
    if not cross_contact_b:
        cross_contact_b = list(range(num_source_a, 2 * num_source_a))
    return cross_rescue_a, cross_rescue_b, cross_contact_a, cross_contact_b


def _build_cross_agent_optimizer_edges(
    vertices: np.ndarray,
    *,
    num_source_a: int,
    prev_cross_edges: tuple[tuple[int, int], ...],
    threshold: float,
    persist_threshold: float,
    contact_k: int,
    rescue_k: int,
) -> set[tuple[int, int]]:
    cross_rescue_a, cross_rescue_b, cross_contact_a, cross_contact_b = _optimizer_cross_indices(num_source_a)
    a_ids = np.asarray(cross_contact_a, dtype=int)
    b_ids = np.asarray(cross_contact_b, dtype=int)
    if a_ids.size == 0 or b_ids.size == 0:
        return set()

    dmat = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
    edges: set[tuple[int, int]] = set()

    def _add_knn(src_ids: np.ndarray, dst_ids: np.ndarray, threshold_value: float) -> None:
        for idx in src_ids:
            distances = dmat[idx, dst_ids]
            keep_mask = distances <= threshold_value
            if not np.any(keep_mask):
                continue
            valid_dst = dst_ids[keep_mask]
            valid_dist = distances[keep_mask]
            order = np.argsort(valid_dist)[:contact_k]
            for dst in valid_dst[order]:
                i0, j0 = (int(idx), int(dst)) if int(idx) < int(dst) else (int(dst), int(idx))
                edges.add((i0, j0))

    _add_knn(a_ids, b_ids, threshold)
    _add_knn(b_ids, a_ids, threshold)

    for i, j in prev_cross_edges:
        if i < 0 or j < 0 or i >= vertices.shape[0] or j >= vertices.shape[0]:
            continue
        if dmat[i, j] <= persist_threshold:
            edges.add((i, j) if i < j else (j, i))

    def _incident_cross_count(idx: int) -> int:
        return sum(1 for i, j in edges if i == idx or j == idx)

    for idx in cross_rescue_a:
        if _incident_cross_count(idx) > 0:
            continue
        order = np.argsort(dmat[idx, b_ids])[:rescue_k]
        for nearest in b_ids[order]:
            nearest = int(nearest)
            if dmat[idx, nearest] <= persist_threshold:
                edges.add((idx, nearest) if idx < nearest else (nearest, idx))

    for idx in cross_rescue_b:
        if _incident_cross_count(idx) > 0:
            continue
        order = np.argsort(dmat[idx, a_ids])[:rescue_k]
        for nearest in a_ids[order]:
            nearest = int(nearest)
            if dmat[idx, nearest] <= persist_threshold:
                edges.add((nearest, idx) if nearest < idx else (idx, nearest))

    return edges


def _precompute_optimizer_graph_segments(
    joints_a: np.ndarray,
    joints_b: np.ndarray | None,
    cfg: Config,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    n_frames = joints_a.shape[0]
    num_source_a = min(22, joints_a.shape[1])
    if num_source_a < 22:
        raise ValueError(
            "Optimizer graph overlay requires at least 22 joints in track A to match the dual retargeter source graph."
        )
    if joints_b is not None and joints_b.shape[1] < 22:
        raise ValueError(
            "Optimizer graph overlay requires at least 22 joints in track B to match the dual retargeter source graph."
        )

    persist_threshold = (
        float(cfg.optimizer_cross_agent_contact_persist_threshold)
        if cfg.optimizer_cross_agent_contact_persist_threshold is not None
        else max(
            float(cfg.optimizer_cross_agent_contact_threshold) * 1.25,
            float(cfg.optimizer_cross_agent_contact_threshold) + 0.08,
        )
    )

    a_segments_by_frame: list[np.ndarray] = []
    b_segments_by_frame: list[np.ndarray] = []
    cross_segments_by_frame: list[np.ndarray] = []
    prev_cross_edges: tuple[tuple[int, int], ...] = tuple()

    for frame_idx in range(n_frames):
        points_a = joints_a[frame_idx, :num_source_a]
        intra_edges_a = sorted(
            _build_intra_agent_optimizer_edges(
                points_a,
                max_len=float(cfg.optimizer_max_source_edge_len),
            )
        )
        a_segments_by_frame.append(_edge_segments(points_a, intra_edges_a) if intra_edges_a else _empty_line_segments())

        if joints_b is None:
            b_segments_by_frame.append(_empty_line_segments())
            cross_segments_by_frame.append(_empty_line_segments())
            continue

        points_b = joints_b[frame_idx, :num_source_a]
        intra_edges_b = sorted(
            _build_intra_agent_optimizer_edges(
                points_b,
                max_len=float(cfg.optimizer_max_source_edge_len),
            )
        )
        b_segments_by_frame.append(_edge_segments(points_b, intra_edges_b) if intra_edges_b else _empty_line_segments())

        combined = np.vstack([points_a, points_b])
        cross_edges = sorted(
            _build_cross_agent_optimizer_edges(
                combined,
                num_source_a=num_source_a,
                prev_cross_edges=prev_cross_edges,
                threshold=float(cfg.optimizer_cross_agent_contact_threshold),
                persist_threshold=persist_threshold,
                contact_k=max(1, int(cfg.optimizer_cross_agent_contact_k)),
                rescue_k=max(1, int(cfg.optimizer_cross_agent_rescue_k)),
            )
        )
        prev_cross_edges = tuple(cross_edges)
        cross_segments_by_frame.append(
            _combined_edge_segments(points_a, points_b, cross_edges, num_source_a=num_source_a)
            if cross_edges
            else _empty_line_segments()
        )

    return a_segments_by_frame, b_segments_by_frame, cross_segments_by_frame


def _convert_y_up_to_z_up_points(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    converted = (R_YUP_TO_ZUP @ flat.T).T
    return converted.reshape(points.shape).astype(np.float32, copy=False)


def _parse_coordinate_frame(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    frame = str(value).strip().lower()
    if not frame:
        return None
    if frame not in {"y_up", "z_up"}:
        raise ValueError(f"Unsupported coordinate frame metadata: {frame}")
    return frame


def _load_qpos_tracks(npz_path: Path, frame_stride: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    data = np.load(str(npz_path), allow_pickle=True)
    qpos_frame = _parse_coordinate_frame(data["qpos_coordinate_frame"]) if "qpos_coordinate_frame" in data else None
    qpos_a = np.asarray(data["qpos_A"], dtype=np.float32)[:: max(1, int(frame_stride))] if "qpos_A" in data else None
    qpos_b = np.asarray(data["qpos_B"], dtype=np.float32)[:: max(1, int(frame_stride))] if "qpos_B" in data else None
    if qpos_a is not None and qpos_b is not None:
        n = min(qpos_a.shape[0], qpos_b.shape[0])
        qpos_a = qpos_a[:n]
        qpos_b = qpos_b[:n]
    return qpos_a, qpos_b, qpos_frame


def _infer_robot_type(data: np.lib.npyio.NpzFile, key: str) -> str | None:
    if key not in data:
        return None
    value = data[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    robot_type = str(value).strip().lower()
    if not robot_type:
        return None
    return robot_type.split("_")[0]


def _resolve_target_types(
    data_npz_path: Path,
    qpos_npz_path: Path,
) -> tuple[str, str]:
    sources: list[np.lib.npyio.NpzFile] = []
    seen: set[Path] = set()
    for path in (qpos_npz_path, data_npz_path):
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        sources.append(np.load(str(path), allow_pickle=True))

    def _read_target(key: str) -> str | None:
        for data in sources:
            if key not in data:
                continue
            value = _coerce_scalar_string(data[key])
            if value is not None:
                return value.lower()
        return None

    qpos_a_present = any("qpos_A" in data for data in sources)
    qpos_b_present = any("qpos_B" in data for data in sources)
    target_a = _read_target("target_A") or ("robot" if qpos_a_present else "human")
    target_b = _read_target("target_B") or ("robot" if qpos_b_present else "human")
    return target_a, target_b


def _resolve_robot_urdf_paths(
    cfg: Config,
    qpos_npz_path: Path,
    *,
    include_a: bool,
    include_b: bool,
) -> tuple[Path | None, Path | None]:
    data = np.load(str(qpos_npz_path), allow_pickle=True)
    robot_type_a = _infer_robot_type(data, "robot_A")
    robot_type_b = _infer_robot_type(data, "robot_B")

    urdf_a_path: Path | None = None
    if include_a:
        urdf_a = cfg.robot_urdf_a or cfg.robot_urdf
        if urdf_a is None:
            urdf_a = DEFAULT_URDF_BY_ROBOT.get(robot_type_a or "", DEFAULT_URDF_BY_ROBOT["g1"])
        urdf_a_path = resolve_portable_path(urdf_a, must_exist=True)

    urdf_b_path: Path | None = None
    if include_b:
        urdf_b = cfg.robot_urdf_b or cfg.robot_urdf
        if urdf_b is None:
            fallback = str(urdf_a_path) if urdf_a_path is not None else DEFAULT_URDF_BY_ROBOT["g1"]
            urdf_b = DEFAULT_URDF_BY_ROBOT.get(robot_type_b or "", fallback)
        urdf_b_path = resolve_portable_path(urdf_b, must_exist=True)

    return urdf_a_path, urdf_b_path


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32, copy=False)


def _convert_y_up_to_z_up_qpos(qpos: np.ndarray) -> np.ndarray:
    out = np.array(qpos, dtype=np.float32, copy=True)
    out[:, :3] = (R_YUP_TO_ZUP @ out[:, :3].T).T
    for i in range(out.shape[0]):
        q_old = _normalize_quat_wxyz(out[i, 3:7])
        # Basis change for world-frame orientation under v_new = R * v_old:
        # orientation_new = Q_basis * orientation_old
        q_new = _quat_mul_wxyz(Q_YUP_TO_ZUP, q_old)
        out[i, 3:7] = _normalize_quat_wxyz(q_new)
    return out


def main(cfg: Config) -> None:
    npz_path = Path(cfg.data_npz)
    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    joints_a, joints_b, fps_from_data, sequence_id, joint_coordinate_frame, joint_source = _load_render_joint_trajectories(
        data_npz_path=npz_path,
        qpos_npz_path=qpos_npz_path if qpos_npz_path.exists() else None,
        frame_stride=cfg.frame_stride,
        prefer_optimization_joints=cfg.prefer_optimization_joints,
        raw_y_up_to_z_up=bool(cfg.y_up_to_z_up),
        joint_scale_mode=cfg.joint_scale_mode,
    )
    should_convert_joints = bool(cfg.y_up_to_z_up)
    if joint_coordinate_frame == "z_up":
        should_convert_joints = False
    if should_convert_joints:
        joints_a = _convert_y_up_to_z_up_points(joints_a)
        if joints_b is not None:
            joints_b = _convert_y_up_to_z_up_points(joints_b)
    fps = float(cfg.fps_override) if cfg.fps_override is not None else fps_from_data
    fps = max(1e-6, fps)

    n_frames, n_joints, _ = joints_a.shape
    print(f"[dual_joint_renderer] sequence={sequence_id} frames={n_frames} joints={n_joints} fps={fps:.2f}")
    print(f"[dual_joint_renderer] rendering {'dual' if joints_b is not None else 'single'} tracks")
    print(
        "[dual_joint_renderer] joint_source="
        f"{joint_source} | joint_coordinate_frame={joint_coordinate_frame or 'assumed_y_up'}"
        f" | joints_y_up_to_z_up_applied={should_convert_joints}"
    )

    optimizer_graph_a = None
    optimizer_graph_b = None
    optimizer_graph_cross = None
    if cfg.show_optimizer_graph:
        optimizer_graph_a, optimizer_graph_b, optimizer_graph_cross = _precompute_optimizer_graph_segments(
            joints_a=joints_a,
            joints_b=joints_b,
            cfg=cfg,
        )
        cross_counts = [segments.shape[0] for segments in optimizer_graph_cross]
        print(
            "[dual_joint_renderer] optimizer_graph="
            f"enabled | frames={len(optimizer_graph_a)}"
            f" | mean_cross_edges={float(np.mean(cross_counts)) if cross_counts else 0.0:.2f}"
            f" | max_cross_edges={max(cross_counts) if cross_counts else 0}"
        )

    qpos_a, qpos_b, qpos_coordinate_frame = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    target_a, target_b = _resolve_target_types(npz_path, qpos_npz_path)
    show_robot_a = bool(cfg.show_robots and target_a != "human")
    show_robot_b = bool(cfg.show_robots and joints_b is not None and target_b != "human")
    print(f"[dual_joint_renderer] target_A={target_a} | target_B={target_b}")
    if cfg.show_robots:
        if show_robot_a and qpos_a is None:
            raise ValueError(
                f"show_robots=True but no qpos_A found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a retarget output with qpos_A."
            )
        if show_robot_b and qpos_b is None:
            raise ValueError(
                f"show_robots=True but no qpos_B found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a retarget output with qpos_B."
            )
        should_convert_qpos = bool(cfg.y_up_to_z_up)
        if qpos_coordinate_frame == "z_up":
            should_convert_qpos = False
        if should_convert_qpos:
            if show_robot_a and qpos_a is not None:
                qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
            if show_robot_b and qpos_b is not None:
                qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)
        print(
            "[dual_joint_renderer] qpos_coordinate_frame="
            f"{qpos_coordinate_frame or 'legacy_assumed_y_up'} | qpos_y_up_to_z_up_applied={should_convert_qpos}"
        )

    server = viser.ViserServer(port=cfg.port)
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))

    # Human A (cyan)
    a_joints_handle = server.scene.add_point_cloud(
        "/humans/A/joints",
        points=joints_a[0],
        colors=np.tile(np.array([[0, 200, 255]], dtype=np.uint8), (n_joints, 1)),
        point_size=cfg.point_size,
        point_shape="circle",
    )
    a_joints_handle.visible = bool(cfg.show_human_joints)
    a_lines_handle = None
    if cfg.show_skeleton and n_joints >= 22:
        a_lines_handle = server.scene.add_line_segments(
            "/humans/A/skeleton",
            points=_edge_segments(joints_a[0], SMPLX22_EDGES),
            colors=_line_segment_colors(len(SMPLX22_EDGES), (0, 180, 230)),
            line_width=cfg.line_width,
        )
    a_optimizer_handle = None
    if optimizer_graph_a is not None:
        a_optimizer_handle = server.scene.add_line_segments(
            "/humans/A/optimizer_graph",
            points=optimizer_graph_a[0],
            colors=_line_segment_colors(optimizer_graph_a[0].shape[0], (100, 230, 255)),
            line_width=float(cfg.optimizer_graph_line_width),
            visible=cfg.show_optimizer_graph,
        )

    # Human B (orange), optional.
    b_joints_handle = None
    b_lines_handle = None
    if joints_b is not None:
        b_joints_handle = server.scene.add_point_cloud(
            "/humans/B/joints",
            points=joints_b[0],
            colors=np.tile(np.array([[255, 160, 0]], dtype=np.uint8), (n_joints, 1)),
            point_size=cfg.point_size,
            point_shape="circle",
        )
        b_joints_handle.visible = bool(cfg.show_human_joints)
        if cfg.show_skeleton and n_joints >= 22:
            b_lines_handle = server.scene.add_line_segments(
                "/humans/B/skeleton",
                points=_edge_segments(joints_b[0], SMPLX22_EDGES),
                colors=_line_segment_colors(len(SMPLX22_EDGES), (230, 140, 0)),
                line_width=cfg.line_width,
            )
        if optimizer_graph_b is not None:
            b_optimizer_handle = server.scene.add_line_segments(
                "/humans/B/optimizer_graph",
                points=optimizer_graph_b[0],
                colors=_line_segment_colors(optimizer_graph_b[0].shape[0], (255, 190, 70)),
                line_width=float(cfg.optimizer_graph_line_width),
                visible=cfg.show_optimizer_graph,
            )
        else:
            b_optimizer_handle = None
        if optimizer_graph_cross is not None:
            cross_optimizer_handle = server.scene.add_line_segments(
                "/humans/cross_optimizer_graph",
                points=optimizer_graph_cross[0],
                colors=_line_segment_colors(optimizer_graph_cross[0].shape[0], (255, 90, 90)),
                line_width=float(cfg.optimizer_graph_line_width),
                visible=cfg.show_optimizer_graph,
            )
        else:
            cross_optimizer_handle = None
    else:
        b_optimizer_handle = None
        cross_optimizer_handle = None

    # Optional robot overlays (A and B) from qpos_A/qpos_B.
    robot_a = None
    robot_b = None
    robot_a_root = None
    robot_b_root = None
    robot_dof_a = None
    robot_dof_b = None
    if show_robot_a or show_robot_b:
        urdf_a_path, urdf_b_path = _resolve_robot_urdf_paths(
            cfg,
            qpos_npz_path,
            include_a=show_robot_a,
            include_b=show_robot_b,
        )
        if show_robot_a:
            if urdf_a_path is None:
                raise ValueError("Could not resolve robot_urdf for A.")
            robot_urdf_a = yourdfpy.URDF.load(str(urdf_a_path), load_meshes=True, build_scene_graph=True)
            robot_a_root = server.scene.add_frame("/robots/A", show_axes=False)
            robot_a = ViserUrdf(server, urdf_or_path=robot_urdf_a, root_node_name="/robots/A")
            robot_a.show_visual = True
            robot_dof_a = len(robot_a.get_actuated_joint_limits())

        if show_robot_b:
            if urdf_b_path is None:
                raise ValueError("Could not resolve robot_urdf for B.")
            robot_urdf_b = yourdfpy.URDF.load(str(urdf_b_path), load_meshes=True, build_scene_graph=True)
            robot_b_root = server.scene.add_frame("/robots/B", show_axes=False)
            robot_b = ViserUrdf(server, urdf_or_path=robot_urdf_b, root_node_name="/robots/B")
            robot_b.show_visual = True
            robot_dof_b = len(robot_b.get_actuated_joint_limits())

        robot_overlay_parts: list[str] = ["[dual_joint_renderer] robot overlay enabled"]
        if show_robot_a:
            robot_overlay_parts.append(f"urdf_A={urdf_a_path} | dof_A={robot_dof_a}")
        if show_robot_b:
            robot_overlay_parts.append(f"urdf_B={urdf_b_path} | dof_B={robot_dof_b}")
        print(" | ".join(robot_overlay_parts))

        # Clamp timeline to available qpos frames.
        if show_robot_a and qpos_a is not None:
            n_frames = min(n_frames, qpos_a.shape[0])
        if show_robot_b and qpos_b is not None:
            n_frames = min(n_frames, qpos_b.shape[0])

    with server.gui.add_folder("Playback"):
        playing_cb = server.gui.add_checkbox("Playing", initial_value=True)
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        fps_in = server.gui.add_number("FPS", initial_value=float(fps), min=1.0, max=240.0, step=1.0)

    with server.gui.add_folder("Display"):
        point_size_slider = server.gui.add_slider(
            "Point size",
            min=0.005,
            max=0.08,
            step=0.001,
            initial_value=cfg.point_size,
        )
        show_human_joints_cb = server.gui.add_checkbox(
            "Show human joints",
            initial_value=cfg.show_human_joints,
        )
        a_joints_handle.visible = bool(show_human_joints_cb.value)
        if b_joints_handle is not None:
            b_joints_handle.visible = bool(show_human_joints_cb.value)

        @show_human_joints_cb.on_update
        def _(_event) -> None:
            a_joints_handle.visible = bool(show_human_joints_cb.value)
            if b_joints_handle is not None:
                b_joints_handle.visible = bool(show_human_joints_cb.value)

        show_skeleton_cb = server.gui.add_checkbox("Show skeleton", initial_value=cfg.show_skeleton)
        if a_lines_handle is not None:
            a_lines_handle.visible = bool(show_skeleton_cb.value)
        if b_lines_handle is not None:
            b_lines_handle.visible = bool(show_skeleton_cb.value)

        @show_skeleton_cb.on_update
        def _(_event) -> None:
            if a_lines_handle is not None:
                a_lines_handle.visible = bool(show_skeleton_cb.value)
            if b_lines_handle is not None:
                b_lines_handle.visible = bool(show_skeleton_cb.value)

        if optimizer_graph_a is not None:
            show_optimizer_graph_cb = server.gui.add_checkbox(
                "Show optimizer graph",
                initial_value=cfg.show_optimizer_graph,
            )
            optimizer_line_width_slider = server.gui.add_slider(
                "Optimizer line width",
                min=0.25,
                max=6.0,
                step=0.05,
                initial_value=float(cfg.optimizer_graph_line_width),
            )

            if a_optimizer_handle is not None:
                a_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)
            if b_optimizer_handle is not None:
                b_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)
            if cross_optimizer_handle is not None:
                cross_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)

            @show_optimizer_graph_cb.on_update
            def _(_event) -> None:
                if a_optimizer_handle is not None:
                    a_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)
                if b_optimizer_handle is not None:
                    b_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)
                if cross_optimizer_handle is not None:
                    cross_optimizer_handle.visible = bool(show_optimizer_graph_cb.value)

        if show_robot_a or show_robot_b:
            show_robot_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=True)

            @show_robot_meshes_cb.on_update
            def _(_event) -> None:
                if robot_a is not None:
                    robot_a.show_visual = bool(show_robot_meshes_cb.value)
                if robot_b is not None:
                    robot_b.show_visual = bool(show_robot_meshes_cb.value)

    def _apply_frame(idx: int) -> None:
        i = int(np.clip(idx, 0, n_frames - 1))
        a_joints_handle.points = joints_a[i]
        a_joints_handle.point_size = float(point_size_slider.value)
        if a_lines_handle is not None:
            a_lines_handle.points = _edge_segments(joints_a[i], SMPLX22_EDGES)
            a_lines_handle.line_width = float(cfg.line_width)
        if a_optimizer_handle is not None and optimizer_graph_a is not None:
            a_optimizer_handle.points = optimizer_graph_a[i]
            a_optimizer_handle.colors = _line_segment_colors(optimizer_graph_a[i].shape[0], (100, 230, 255))
            a_optimizer_handle.line_width = float(optimizer_line_width_slider.value)
        if joints_b is not None and b_joints_handle is not None:
            b_joints_handle.points = joints_b[i]
            b_joints_handle.point_size = float(point_size_slider.value)
        if joints_b is not None and b_lines_handle is not None:
            b_lines_handle.points = _edge_segments(joints_b[i], SMPLX22_EDGES)
            b_lines_handle.line_width = float(cfg.line_width)
        if joints_b is not None and b_optimizer_handle is not None and optimizer_graph_b is not None:
            b_optimizer_handle.points = optimizer_graph_b[i]
            b_optimizer_handle.colors = _line_segment_colors(optimizer_graph_b[i].shape[0], (255, 190, 70))
            b_optimizer_handle.line_width = float(optimizer_line_width_slider.value)
        if cross_optimizer_handle is not None and optimizer_graph_cross is not None:
            cross_optimizer_handle.points = optimizer_graph_cross[i]
            cross_optimizer_handle.colors = _line_segment_colors(optimizer_graph_cross[i].shape[0], (255, 90, 90))
            cross_optimizer_handle.line_width = float(optimizer_line_width_slider.value)

        if cfg.show_robots and qpos_a is not None and robot_a is not None and robot_a_root is not None and robot_dof_a is not None:
            qa = qpos_a[i]
            if qa.shape[0] < 7 + robot_dof_a:
                raise ValueError(f"qpos_A frame has width {qa.shape[0]}, expected at least {7 + robot_dof_a}")
            robot_a_root.position = qa[0:3]
            robot_a_root.wxyz = qa[3:7]
            robot_a.update_cfg(qa[7 : 7 + robot_dof_a])
        if (
            cfg.show_robots
            and joints_b is not None
            and qpos_b is not None
            and robot_b is not None
            and robot_b_root is not None
            and robot_dof_b is not None
        ):
            qb = qpos_b[i]
            if qb.shape[0] < 7 + robot_dof_b:
                raise ValueError(f"qpos_B frame has width {qb.shape[0]}, expected at least {7 + robot_dof_b}")
            robot_b_root.position = qb[0:3]
            robot_b_root.wxyz = qb[3:7]
            robot_b.update_cfg(qb[7 : 7 + robot_dof_b])

    @frame_slider.on_update
    def _(_event) -> None:
        _apply_frame(int(frame_slider.value))

    _apply_frame(0)

    def _player_loop() -> None:
        while True:
            if bool(playing_cb.value):
                next_idx = int(frame_slider.value) + 1
                if next_idx >= n_frames:
                    if cfg.loop:
                        next_idx = 0
                    else:
                        next_idx = n_frames - 1
                        playing_cb.value = False
                frame_slider.value = next_idx
                _apply_frame(next_idx)
                time.sleep(1.0 / max(1.0, float(fps_in.value)))
            else:
                time.sleep(0.02)

    threading.Thread(target=_player_loop, daemon=True).start()

    print("[dual_joint_renderer] Open the Viser URL above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Config))
