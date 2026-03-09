#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


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


@dataclass
class Config:
    data_npz: str
    """Path to an Inter-X joint npz file (dual or single)."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from npz (default 30 if absent)."""

    frame_stride: int = 1
    """Subsample frames for faster playback."""

    point_size: float = 0.02
    """Joint marker size."""

    line_width: float = 2.0
    """Skeleton line width."""

    show_skeleton: bool = True
    """Render skeleton edges in addition to joints."""

    loop: bool = True
    """Loop playback."""

    y_up_to_z_up: bool = True
    """Convert incoming joints from y-up to z-up before rendering."""

    show_robots: bool = False
    """Render robot overlays if qpos is available."""

    robot_urdf: str = "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf"
    """URDF path used for robot overlay rendering (e.g., Unitree G1)."""

    qpos_npz: str | None = None
    """Optional npz containing qpos_A/qpos_B. If None, try reading from data_npz."""


def _load_joint_trajectories(npz_path: Path, frame_stride: int) -> tuple[np.ndarray, np.ndarray | None, float, str]:
    data = np.load(str(npz_path), allow_pickle=True)

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
    sequence_id = str(data["sequence_id"]) if "sequence_id" in data else npz_path.stem
    return joints_a, joints_b, fps, sequence_id


def _edge_segments(joints: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([[joints[i], joints[j]] for i, j in edges], dtype=np.float32)


def _convert_y_up_to_z_up_points(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    converted = (R_YUP_TO_ZUP @ flat.T).T
    return converted.reshape(points.shape).astype(np.float32, copy=False)


def _load_qpos_tracks(npz_path: Path, frame_stride: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    data = np.load(str(npz_path), allow_pickle=True)
    qpos_a = np.asarray(data["qpos_A"], dtype=np.float32)[:: max(1, int(frame_stride))] if "qpos_A" in data else None
    qpos_b = np.asarray(data["qpos_B"], dtype=np.float32)[:: max(1, int(frame_stride))] if "qpos_B" in data else None
    if qpos_a is not None and qpos_b is not None:
        n = min(qpos_a.shape[0], qpos_b.shape[0])
        qpos_a = qpos_a[:n]
        qpos_b = qpos_b[:n]
    return qpos_a, qpos_b


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
    joints_a, joints_b, fps_from_data, sequence_id = _load_joint_trajectories(npz_path, cfg.frame_stride)
    if cfg.y_up_to_z_up:
        joints_a = _convert_y_up_to_z_up_points(joints_a)
        if joints_b is not None:
            joints_b = _convert_y_up_to_z_up_points(joints_b)
    fps = float(cfg.fps_override) if cfg.fps_override is not None else fps_from_data
    fps = max(1e-6, fps)

    n_frames, n_joints, _ = joints_a.shape
    print(f"[dual_joint_renderer] sequence={sequence_id} frames={n_frames} joints={n_joints} fps={fps:.2f}")
    print(f"[dual_joint_renderer] rendering {'dual' if joints_b is not None else 'single'} tracks")
    print(f"[dual_joint_renderer] y_up_to_z_up={cfg.y_up_to_z_up}")

    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    qpos_a, qpos_b = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    if cfg.show_robots:
        if qpos_a is None:
            raise ValueError(
                f"show_robots=True but no qpos_A found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a dual retarget output."
            )
        if joints_b is not None and qpos_b is None:
            raise ValueError(f"show_robots=True with dual joints, but no qpos_B found in {qpos_npz_path}.")
        if cfg.y_up_to_z_up:
            qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
            if qpos_b is not None:
                qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))

    # Human A (cyan)
    a_joints_handle = server.scene.add_point_cloud(
        "/humans/A/joints",
        points=joints_a[0],
        colors=np.tile(np.array([[0, 200, 255]], dtype=np.uint8), (n_joints, 1)),
        point_size=cfg.point_size,
        point_shape="circle",
    )
    a_lines_handle = None
    if cfg.show_skeleton and n_joints >= 22:
        a_lines_handle = server.scene.add_line_segments(
            "/humans/A/skeleton",
            points=_edge_segments(joints_a[0], SMPLX22_EDGES),
            colors=np.tile(np.array([[[0, 180, 230], [0, 180, 230]]], dtype=np.uint8), (len(SMPLX22_EDGES), 1, 1)),
            line_width=cfg.line_width,
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
        if cfg.show_skeleton and n_joints >= 22:
            b_lines_handle = server.scene.add_line_segments(
                "/humans/B/skeleton",
                points=_edge_segments(joints_b[0], SMPLX22_EDGES),
                colors=np.tile(
                    np.array([[[230, 140, 0], [230, 140, 0]]], dtype=np.uint8),
                    (len(SMPLX22_EDGES), 1, 1),
                ),
                line_width=cfg.line_width,
            )

    # Optional robot overlays (A and B) from qpos_A/qpos_B.
    robot_a = None
    robot_b = None
    robot_a_root = None
    robot_b_root = None
    robot_dof = None
    if cfg.show_robots:
        urdf_path = Path(cfg.robot_urdf)
        robot_urdf = yourdfpy.URDF.load(str(urdf_path), load_meshes=True, build_scene_graph=True)

        robot_a_root = server.scene.add_frame("/robots/A", show_axes=False)
        robot_a = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robots/A")
        robot_a.show_visual = True

        if joints_b is not None:
            robot_b_root = server.scene.add_frame("/robots/B", show_axes=False)
            robot_b = ViserUrdf(server, urdf_or_path=robot_urdf, root_node_name="/robots/B")
            robot_b.show_visual = True

        robot_dof = len(robot_a.get_actuated_joint_limits())
        print(f"[dual_joint_renderer] robot overlay enabled | urdf={urdf_path} | dof={robot_dof}")

        # Clamp timeline to available qpos frames.
        n_frames = min(n_frames, qpos_a.shape[0])  # type: ignore[union-attr]
        if joints_b is not None and qpos_b is not None:
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
        if cfg.show_robots:
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
        if joints_b is not None and b_joints_handle is not None:
            b_joints_handle.points = joints_b[i]
            b_joints_handle.point_size = float(point_size_slider.value)
        if joints_b is not None and b_lines_handle is not None:
            b_lines_handle.points = _edge_segments(joints_b[i], SMPLX22_EDGES)

        if cfg.show_robots and qpos_a is not None and robot_a is not None and robot_a_root is not None and robot_dof is not None:
            qa = qpos_a[i]
            robot_a_root.position = qa[0:3]
            robot_a_root.wxyz = qa[3:7]
            robot_a.update_cfg(qa[7 : 7 + robot_dof])
        if (
            cfg.show_robots
            and joints_b is not None
            and qpos_b is not None
            and robot_b is not None
            and robot_b_root is not None
            and robot_dof is not None
        ):
            qb = qpos_b[i]
            robot_b_root.position = qb[0:3]
            robot_b_root.wxyz = qb[3:7]
            robot_b.update_cfg(qb[7 : 7 + robot_dof])

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
