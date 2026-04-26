#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Literal

import cv2
import numpy as np
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]
from viser import transforms as vtf  # type: ignore[import-not-found]

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.examples.dual_joint_renderer import (  # noqa: E402
    SMPLX22_EDGES,
    _convert_y_up_to_z_up_points,
    _convert_y_up_to_z_up_qpos,
    _edge_segments,
    _line_segment_colors,
    _load_qpos_tracks,
    _load_render_joint_trajectories,
    _precompute_optimizer_graph_segments,
    _resolve_robot_urdf_paths,
)


DEFAULT_CAMERA_DIRECTION = (1.0, -1.0, 0.5)


@dataclass
class Config:
    data_npz: str
    """Path to an Inter-X joint npz file (dual or single)."""

    output_mp4: str | None = None
    """Destination MP4 path. Defaults to <qpos_npz stem>_viser.mp4."""

    qpos_npz: str | None = None
    """Optional npz containing qpos_A/qpos_B. If None, try reading from data_npz."""

    port: int = 8080
    """Port for the Viser server."""

    prefer_optimization_joints: bool = True
    """Prefer the resized/preprocessed joints used by the optimizer when available."""

    joint_scale_mode: Literal["individual", "unified"] = "individual"
    """Choose whether to render individually-scaled optimizer joints (P_ind) or unified-scale joints (P_uni)."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from npz (default 30 if absent)."""

    frame_stride: int = 1
    """Subsample frames for faster export."""

    max_frames: int | None = None
    """Optional cap on the number of exported frames."""

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

    width: int = 1920
    """Output width in pixels."""

    height: int = 1080
    """Output height in pixels."""

    camera_mode: Literal["auto_fit", "current_client", "preset"] = "auto_fit"
    """Use automatic framing, freeze the current browser camera, or load a saved camera preset."""

    camera_preset_json: str | None = None
    """Optional camera preset JSON used by --camera-mode preset."""

    save_camera_preset_json: str | None = None
    """Optional output path for saving the resolved camera pose as JSON."""

    camera_direction: tuple[float, float, float] = DEFAULT_CAMERA_DIRECTION
    """Camera direction from target to eye for auto_fit mode."""

    camera_yfov: float = float(np.pi / 3.0)
    """Vertical field of view in radians for auto_fit mode."""

    camera_margin: float = 1.1
    """Distance multiplier applied after fitting content into frame."""

    camera_min_distance: float = 2.8
    """Minimum camera distance from target."""

    camera_target_vertical_bias: float = 0.0
    """Vertical offset added to the fitted camera target."""

    client_wait_timeout: float = 120.0
    """Seconds to wait for a browser client to connect."""

    camera_settle_seconds: float = 0.35
    """Delay after setting camera pose before capture begins."""

    render_delay_seconds: float = 0.03
    """Delay after each frame update before requesting the browser render."""

    transport_format: Literal["png", "jpeg"] = "png"
    """Format used for browser-to-Python frame transfer."""

    background_style: Literal["original", "viewer", "white"] = "viewer"
    """Background fill used when compositing transparent PNG renders into MP4 frames."""


@dataclass(frozen=True)
class _CameraPose:
    wxyz: np.ndarray
    position: np.ndarray
    look_at: np.ndarray
    up_direction: np.ndarray
    fov: float


def _collect_focus_points(
    joints_a: np.ndarray,
    joints_b: np.ndarray | None,
    qpos_a: np.ndarray | None,
    qpos_b: np.ndarray | None,
) -> np.ndarray:
    points = [joints_a.reshape(-1, 3)]
    if joints_b is not None:
        points.append(joints_b.reshape(-1, 3))
    if qpos_a is not None:
        points.append(np.asarray(qpos_a[:, :3], dtype=np.float32))
    if qpos_b is not None:
        points.append(np.asarray(qpos_b[:, :3], dtype=np.float32))
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def _calculate_camera_pose(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    direction: tuple[float, float, float],
    yfov: float,
    distance_margin: float,
    min_distance: float,
    target_vertical_bias: float,
) -> _CameraPose:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    target = points.mean(axis=0)
    target[2] += float(target_vertical_bias)

    direction_vec = np.asarray(direction, dtype=np.float32)
    direction_norm = float(np.linalg.norm(direction_vec))
    if direction_norm <= 1.0e-8:
        raise ValueError("camera_direction must be non-zero.")
    direction_vec /= direction_norm

    z_cam = direction_vec
    x_cam = np.cross(np.array([0.0, 0.0, 1.0], dtype=np.float32), z_cam)
    if float(np.linalg.norm(x_cam)) < 1.0e-8:
        x_cam = np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), z_cam)
    x_cam /= float(np.linalg.norm(x_cam))
    y_cam = np.cross(z_cam, x_cam)
    y_cam /= float(np.linalg.norm(y_cam))

    offsets = points - target[None, :]
    x_proj = offsets @ x_cam
    y_proj = offsets @ y_cam
    max_x_spread = float(np.max(np.abs(x_proj))) if x_proj.size else 0.0
    max_y_spread = float(np.max(np.abs(y_proj))) if y_proj.size else 0.0

    aspect = float(width) / float(height)
    dist_for_y = max_y_spread / np.tan(float(yfov) / 2.0) if max_y_spread > 0.0 else 0.0
    hfov = 2.0 * np.arctan(aspect * np.tan(float(yfov) / 2.0))
    dist_for_x = max_x_spread / np.tan(hfov / 2.0) if max_x_spread > 0.0 else 0.0
    distance = max(float(min_distance), float(distance_margin) * max(dist_for_x, dist_for_y))

    position = target + distance * direction_vec
    return _CameraPose(
        wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        position=position.astype(np.float32),
        look_at=target.astype(np.float32),
        up_direction=y_cam.astype(np.float32),
        fov=float(yfov),
    )


def _wait_for_client(server: viser.ViserServer, timeout_seconds: float) -> viser.ClientHandle:
    deadline = time.time() + max(0.0, float(timeout_seconds))
    while True:
        clients = server.get_clients()
        if clients:
            return next(iter(clients.values()))
        if time.time() > deadline:
            raise TimeoutError(
                f"No browser client connected within {timeout_seconds:.1f}s. "
                "Open the Viser URL printed above and retry."
            )
        time.sleep(0.05)


def _ensure_client_viewport_size(client: viser.ClientHandle, width: int, height: int) -> None:
    client_width = int(client.camera.image_width)
    client_height = int(client.camera.image_height)
    if client_width < int(width) or client_height < int(height):
        raise ValueError(
            "Browser viewport is smaller than requested export size: "
            f"viewport={client_width}x{client_height}, requested={width}x{height}. "
            "Resize the browser window or lower --width/--height."
        )


def _set_client_camera(
    server: viser.ViserServer,
    client: viser.ClientHandle,
    pose: _CameraPose,
) -> None:
    server.initial_camera.position = tuple(float(x) for x in pose.position)
    server.initial_camera.look_at = tuple(float(x) for x in pose.look_at)
    server.initial_camera.up = tuple(float(x) for x in pose.up_direction)
    server.initial_camera.fov = float(pose.fov)
    with client.atomic():
        client.camera.position = pose.position
        client.camera.look_at = pose.look_at
        client.camera.up_direction = pose.up_direction
        client.camera.wxyz = pose.wxyz
        client.camera.fov = float(pose.fov)
    server.flush()


def _snapshot_client_camera(client: viser.ClientHandle) -> _CameraPose:
    return _CameraPose(
        wxyz=np.array(client.camera.wxyz, dtype=np.float32, copy=True),
        position=np.array(client.camera.position, dtype=np.float32, copy=True),
        look_at=np.array(client.camera.look_at, dtype=np.float32, copy=True),
        up_direction=np.array(client.camera.up_direction, dtype=np.float32, copy=True),
        fov=float(client.camera.fov),
    )


def _camera_pose_to_json_dict(pose: _CameraPose) -> dict[str, object]:
    return {
        "wxyz": [float(x) for x in pose.wxyz],
        "position": [float(x) for x in pose.position],
        "look_at": [float(x) for x in pose.look_at],
        "up_direction": [float(x) for x in pose.up_direction],
        "fov": float(pose.fov),
    }


def _write_camera_preset_json(path: Path, pose: _CameraPose) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_camera_pose_to_json_dict(pose), f, indent=2)
        f.write("\n")


def _load_camera_preset_json(path: Path) -> _CameraPose:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        wxyz_payload = payload.get("wxyz")
        position = np.asarray(payload["position"], dtype=np.float32)
        look_at = np.asarray(payload["look_at"], dtype=np.float32)
        up_direction = np.asarray(payload["up_direction"], dtype=np.float32)
        fov = float(payload["fov"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid camera preset JSON: {path}") from exc
    if position.shape != (3,) or look_at.shape != (3,) or up_direction.shape != (3,):
        raise ValueError(
            f"Invalid camera preset JSON: {path} "
            "(position, look_at, up_direction must all be length-3 vectors)."
        )
    if wxyz_payload is None:
        z = look_at - position
        z /= float(np.linalg.norm(z))
        y = (2.0 * float(np.dot(z, up_direction)) * z) - up_direction
        y = y - float(np.dot(z, y)) * z
        y /= float(np.linalg.norm(y))
        x = np.cross(y, z)
        wxyz = vtf.SO3.from_matrix(np.stack([x, y, z], axis=1)).wxyz.astype(np.float32)
    else:
        wxyz = np.asarray(wxyz_payload, dtype=np.float32)
        if wxyz.shape != (4,):
            raise ValueError(f"Invalid camera preset JSON: {path} (wxyz must be a length-4 vector).")
    return _CameraPose(
        wxyz=wxyz,
        position=position,
        look_at=look_at,
        up_direction=up_direction,
        fov=fov,
    )


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected an image with shape (H, W, C), got {frame.shape}")
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
    if frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Expected 3 or 4 channels, got {frame.shape}")


def _make_background_rgb(
    width: int,
    height: int,
    style: Literal["original", "viewer", "white"],
) -> np.ndarray | None:
    if style == "original":
        return None
    if style == "white":
        return np.full((height, width, 3), 255, dtype=np.uint8)

    top = np.array([228, 235, 243], dtype=np.float32)
    bottom = np.array([247, 249, 251], dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    rgb = top[None, None, :] * (1.0 - vertical) + bottom[None, None, :] * vertical
    return np.broadcast_to(rgb.astype(np.uint8), (height, width, 3)).copy()


def _composite_frame_for_video(
    frame: np.ndarray,
    background_style: Literal["original", "viewer", "white"],
) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError(f"Expected an image with shape (H, W, C), got {frame.shape}")

    if frame.shape[2] == 3:
        return frame
    if frame.shape[2] != 4:
        raise ValueError(f"Expected 3 or 4 channels, got {frame.shape}")

    background = _make_background_rgb(frame.shape[1], frame.shape[0], background_style)
    if background is None:
        background = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    rgb = frame[..., :3].astype(np.float32)
    alpha = (frame[..., 3:4].astype(np.float32) / 255.0)
    composite = rgb * alpha + background.astype(np.float32) * (1.0 - alpha)
    return np.clip(composite, 0.0, 255.0).astype(np.uint8)


def main(cfg: Config) -> None:
    npz_path = Path(cfg.data_npz)
    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    output_mp4_path = (
        Path(cfg.output_mp4)
        if cfg.output_mp4 is not None
        else qpos_npz_path.with_name(f"{qpos_npz_path.stem}_viser.mp4")
    )
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

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
    fps = max(1.0e-6, fps)

    qpos_a, qpos_b, qpos_coordinate_frame = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    if cfg.show_robots:
        if qpos_a is None:
            raise ValueError(
                f"show_robots=True but no qpos_A found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a dual retarget output."
            )
        if joints_b is not None and qpos_b is None:
            raise ValueError(f"show_robots=True with dual joints, but no qpos_B found in {qpos_npz_path}.")
        should_convert_qpos = bool(cfg.y_up_to_z_up)
        if qpos_coordinate_frame == "z_up":
            should_convert_qpos = False
        if should_convert_qpos:
            qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
            if qpos_b is not None:
                qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)

    optimizer_graph_a = None
    optimizer_graph_b = None
    optimizer_graph_cross = None
    if cfg.show_optimizer_graph:
        optimizer_graph_a, optimizer_graph_b, optimizer_graph_cross = _precompute_optimizer_graph_segments(
            joints_a=joints_a,
            joints_b=joints_b,
            cfg=cfg,
        )

    n_frames = joints_a.shape[0]
    if joints_b is not None:
        n_frames = min(n_frames, joints_b.shape[0])
    if qpos_a is not None:
        n_frames = min(n_frames, qpos_a.shape[0])
    if qpos_b is not None:
        n_frames = min(n_frames, qpos_b.shape[0])
    if cfg.max_frames is not None:
        n_frames = min(n_frames, max(0, int(cfg.max_frames)))
    if n_frames <= 0:
        raise ValueError("No frames available after applying frame limits.")

    joints_a = joints_a[:n_frames]
    if joints_b is not None:
        joints_b = joints_b[:n_frames]
    if qpos_a is not None:
        qpos_a = qpos_a[:n_frames]
    if qpos_b is not None:
        qpos_b = qpos_b[:n_frames]
    if optimizer_graph_a is not None:
        optimizer_graph_a = optimizer_graph_a[:n_frames]
    if optimizer_graph_b is not None:
        optimizer_graph_b = optimizer_graph_b[:n_frames]
    if optimizer_graph_cross is not None:
        optimizer_graph_cross = optimizer_graph_cross[:n_frames]

    n_joints = joints_a.shape[1]
    print(f"[dual_joint_viser_mp4_export] sequence={sequence_id} frames={n_frames} joints={n_joints} fps={fps:.2f}")
    print(f"[dual_joint_viser_mp4_export] data_npz={npz_path}")
    print(f"[dual_joint_viser_mp4_export] qpos_npz={qpos_npz_path}")
    print(
        "[dual_joint_viser_mp4_export] joint_source="
        f"{joint_source} | joint_coordinate_frame={joint_coordinate_frame or 'assumed_y_up'}"
        f" | width={cfg.width} | height={cfg.height}"
    )
    print(
        "[dual_joint_viser_mp4_export] capture="
        f"{cfg.transport_format} | background_style={cfg.background_style}"
    )

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=4.0,
        height=4.0,
        position=(0.0, 0.0, 0.0),
        fade_from="camera",
    )

    a_joints_handle = server.scene.add_point_cloud(
        "/humans/A/joints",
        points=joints_a[0],
        colors=np.tile(np.array([[0, 200, 255]], dtype=np.uint8), (n_joints, 1)),
        point_size=cfg.point_size,
        point_shape="circle",
        visible=cfg.show_human_joints,
    )
    a_lines_handle = None
    if cfg.show_skeleton and n_joints >= 22:
        a_lines_handle = server.scene.add_line_segments(
            "/humans/A/skeleton",
            points=_edge_segments(joints_a[0], SMPLX22_EDGES),
            colors=_line_segment_colors(len(SMPLX22_EDGES), (0, 180, 230)),
            line_width=cfg.line_width,
        )

    b_joints_handle = None
    b_lines_handle = None
    b_optimizer_handle = None
    cross_optimizer_handle = None
    if joints_b is not None:
        b_joints_handle = server.scene.add_point_cloud(
            "/humans/B/joints",
            points=joints_b[0],
            colors=np.tile(np.array([[255, 160, 0]], dtype=np.uint8), (n_joints, 1)),
            point_size=cfg.point_size,
            point_shape="circle",
            visible=cfg.show_human_joints,
        )
        if cfg.show_skeleton and n_joints >= 22:
            b_lines_handle = server.scene.add_line_segments(
                "/humans/B/skeleton",
                points=_edge_segments(joints_b[0], SMPLX22_EDGES),
                colors=_line_segment_colors(len(SMPLX22_EDGES), (230, 140, 0)),
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
        if joints_b is not None and optimizer_graph_b is not None:
            b_optimizer_handle = server.scene.add_line_segments(
                "/humans/B/optimizer_graph",
                points=optimizer_graph_b[0],
                colors=_line_segment_colors(optimizer_graph_b[0].shape[0], (255, 190, 70)),
                line_width=float(cfg.optimizer_graph_line_width),
                visible=cfg.show_optimizer_graph,
            )
        if joints_b is not None and optimizer_graph_cross is not None:
            cross_optimizer_handle = server.scene.add_line_segments(
                "/humans/cross_optimizer_graph",
                points=optimizer_graph_cross[0],
                colors=_line_segment_colors(optimizer_graph_cross[0].shape[0], (255, 90, 90)),
                line_width=float(cfg.optimizer_graph_line_width),
                visible=cfg.show_optimizer_graph,
            )

    robot_a = None
    robot_b = None
    robot_a_root = None
    robot_b_root = None
    robot_dof_a = None
    robot_dof_b = None
    if cfg.show_robots:
        urdf_a_path, urdf_b_path = _resolve_robot_urdf_paths(
            cfg,
            qpos_npz_path,
            include_a=True,
            include_b=joints_b is not None,
        )
        robot_urdf_a = yourdfpy.URDF.load(str(urdf_a_path), load_meshes=True, build_scene_graph=True)
        robot_a_root = server.scene.add_frame("/robots/A", show_axes=False)
        robot_a = ViserUrdf(server, urdf_or_path=robot_urdf_a, root_node_name="/robots/A")
        robot_a.show_visual = True
        robot_dof_a = len(robot_a.get_actuated_joint_limits())

        if joints_b is not None:
            if urdf_b_path is None:
                raise ValueError("Could not resolve robot_urdf for B.")
            robot_urdf_b = yourdfpy.URDF.load(str(urdf_b_path), load_meshes=True, build_scene_graph=True)
            robot_b_root = server.scene.add_frame("/robots/B", show_axes=False)
            robot_b = ViserUrdf(server, urdf_or_path=robot_urdf_b, root_node_name="/robots/B")
            robot_b.show_visual = True
            robot_dof_b = len(robot_b.get_actuated_joint_limits())

        print(
            f"[dual_joint_viser_mp4_export] robot overlay enabled | urdf_A={urdf_a_path} | dof_A={robot_dof_a}"
            + (f" | urdf_B={urdf_b_path} | dof_B={robot_dof_b}" if joints_b is not None else "")
        )

    def _apply_frame(idx: int) -> None:
        i = int(np.clip(idx, 0, n_frames - 1))
        a_joints_handle.points = joints_a[i]
        a_joints_handle.point_size = float(cfg.point_size)
        a_joints_handle.visible = bool(cfg.show_human_joints)
        if a_lines_handle is not None:
            a_lines_handle.points = _edge_segments(joints_a[i], SMPLX22_EDGES)
            a_lines_handle.line_width = float(cfg.line_width)
        if a_optimizer_handle is not None and optimizer_graph_a is not None:
            a_optimizer_handle.points = optimizer_graph_a[i]
            a_optimizer_handle.colors = _line_segment_colors(optimizer_graph_a[i].shape[0], (100, 230, 255))
            a_optimizer_handle.line_width = float(cfg.optimizer_graph_line_width)
        if joints_b is not None and b_joints_handle is not None:
            b_joints_handle.points = joints_b[i]
            b_joints_handle.point_size = float(cfg.point_size)
            b_joints_handle.visible = bool(cfg.show_human_joints)
        if joints_b is not None and b_lines_handle is not None:
            b_lines_handle.points = _edge_segments(joints_b[i], SMPLX22_EDGES)
            b_lines_handle.line_width = float(cfg.line_width)
        if joints_b is not None and b_optimizer_handle is not None and optimizer_graph_b is not None:
            b_optimizer_handle.points = optimizer_graph_b[i]
            b_optimizer_handle.colors = _line_segment_colors(optimizer_graph_b[i].shape[0], (255, 190, 70))
            b_optimizer_handle.line_width = float(cfg.optimizer_graph_line_width)
        if cross_optimizer_handle is not None and optimizer_graph_cross is not None:
            cross_optimizer_handle.points = optimizer_graph_cross[i]
            cross_optimizer_handle.colors = _line_segment_colors(optimizer_graph_cross[i].shape[0], (255, 90, 90))
            cross_optimizer_handle.line_width = float(cfg.optimizer_graph_line_width)

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

    _apply_frame(0)
    server.flush()

    print(f"[dual_joint_viser_mp4_export] Open http://localhost:{cfg.port} in a browser.")
    client = _wait_for_client(server, cfg.client_wait_timeout)
    _ensure_client_viewport_size(client, cfg.width, cfg.height)

    if cfg.camera_mode == "auto_fit":
        focus_points = _collect_focus_points(joints_a, joints_b, qpos_a, qpos_b)
        camera_pose = _calculate_camera_pose(
            focus_points,
            width=cfg.width,
            height=cfg.height,
            direction=cfg.camera_direction,
            yfov=cfg.camera_yfov,
            distance_margin=cfg.camera_margin,
            min_distance=cfg.camera_min_distance,
            target_vertical_bias=cfg.camera_target_vertical_bias,
        )
        _set_client_camera(server, client, camera_pose)
        print(
            "[dual_joint_viser_mp4_export] auto-fit camera="
            f"position={camera_pose.position.tolist()} look_at={camera_pose.look_at.tolist()} "
            f"fov={camera_pose.fov:.4f}"
        )
        time.sleep(max(0.0, float(cfg.camera_settle_seconds)))
        camera_pose = _snapshot_client_camera(client)
    elif cfg.camera_mode == "current_client":
        print("[dual_joint_viser_mp4_export] Adjust the browser camera, then press Enter to start export.")
        input()
        camera_pose = _snapshot_client_camera(client)
        print(
            "[dual_joint_viser_mp4_export] locked client camera="
            f"position={camera_pose.position.tolist()} look_at={camera_pose.look_at.tolist()} "
            f"fov={camera_pose.fov:.4f}"
        )
    else:
        if cfg.camera_preset_json is None:
            raise ValueError("--camera-mode preset requires --camera-preset-json.")
        preset_path = Path(cfg.camera_preset_json)
        camera_pose = _load_camera_preset_json(preset_path)
        _set_client_camera(server, client, camera_pose)
        print(f"[dual_joint_viser_mp4_export] loaded camera preset {preset_path}")
        time.sleep(max(0.0, float(cfg.camera_settle_seconds)))

    if cfg.save_camera_preset_json is not None:
        preset_out = Path(cfg.save_camera_preset_json)
        _write_camera_preset_json(preset_out, camera_pose)
        print(f"[dual_joint_viser_mp4_export] saved camera preset {preset_out}")

    writer = cv2.VideoWriter(
        str(output_mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(cfg.width), int(cfg.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {output_mp4_path}")

    try:
        for frame_idx in range(n_frames):
            with server.atomic():
                _apply_frame(frame_idx)
            server.flush()
            time.sleep(max(0.0, float(cfg.render_delay_seconds)))
            frame = client.get_render(
                height=int(cfg.height),
                width=int(cfg.width),
                wxyz=camera_pose.wxyz,
                position=camera_pose.position,
                fov=float(camera_pose.fov),
                transport_format=cfg.transport_format,
            )
            writer.write(_rgb_to_bgr(_composite_frame_for_video(frame, cfg.background_style)))
            if (frame_idx + 1) % 10 == 0 or frame_idx + 1 == n_frames:
                print(f"[dual_joint_viser_mp4_export] rendered frame {frame_idx + 1}/{n_frames}")
    finally:
        writer.release()

    print(f"[dual_joint_viser_mp4_export] wrote {output_mp4_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
