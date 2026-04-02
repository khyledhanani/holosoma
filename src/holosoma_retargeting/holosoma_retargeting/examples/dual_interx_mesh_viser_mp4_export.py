#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
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

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.examples.dual_interx_mesh_renderer import (  # noqa: E402
    _build_proxy_mesh_frame,
    _choose_device,
    _convert_y_up_to_z_up_points,
    _convert_y_up_to_z_up_qpos,
    _infer_sequence_id,
    _load_direct_mesh_from_npz,
    _load_mesh_from_motion_folder,
    _load_proxy_joints_from_npz,
    _load_qpos_tracks,
    _load_reference_timeline,
    _resolve_mesh_scales,
    _resolve_robot_urdf_paths,
    _resolve_target_types,
    _update_mesh,
)
from holosoma_retargeting.examples.dual_joint_viser_mp4_export import (  # noqa: E402
    DEFAULT_CAMERA_DIRECTION,
    _calculate_camera_pose,
    _composite_frame_for_video,
    _ensure_client_viewport_size,
    _load_camera_preset_json,
    _rgb_to_bgr,
    _set_client_camera,
    _snapshot_client_camera,
    _wait_for_client,
)


@dataclass
class Config:
    data_npz: str
    """Path to an Inter-X sequence npz or mixed retarget output."""

    output_mp4: str | None = None
    """Destination MP4 path. Defaults to <qpos_npz stem>_mesh_viser.mp4."""

    qpos_npz: str | None = None
    """Optional npz containing qpos_A/qpos_B. If None, try reading from data_npz."""

    sequence_id: str | None = None
    """Optional override for sequence id. If None, uses npz sequence_id field or file stem."""

    interx_motion_root: str = "DATA/motions"
    """Root containing raw per-sequence folders with P1.npz/P2.npz."""

    smplx_model_root: str = "DATA/smpl_all_models"
    """Root for SMPL-X model files."""

    prefer_neutral_gender: bool = True
    """If True, force neutral SMPL-X model for both persons."""

    device: str = "cpu"
    """Torch device for SMPL-X inference ('cpu', 'cuda', or 'auto')."""

    batch_size: int = 256
    """Chunk size for SMPL-X forward passes."""

    port: int = 8080
    """Port for the Viser server."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from data_npz."""

    frame_stride: int = 1
    """Subsample frames for faster export."""

    max_frames: int | None = None
    """Optional cap on the number of exported frames."""

    mesh_opacity: float = 0.9
    """Human mesh opacity in [0, 1]."""

    show_human_proxy: bool = False
    """Render the MuJoCo human collision proxy overlay when human joints are available."""

    human_proxy_opacity: float = 1.0
    """Opacity for the MuJoCo human proxy overlay."""

    hide_human_mesh_on_robot_side: bool = True
    """In mixed mode, hide the SMPL-X mesh on the robot-retargeted side."""

    y_up_to_z_up: bool = True
    """Convert incoming points from y-up to z-up before rendering."""

    show_robots: bool = False
    """Render robot overlays if qpos is available."""

    robot_urdf: str | None = None
    """Deprecated shared URDF path used for both A and B when per-agent paths are not provided."""

    robot_urdf_a: str | None = None
    """URDF path for robot A overlay. If None, infer from npz metadata."""

    robot_urdf_b: str | None = None
    """URDF path for robot B overlay. If None, infer from npz metadata."""

    width: int = 1920
    """Output width in pixels."""

    height: int = 1080
    """Output height in pixels."""

    camera_mode: Literal["auto_fit", "current_client", "preset"] = "auto_fit"
    """Use automatic framing, freeze the current browser camera, or load a saved camera preset."""

    camera_preset_json: str | None = None
    """Optional camera preset JSON used by --camera-mode preset."""

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


def _collect_focus_points(
    verts_a: np.ndarray,
    verts_b: np.ndarray,
    proxy_joints_a: np.ndarray | None,
    proxy_joints_b: np.ndarray | None,
    qpos_a: np.ndarray | None,
    qpos_b: np.ndarray | None,
) -> np.ndarray:
    points: list[np.ndarray] = []
    if verts_a.size > 0:
        points.append(verts_a.reshape(-1, 3))
    if verts_b.size > 0:
        points.append(verts_b.reshape(-1, 3))
    if proxy_joints_a is not None:
        points.append(proxy_joints_a.reshape(-1, 3))
    if proxy_joints_b is not None:
        points.append(proxy_joints_b.reshape(-1, 3))
    if qpos_a is not None:
        points.append(np.asarray(qpos_a[:, :3], dtype=np.float32))
    if qpos_b is not None:
        points.append(np.asarray(qpos_b[:, :3], dtype=np.float32))
    if not points:
        raise ValueError("No geometry available to fit the camera.")
    return np.concatenate(points, axis=0).astype(np.float32, copy=False)


def main(cfg: Config) -> None:
    npz_path = Path(cfg.data_npz)
    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    output_mp4_path = (
        Path(cfg.output_mp4)
        if cfg.output_mp4 is not None
        else qpos_npz_path.with_name(f"{qpos_npz_path.stem}_mesh_viser.mp4")
    )
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    sequence_id_ref, fps_ref, target_frames = _load_reference_timeline(npz_path, cfg.frame_stride)
    sequence_id = cfg.sequence_id if cfg.sequence_id is not None else _infer_sequence_id(npz_path, sequence_id_ref)
    fps = float(cfg.fps_override) if cfg.fps_override is not None else float(fps_ref)
    fps = max(1.0e-6, fps)

    proxy_joints_a, proxy_joints_b, proxy_coordinate_frame, proxy_names_a, proxy_names_b = _load_proxy_joints_from_npz(
        npz_path,
        cfg.frame_stride,
    )
    direct_mesh = _load_direct_mesh_from_npz(npz_path=npz_path, frame_stride=cfg.frame_stride)
    mesh_available = True
    mesh_warning: str | None = None
    if direct_mesh is not None:
        verts_a, verts_b, faces = direct_mesh
        source_mode = "direct_npz_mesh"
    else:
        try:
            device = _choose_device(cfg.device)
            verts_a, verts_b, faces = _load_mesh_from_motion_folder(
                sequence_id=sequence_id,
                motion_root=Path(cfg.interx_motion_root),
                model_root=Path(cfg.smplx_model_root),
                prefer_neutral_gender=cfg.prefer_neutral_gender,
                device=device,
                batch_size=cfg.batch_size,
            )
            if target_frames is not None:
                idx = np.rint(np.linspace(0, verts_a.shape[0] - 1, num=target_frames)).astype(np.int64)
                verts_a = verts_a[idx]
                verts_b = verts_b[idx]
            else:
                stride = max(1, int(cfg.frame_stride))
                verts_a = verts_a[::stride]
                verts_b = verts_b[::stride]
            source_mode = "smplx_reconstructed"
        except Exception as exc:
            if proxy_joints_a is None or proxy_joints_b is None:
                raise
            mesh_available = False
            mesh_warning = str(exc)
            n_proxy = min(proxy_joints_a.shape[0], proxy_joints_b.shape[0])
            verts_a = np.zeros((n_proxy, 0, 3), dtype=np.float32)
            verts_b = np.zeros((n_proxy, 0, 3), dtype=np.float32)
            faces = np.zeros((0, 3), dtype=np.uint32)
            source_mode = "proxy_only_fallback"

    if mesh_available and cfg.y_up_to_z_up:
        verts_a = _convert_y_up_to_z_up_points(verts_a)
        verts_b = _convert_y_up_to_z_up_points(verts_b)

    mesh_scale_a, mesh_scale_b = _resolve_mesh_scales(npz_path, qpos_npz_path)
    if mesh_available:
        verts_a = verts_a * float(mesh_scale_a)
        verts_b = verts_b * float(mesh_scale_b)

    qpos_a, qpos_b, qpos_coordinate_frame = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    target_a, target_b = _resolve_target_types(npz_path, qpos_npz_path)
    show_robot_a = bool(cfg.show_robots and target_a != "human")
    show_robot_b = bool(cfg.show_robots and target_b != "human")
    show_proxy_a = bool(cfg.show_human_proxy and proxy_joints_a is not None and target_a == "human")
    show_proxy_b = bool(cfg.show_human_proxy and proxy_joints_b is not None and target_b == "human")
    hide_human_mesh_a = bool(cfg.hide_human_mesh_on_robot_side and target_a == "robot" and target_b == "human")
    hide_human_mesh_b = bool(cfg.hide_human_mesh_on_robot_side and target_b == "robot" and target_a == "human")

    should_convert_proxy = bool(cfg.y_up_to_z_up)
    if proxy_coordinate_frame == "z_up":
        should_convert_proxy = False
    if should_convert_proxy:
        if proxy_joints_a is not None:
            proxy_joints_a = _convert_y_up_to_z_up_points(proxy_joints_a)
        if proxy_joints_b is not None:
            proxy_joints_b = _convert_y_up_to_z_up_points(proxy_joints_b)

    should_convert_qpos = bool(cfg.y_up_to_z_up)
    if qpos_coordinate_frame == "z_up":
        should_convert_qpos = False
    if should_convert_qpos:
        if show_robot_a and qpos_a is not None:
            qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
        if show_robot_b and qpos_b is not None:
            qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)

    n_frames = min(verts_a.shape[0], verts_b.shape[0])
    if show_robot_a and qpos_a is not None:
        n_frames = min(n_frames, qpos_a.shape[0])
    if show_robot_b and qpos_b is not None:
        n_frames = min(n_frames, qpos_b.shape[0])
    if proxy_joints_a is not None:
        n_frames = min(n_frames, proxy_joints_a.shape[0])
    if proxy_joints_b is not None:
        n_frames = min(n_frames, proxy_joints_b.shape[0])
    if cfg.max_frames is not None:
        n_frames = min(n_frames, max(0, int(cfg.max_frames)))
    if n_frames <= 0:
        raise ValueError("No frames available after applying frame limits.")

    verts_a = verts_a[:n_frames]
    verts_b = verts_b[:n_frames]
    if qpos_a is not None:
        qpos_a = qpos_a[:n_frames]
    if qpos_b is not None:
        qpos_b = qpos_b[:n_frames]
    if proxy_joints_a is not None:
        proxy_joints_a = proxy_joints_a[:n_frames]
    if proxy_joints_b is not None:
        proxy_joints_b = proxy_joints_b[:n_frames]

    print(
        f"[dual_interx_mesh_viser_mp4_export] sequence={sequence_id} frames={n_frames} "
        f"verts={verts_a.shape[1]} faces={faces.shape[0]} fps={fps:.2f} source={source_mode}"
    )
    print(f"[dual_interx_mesh_viser_mp4_export] data_npz={npz_path}")
    print(f"[dual_interx_mesh_viser_mp4_export] qpos_npz={qpos_npz_path}")
    print(
        "[dual_interx_mesh_viser_mp4_export] target_A="
        f"{target_a} | target_B={target_b} | y_up_to_z_up={cfg.y_up_to_z_up}"
    )
    if mesh_warning is not None:
        print(f"[dual_interx_mesh_viser_mp4_export] mesh_fallback={mesh_warning}")

    server = viser.ViserServer(port=cfg.port)
    server.scene.set_up_direction("+z")
    server.scene.add_grid(
        "/grid",
        width=4.0,
        height=4.0,
        position=(0.0, 0.0, 0.0),
        fade_from="camera",
    )

    mesh_a = None
    mesh_b = None
    if mesh_available:
        mesh_a = server.scene.add_mesh_simple(
            "/humans/A/mesh",
            vertices=verts_a[0],
            faces=faces,
            color=(0, 200, 255),
            opacity=float(cfg.mesh_opacity),
        )
        mesh_a.visible = not hide_human_mesh_a
        mesh_b = server.scene.add_mesh_simple(
            "/humans/B/mesh",
            vertices=verts_b[0],
            faces=faces,
            color=(255, 160, 0),
            opacity=float(cfg.mesh_opacity),
        )
        mesh_b.visible = not hide_human_mesh_b

    proxy_a_handle = None
    proxy_b_handle = None
    if show_proxy_a and proxy_joints_a is not None:
        proxy_a_verts, proxy_a_faces = _build_proxy_mesh_frame(proxy_joints_a[0], joint_names=proxy_names_a)
        proxy_a_handle = server.scene.add_mesh_simple(
            "/humans/A/proxy",
            vertices=proxy_a_verts,
            faces=proxy_a_faces,
            color=(215, 85, 70),
            opacity=float(cfg.human_proxy_opacity),
        )
    if show_proxy_b and proxy_joints_b is not None:
        proxy_b_verts, proxy_b_faces = _build_proxy_mesh_frame(proxy_joints_b[0], joint_names=proxy_names_b)
        proxy_b_handle = server.scene.add_mesh_simple(
            "/humans/B/proxy",
            vertices=proxy_b_verts,
            faces=proxy_b_faces,
            color=(215, 85, 70),
            opacity=float(cfg.human_proxy_opacity),
        )

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

    def _apply_frame(idx: int) -> None:
        nonlocal mesh_a, mesh_b, proxy_a_handle, proxy_b_handle
        i = int(np.clip(idx, 0, n_frames - 1))
        if mesh_available and mesh_a is not None:
            mesh_a = _update_mesh(
                server=server,
                handle=mesh_a,
                path="/humans/A/mesh",
                vertices=verts_a[i],
                faces=faces,
                color=(0, 200, 255),
                opacity=float(cfg.mesh_opacity),
            )
            if hasattr(mesh_a, "visible"):
                mesh_a.visible = not hide_human_mesh_a
        if mesh_available and mesh_b is not None:
            mesh_b = _update_mesh(
                server=server,
                handle=mesh_b,
                path="/humans/B/mesh",
                vertices=verts_b[i],
                faces=faces,
                color=(255, 160, 0),
                opacity=float(cfg.mesh_opacity),
            )
            if hasattr(mesh_b, "visible"):
                mesh_b.visible = not hide_human_mesh_b
        if show_proxy_a and proxy_joints_a is not None:
            proxy_a_verts, proxy_a_faces = _build_proxy_mesh_frame(proxy_joints_a[i], joint_names=proxy_names_a)
            proxy_a_handle = _update_mesh(
                server=server,
                handle=proxy_a_handle,
                path="/humans/A/proxy",
                vertices=proxy_a_verts,
                faces=proxy_a_faces,
                color=(215, 85, 70),
                opacity=float(cfg.human_proxy_opacity),
            )
        if show_proxy_b and proxy_joints_b is not None:
            proxy_b_verts, proxy_b_faces = _build_proxy_mesh_frame(proxy_joints_b[i], joint_names=proxy_names_b)
            proxy_b_handle = _update_mesh(
                server=server,
                handle=proxy_b_handle,
                path="/humans/B/proxy",
                vertices=proxy_b_verts,
                faces=proxy_b_faces,
                color=(215, 85, 70),
                opacity=float(cfg.human_proxy_opacity),
            )
        if show_robot_a and qpos_a is not None and robot_a is not None and robot_a_root is not None and robot_dof_a is not None:
            qa = qpos_a[i]
            robot_a_root.position = qa[0:3]
            robot_a_root.wxyz = qa[3:7]
            robot_a.update_cfg(qa[7 : 7 + robot_dof_a])
        if show_robot_b and qpos_b is not None and robot_b is not None and robot_b_root is not None and robot_dof_b is not None:
            qb = qpos_b[i]
            robot_b_root.position = qb[0:3]
            robot_b_root.wxyz = qb[3:7]
            robot_b.update_cfg(qb[7 : 7 + robot_dof_b])

    _apply_frame(0)
    server.flush()

    print(f"[dual_interx_mesh_viser_mp4_export] Open http://localhost:{cfg.port} in a browser.")
    client = _wait_for_client(server, cfg.client_wait_timeout)
    _ensure_client_viewport_size(client, cfg.width, cfg.height)

    if cfg.camera_mode == "auto_fit":
        focus_points = _collect_focus_points(verts_a, verts_b, proxy_joints_a, proxy_joints_b, qpos_a, qpos_b)
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
        time.sleep(max(0.0, float(cfg.camera_settle_seconds)))
        camera_pose = _snapshot_client_camera(client)
    elif cfg.camera_mode == "current_client":
        print("[dual_interx_mesh_viser_mp4_export] Adjust the browser camera, then press Enter to start export.")
        input()
        camera_pose = _snapshot_client_camera(client)
    else:
        if cfg.camera_preset_json is None:
            raise ValueError("--camera-mode preset requires --camera-preset-json.")
        camera_pose = _load_camera_preset_json(Path(cfg.camera_preset_json))
        _set_client_camera(server, client, camera_pose)
        time.sleep(max(0.0, float(cfg.camera_settle_seconds)))

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
                print(f"[dual_interx_mesh_viser_mp4_export] rendered frame {frame_idx + 1}/{n_frames}")
    finally:
        writer.release()

    print(f"[dual_interx_mesh_viser_mp4_export] wrote {output_mp4_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
