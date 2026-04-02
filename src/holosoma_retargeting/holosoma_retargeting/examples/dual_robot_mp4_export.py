#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Literal

import cv2
import mujoco
import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.examples.dual_robot_retarget import (  # noqa: E402
    _build_dual_scene_xml_from_pair,
)


Q_YUP_TO_ZUP = np.array([np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0], dtype=np.float32)
R_YUP_TO_ZUP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)
DEFAULT_CAMERA_DIRECTION = (1.0, -1.0, 0.5)
DEFAULT_GRID_REPEAT = 12.0


@dataclass
class Config:
    qpos_npz: str
    """Retarget output npz with qpos_A/qpos_B and dual_scene_xml."""

    output_mp4: str | None = None
    """Destination MP4 path. Defaults to <qpos_npz stem>_robots.mp4."""

    dual_scene_xml: str | None = None
    """Optional override for the dual-scene MuJoCo XML path."""

    frame_stride: int = 1
    """Subsample frames for faster export."""

    max_frames: int | None = None
    """Optional cap on the number of exported frames."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from npz (default 30 if absent)."""

    width: int = 640
    """Output width in pixels."""

    height: int = 480
    """Output height in pixels."""

    camera_direction: tuple[float, float, float] = DEFAULT_CAMERA_DIRECTION
    """Camera direction from target to eye, matching egohuman-rl defaults."""

    camera_yfov: float = float(np.pi / 3.0)
    """Vertical field of view in radians."""

    camera_margin: float = 1.25
    """Distance multiplier applied after fitting content into frame."""

    camera_min_distance: float = 4.0
    """Minimum camera distance from target."""

    camera_target_vertical_bias: float = 0.0
    """Vertical offset added to the camera target."""

    lock_camera: bool = False
    """If True, fit the camera once across the entire clip and keep it fixed."""

    y_up_to_z_up: bool = True
    """Convert qpos from y-up to z-up if metadata says the source is y-up."""

    floor_style: Literal["white_grid", "white", "original"] = "white_grid"
    """Visual style for the floor plane."""

    floor_grid_repeat: float = DEFAULT_GRID_REPEAT
    """How many grid cells to repeat across the plane in each axis."""

    floor_grid_line_px: int = 4
    """Grid line thickness in texture pixels for the white-grid floor."""

    background_style: Literal["original", "viewer", "white"] = "original"
    """Background fill behind rendered geometry."""


@dataclass(frozen=True)
class _DualSceneLayout:
    q_block_a: np.ndarray
    q_block_b: np.ndarray
    min_width_a: int
    min_width_b: int


@dataclass(frozen=True)
class _CameraState:
    pos: np.ndarray
    forward: np.ndarray
    up: np.ndarray


def _coerce_scalar_string(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    text = str(value).strip()
    return text or None


def _coerce_robot_type(value: object) -> str | None:
    text = _coerce_scalar_string(value)
    if text is None:
        return None
    return text.lower().split("_")[0]


def _parse_coordinate_frame(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    frame = str(value).strip().lower()
    if not frame:
        return None
    if frame not in {"y_up", "z_up"}:
        raise ValueError(f"Unsupported coordinate frame metadata: {frame}")
    return frame


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
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32, copy=False)


def _convert_y_up_to_z_up_qpos(qpos: np.ndarray) -> np.ndarray:
    out = np.array(qpos, dtype=np.float32, copy=True)
    out[:, :3] = (R_YUP_TO_ZUP @ out[:, :3].T).T
    for i in range(out.shape[0]):
        q_old = _normalize_quat_wxyz(out[i, 3:7])
        out[i, 3:7] = _normalize_quat_wxyz(_quat_mul_wxyz(Q_YUP_TO_ZUP, q_old))
    return out


def _load_qpos_tracks(
    npz_path: Path,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, float, str | None, str, str, str, str | None, str | None]:
    data = np.load(str(npz_path), allow_pickle=True)
    if "qpos_A" not in data or "qpos_B" not in data:
        raise ValueError(f"{npz_path} must contain qpos_A and qpos_B.")

    stride = max(1, int(frame_stride))
    qpos_a = np.asarray(data["qpos_A"], dtype=np.float32)[::stride]
    qpos_b = np.asarray(data["qpos_B"], dtype=np.float32)[::stride]
    n = min(qpos_a.shape[0], qpos_b.shape[0])
    if max_frames is not None:
        n = min(n, max(0, int(max_frames)))
    qpos_a = qpos_a[:n]
    qpos_b = qpos_b[:n]

    fps = float(data["fps"]) if "fps" in data else 30.0
    if stride > 1:
        fps = fps / stride
    qpos_frame = _parse_coordinate_frame(data["qpos_coordinate_frame"]) if "qpos_coordinate_frame" in data else None
    dual_scene_xml = _coerce_scalar_string(data["dual_scene_xml"]) if "dual_scene_xml" in data else None
    prefix_a = _coerce_scalar_string(data["dual_prefix_A"]) if "dual_prefix_A" in data else None
    prefix_b = _coerce_scalar_string(data["dual_prefix_B"]) if "dual_prefix_B" in data else None
    prefix_a = prefix_a or "A_"
    prefix_b = prefix_b or "B_"
    robot_type_a = _coerce_robot_type(data["robot_A"]) if "robot_A" in data else None
    robot_type_b = _coerce_robot_type(data["robot_B"]) if "robot_B" in data else None
    return qpos_a, qpos_b, fps, qpos_frame, dual_scene_xml or "", prefix_a, prefix_b, robot_type_a, robot_type_b


def _resolve_default_robot_xml(robot_type: str) -> Path:
    asset_root = Path(__file__).resolve().parents[1]
    robot_cfg = RobotConfig(robot_type=robot_type)
    robot_urdf = asset_root / robot_cfg.ROBOT_URDF_FILE
    robot_xml = robot_urdf.with_suffix(".xml")
    if not robot_xml.exists():
        raise FileNotFoundError(f"Robot XML not found for robot '{robot_type}': {robot_xml}")
    return robot_xml


def _resolve_dual_scene_xml(
    *,
    qpos_npz_path: Path,
    dual_scene_xml_raw: str,
    robot_type_a: str | None,
    robot_type_b: str | None,
    prefix_a: str,
    prefix_b: str,
) -> Path:
    if dual_scene_xml_raw:
        dual_scene_xml = Path(dual_scene_xml_raw)
        if not dual_scene_xml.exists():
            raise FileNotFoundError(f"Dual scene XML not found: {dual_scene_xml}")
        return dual_scene_xml

    robot_type_a = robot_type_a or "g1"
    robot_type_b = robot_type_b or "g1"
    robot_xml_a = _resolve_default_robot_xml(robot_type_a)
    robot_xml_b = _resolve_default_robot_xml(robot_type_b)
    generated_xml = qpos_npz_path.with_name(f"{qpos_npz_path.stem}_dual_scene.xml")
    return _build_dual_scene_xml_from_pair(
        robot_xml_a=robot_xml_a,
        robot_xml_b=robot_xml_b,
        out_path=generated_xml,
        prefix_a=prefix_a,
        prefix_b=prefix_b,
    )


def _resolve_dual_scene_layout(
    model: mujoco.MjModel,
    prefix_a: str,
    prefix_b: str,
) -> _DualSceneLayout:
    free_joint_ids = [
        j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
    ]
    if len(free_joint_ids) < 2:
        raise ValueError("Expected a dual scene with at least two free joints.")

    free_joint_ids_sorted = sorted(free_joint_ids, key=lambda j: int(model.jnt_qposadr[j]))
    joint_id_a: int | None = None
    joint_id_b: int | None = None
    for joint_id in free_joint_ids_sorted:
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if joint_id_a is None and body_name.startswith(prefix_a):
            joint_id_a = int(joint_id)
        if joint_id_b is None and body_name.startswith(prefix_b):
            joint_id_b = int(joint_id)

    if joint_id_a is None or joint_id_b is None:
        joint_id_a = int(free_joint_ids_sorted[0])
        joint_id_b = int(free_joint_ids_sorted[1])

    qadr_a = int(model.jnt_qposadr[joint_id_a])
    qadr_b = int(model.jnt_qposadr[joint_id_b])
    qadr_all_free = sorted(int(model.jnt_qposadr[j]) for j in free_joint_ids)

    def _next_qadr(qadr: int) -> int:
        return next((q for q in qadr_all_free if q > qadr), int(model.nq))

    qend_a = _next_qadr(qadr_a)
    qend_b = _next_qadr(qadr_b)
    if qend_a <= qadr_a or qend_b <= qadr_b:
        raise ValueError("Invalid free-joint qpos layout in dual scene model.")

    q_block_a = np.arange(qadr_a, qend_a, dtype=int)
    q_block_b = np.arange(qadr_b, qend_b, dtype=int)
    min_width_a = len(q_block_a)
    min_width_b = len(q_block_b)
    if min_width_a < 7 or min_width_b < 7:
        raise ValueError("Each robot block must contain at least 7 qpos entries.")

    return _DualSceneLayout(
        q_block_a=q_block_a,
        q_block_b=q_block_b,
        min_width_a=min_width_a,
        min_width_b=min_width_b,
    )


def _compose_full_qpos(
    model: mujoco.MjModel,
    layout: _DualSceneLayout,
    qpos_a_local: np.ndarray,
    qpos_b_local: np.ndarray,
) -> np.ndarray:
    if qpos_a_local.shape[0] < layout.min_width_a:
        raise ValueError(
            f"qpos_A frame has width {qpos_a_local.shape[0]}, expected at least {layout.min_width_a}."
        )
    if qpos_b_local.shape[0] < layout.min_width_b:
        raise ValueError(
            f"qpos_B frame has width {qpos_b_local.shape[0]}, expected at least {layout.min_width_b}."
        )

    q_full = np.array(model.qpos0, dtype=np.float32, copy=True)
    q_full[layout.q_block_a[: layout.min_width_a]] = qpos_a_local[: layout.min_width_a]
    q_full[layout.q_block_b[: layout.min_width_b]] = qpos_b_local[: layout.min_width_b]
    return q_full


def _geom_belongs_to_prefix(model: mujoco.MjModel, geom_id: int, prefixes: tuple[str, ...]) -> bool:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    if geom_name.startswith(prefixes):
        return True
    body_id = int(model.geom_bodyid[geom_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    return body_name.startswith(prefixes)


def _is_floor_like_geom(model: mujoco.MjModel, geom_id: int) -> bool:
    if int(model.geom_type[geom_id]) == mujoco.mjtGeom.mjGEOM_PLANE:
        return True
    geom_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
    body_id = int(model.geom_bodyid[geom_id])
    body_name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").lower()
    return "ground" in geom_name or "floor" in geom_name or "ground" in body_name or "floor" in body_name


def _find_texture_id_for_material(model: mujoco.MjModel, mat_id: int) -> int | None:
    if mat_id < 0 or mat_id >= model.nmat:
        return None
    tex_row = np.asarray(model.mat_texid[mat_id], dtype=int)
    valid = tex_row[tex_row >= 0]
    if valid.size == 0:
        return None
    return int(valid[0])


def _write_texture_rgb(
    model: mujoco.MjModel,
    tex_id: int,
    rgb: np.ndarray,
) -> None:
    width = int(model.tex_width[tex_id])
    height = int(model.tex_height[tex_id])
    nchannel = int(model.tex_nchannel[tex_id])
    if nchannel < 3:
        raise ValueError(f"Texture {tex_id} has {nchannel} channels; expected at least 3.")
    expected = width * height * nchannel
    adr = int(model.tex_adr[tex_id])
    flat = np.asarray(rgb, dtype=np.uint8).reshape(height, width, 3)
    tex = np.full((height, width, nchannel), 255, dtype=np.uint8)
    tex[..., :3] = flat
    model.tex_data[adr : adr + expected] = tex.reshape(-1)


def _make_white_grid_texture(
    width: int,
    height: int,
    *,
    line_px: int,
) -> np.ndarray:
    tex = np.full((height, width, 3), 255, dtype=np.uint8)
    line_px = max(1, int(line_px))
    tex[:line_px, :, :] = 0
    tex[-line_px:, :, :] = 0
    tex[:, :line_px, :] = 0
    tex[:, -line_px:, :] = 0
    return tex


def _make_background_rgb(width: int, height: int, style: Literal["original", "viewer", "white"]) -> np.ndarray | None:
    if style == "original":
        return None
    if style == "white":
        return np.full((height, width, 3), 255, dtype=np.uint8)

    top = np.array([228, 235, 243], dtype=np.float32)
    bottom = np.array([247, 249, 251], dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    rgb = top[None, None, :] * (1.0 - vertical) + bottom[None, None, :] * vertical
    return np.broadcast_to(rgb.astype(np.uint8), (height, width, 3)).copy()


def _apply_floor_style(model: mujoco.MjModel, cfg: Config) -> None:
    floor_geom_ids = [geom_id for geom_id in range(model.ngeom) if _is_floor_like_geom(model, geom_id)]
    if not floor_geom_ids:
        return

    primary_floor = floor_geom_ids[0]
    duplicate_floors = floor_geom_ids[1:]

    for geom_id in duplicate_floors:
        model.geom_pos[geom_id, 2] = -5.0
        model.geom_rgba[geom_id] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    primary_mat_id = int(model.geom_matid[primary_floor])
    primary_tex_id = _find_texture_id_for_material(model, primary_mat_id)
    model.geom_rgba[primary_floor] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    if primary_mat_id >= 0:
        model.mat_rgba[primary_mat_id] = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        model.mat_specular[primary_mat_id] = 0.0
        model.mat_shininess[primary_mat_id] = 0.0
        model.mat_reflectance[primary_mat_id] = 0.0
        model.mat_emission[primary_mat_id] = 0.0
        model.mat_texrepeat[primary_mat_id] = np.array(
            [float(cfg.floor_grid_repeat), float(cfg.floor_grid_repeat)],
            dtype=np.float32,
        )

    if cfg.floor_style == "original":
        return

    if primary_tex_id is None:
        return

    tex_width = int(model.tex_width[primary_tex_id])
    tex_height = int(model.tex_height[primary_tex_id])
    if cfg.floor_style == "white":
        tex_rgb = np.full((tex_height, tex_width, 3), 255, dtype=np.uint8)
    else:
        tex_rgb = _make_white_grid_texture(
            tex_width,
            tex_height,
            line_px=cfg.floor_grid_line_px,
        )
    _write_texture_rgb(model, primary_tex_id, tex_rgb)


def _collect_focus_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    prefixes: tuple[str, ...],
) -> np.ndarray:
    points: list[np.ndarray] = []
    for geom_id in range(model.ngeom):
        if _is_floor_like_geom(model, geom_id):
            continue
        if prefixes and not _geom_belongs_to_prefix(model, geom_id, prefixes):
            continue
        center = np.asarray(data.geom_xpos[geom_id], dtype=np.float32)
        radius = float(model.geom_rbound[geom_id])
        if radius <= 1.0e-6:
            radius = float(np.max(model.geom_size[geom_id]))
        radius = max(radius, 0.02)
        points.append(center)
        for axis in np.eye(3, dtype=np.float32):
            points.append(center + radius * axis)
            points.append(center - radius * axis)

    if points:
        return np.asarray(points, dtype=np.float32)

    fallback_points = [np.asarray(data.xpos[body_id], dtype=np.float32) for body_id in range(model.nbody)]
    return np.asarray(fallback_points, dtype=np.float32)


def _calculate_camera_state(
    points: np.ndarray,
    *,
    width: int,
    height: int,
    direction: tuple[float, float, float],
    yfov: float,
    distance_margin: float,
    min_distance: float,
    target_vertical_bias: float,
) -> _CameraState:
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

    pos = target + distance * direction_vec
    forward = target - pos
    forward /= float(np.linalg.norm(forward))
    return _CameraState(pos=pos.astype(np.float32), forward=forward.astype(np.float32), up=y_cam.astype(np.float32))


def _apply_camera_state(scene: mujoco.MjvScene, camera_state: _CameraState) -> None:
    for gl_camera in scene.camera:
        gl_camera.pos[:] = camera_state.pos
        gl_camera.forward[:] = camera_state.forward
        gl_camera.up[:] = camera_state.up


def main(cfg: Config) -> None:
    qpos_npz_path = Path(cfg.qpos_npz)
    output_mp4_path = (
        Path(cfg.output_mp4)
        if cfg.output_mp4 is not None
        else qpos_npz_path.with_name(f"{qpos_npz_path.stem}_robots.mp4")
    )
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    (
        qpos_a,
        qpos_b,
        fps_from_data,
        qpos_coordinate_frame,
        dual_scene_xml_from_npz,
        prefix_a,
        prefix_b,
        robot_type_a,
        robot_type_b,
    ) = _load_qpos_tracks(
        qpos_npz_path,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
    )
    if qpos_a.shape[0] == 0 or qpos_b.shape[0] == 0:
        raise ValueError("No frames available after applying frame_stride/max_frames.")

    dual_scene_xml = _resolve_dual_scene_xml(
        qpos_npz_path=qpos_npz_path,
        dual_scene_xml_raw=cfg.dual_scene_xml or dual_scene_xml_from_npz,
        robot_type_a=robot_type_a,
        robot_type_b=robot_type_b,
        prefix_a=prefix_a,
        prefix_b=prefix_b,
    )

    should_convert_qpos = bool(cfg.y_up_to_z_up)
    if qpos_coordinate_frame == "z_up":
        should_convert_qpos = False
    if should_convert_qpos:
        qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
        qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)

    fps = float(cfg.fps_override) if cfg.fps_override is not None else fps_from_data
    fps = max(1.0e-6, fps)

    print(
        f"[dual_robot_mp4_export] frames={qpos_a.shape[0]} fps={fps:.2f} size={cfg.width}x{cfg.height}"
    )
    print(f"[dual_robot_mp4_export] qpos_npz={qpos_npz_path}")
    print(f"[dual_robot_mp4_export] dual_scene_xml={dual_scene_xml}")
    print(
        "[dual_robot_mp4_export] qpos_coordinate_frame="
        f"{qpos_coordinate_frame or 'legacy_assumed_y_up'} | qpos_y_up_to_z_up_applied={should_convert_qpos}"
    )
    print(
        "[dual_robot_mp4_export] camera="
        f"direction={cfg.camera_direction} yfov={cfg.camera_yfov:.4f}"
        f" margin={cfg.camera_margin:.3f} min_distance={cfg.camera_min_distance:.3f}"
        f" lock_camera={cfg.lock_camera}"
    )
    print(
        "[dual_robot_mp4_export] floor="
        f"style={cfg.floor_style} grid_repeat={cfg.floor_grid_repeat:.2f}"
        f" grid_line_px={cfg.floor_grid_line_px}"
    )
    print(f"[dual_robot_mp4_export] background_style={cfg.background_style}")

    model = mujoco.MjModel.from_xml_path(str(dual_scene_xml))
    # Raise the offscreen framebuffer ceiling to the requested output size.
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(cfg.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(cfg.height))
    _apply_floor_style(model, cfg)
    data = mujoco.MjData(model)
    layout = _resolve_dual_scene_layout(model, prefix_a=prefix_a, prefix_b=prefix_b)
    prefixes = (prefix_a, prefix_b)

    static_camera_state: _CameraState | None = None
    if cfg.lock_camera:
        all_focus_points: list[np.ndarray] = []
        for frame_idx in range(qpos_a.shape[0]):
            q_full = _compose_full_qpos(model, layout, qpos_a[frame_idx], qpos_b[frame_idx])
            data.qpos[:] = q_full
            mujoco.mj_forward(model, data)
            all_focus_points.append(_collect_focus_points(model, data, prefixes=prefixes))
        static_camera_state = _calculate_camera_state(
            np.concatenate(all_focus_points, axis=0),
            width=cfg.width,
            height=cfg.height,
            direction=cfg.camera_direction,
            yfov=cfg.camera_yfov,
            distance_margin=cfg.camera_margin,
            min_distance=cfg.camera_min_distance,
            target_vertical_bias=cfg.camera_target_vertical_bias,
        )

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    mujoco.mjv_defaultFreeCamera(model, camera)

    writer = cv2.VideoWriter(
        str(output_mp4_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(cfg.width), int(cfg.height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open video writer for {output_mp4_path}")

    with mujoco.Renderer(model, width=cfg.width, height=cfg.height) as renderer:
        try:
            background_rgb = _make_background_rgb(int(cfg.width), int(cfg.height), cfg.background_style)
            for frame_idx in range(qpos_a.shape[0]):
                q_full = _compose_full_qpos(model, layout, qpos_a[frame_idx], qpos_b[frame_idx])
                data.qpos[:] = q_full
                mujoco.mj_forward(model, data)

                camera_state = static_camera_state
                if camera_state is None:
                    camera_state = _calculate_camera_state(
                        _collect_focus_points(model, data, prefixes=prefixes),
                        width=cfg.width,
                        height=cfg.height,
                        direction=cfg.camera_direction,
                        yfov=cfg.camera_yfov,
                        distance_margin=cfg.camera_margin,
                        min_distance=cfg.camera_min_distance,
                        target_vertical_bias=cfg.camera_target_vertical_bias,
                    )

                renderer.update_scene(data, camera=camera)
                _apply_camera_state(renderer.scene, camera_state)
                frame_rgb = renderer.render()
                if background_rgb is not None:
                    renderer.enable_segmentation_rendering()
                    seg = renderer.render()
                    renderer.disable_segmentation_rendering()
                    background_mask = seg[:, :, 0] < 0
                    if np.any(background_mask):
                        frame_rgb = np.array(frame_rgb, copy=True)
                        frame_rgb[background_mask] = background_rgb[background_mask]
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

                if frame_idx == 0 or frame_idx == qpos_a.shape[0] - 1 or (frame_idx + 1) % 50 == 0:
                    print(f"[dual_robot_mp4_export] rendered frame {frame_idx + 1}/{qpos_a.shape[0]}")
        finally:
            writer.release()

    print(f"[dual_robot_mp4_export] wrote {output_mp4_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
