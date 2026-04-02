#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Literal

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.examples.dual_robot_mp4_export import (  # noqa: E402
    DEFAULT_CAMERA_DIRECTION,
    DEFAULT_GRID_REPEAT,
    Q_YUP_TO_ZUP,
    R_YUP_TO_ZUP,
    _apply_camera_state,
    _apply_floor_style,
    _calculate_camera_state,
    _collect_focus_points,
    _make_background_rgb,
)


SMPLX22_JOINTS = [
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

HUMAN_PROXY_SPHERES: tuple[str, ...] = (
    "Pelvis",
    "Head",
    "L_Wrist",
    "R_Wrist",
    "L_Foot",
    "R_Foot",
)

HUMAN_PROXY_CAPSULES: tuple[tuple[str, str, str], ...] = (
    ("pelvis_spine1", "Pelvis", "Spine1"),
    ("spine1_spine2", "Spine1", "Spine2"),
    ("spine2_spine3", "Spine2", "Spine3"),
    ("spine3_neck", "Spine3", "Neck"),
    ("neck_head", "Neck", "Head"),
    ("l_collar_shoulder", "L_Collar", "L_Shoulder"),
    ("r_collar_shoulder", "R_Collar", "R_Shoulder"),
    ("l_shoulder_elbow", "L_Shoulder", "L_Elbow"),
    ("r_shoulder_elbow", "R_Shoulder", "R_Elbow"),
    ("l_elbow_wrist", "L_Elbow", "L_Wrist"),
    ("r_elbow_wrist", "R_Elbow", "R_Wrist"),
    ("l_hip_knee", "L_Hip", "L_Knee"),
    ("r_hip_knee", "R_Hip", "R_Knee"),
    ("l_knee_ankle", "L_Knee", "L_Ankle"),
    ("r_knee_ankle", "R_Knee", "R_Ankle"),
    ("l_ankle_foot", "L_Ankle", "L_Foot"),
    ("r_ankle_foot", "R_Ankle", "R_Foot"),
)


@dataclass
class Config:
    qpos_npz: str
    """Mixed retarget output npz with one robot qpos track and one human proxy track."""

    output_mp4: str | None = None
    """Destination MP4 path. Defaults to <qpos_npz stem>_proxy.mp4."""

    mixed_scene_xml: str | None = None
    """Optional override for the mixed MuJoCo scene XML path."""

    frame_stride: int = 1
    """Subsample frames for faster export."""

    max_frames: int | None = None
    """Optional cap on the number of exported frames."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from npz."""

    width: int = 1280
    """Output width in pixels."""

    height: int = 720
    """Output height in pixels."""

    camera_mode: Literal["auto_fit", "preset"] = "preset"
    """Use automatic framing or load a saved camera preset."""

    camera_preset_json: str | None = None
    """Optional camera preset JSON used by --camera-mode preset."""

    camera_direction: tuple[float, float, float] = DEFAULT_CAMERA_DIRECTION
    """Camera direction from target to eye for auto_fit mode."""

    camera_yfov: float = float(np.pi / 3.0)
    """Vertical field of view in radians."""

    camera_margin: float = 1.25
    """Distance multiplier applied after fitting content into frame."""

    camera_min_distance: float = 4.0
    """Minimum camera distance from target."""

    camera_target_vertical_bias: float = 0.0
    """Vertical offset added to the fitted camera target."""

    lock_camera: bool = True
    """If True, fit the auto camera once across the whole clip."""

    y_up_to_z_up: bool = True
    """Convert qpos and proxy joints from y-up to z-up when metadata says the source is y-up."""

    floor_style: Literal["white_grid", "white", "original"] = "white_grid"
    """Visual style for the floor plane."""

    floor_grid_repeat: float = DEFAULT_GRID_REPEAT
    """How many grid cells to repeat across the plane in each axis."""

    floor_grid_line_px: int = 4
    """Grid line thickness in texture pixels for the white-grid floor."""

    background_style: Literal["original", "viewer", "white"] = "original"
    """Background fill behind rendered geometry."""


def _coerce_scalar_string(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    text = str(value).strip()
    return text or None


def _parse_coordinate_frame(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    frame = str(value).strip().lower()
    if not frame:
        return None
    if frame not in {"y_up", "z_up"}:
        raise ValueError(f"Unsupported coordinate frame metadata: {frame}")
    return frame


def _normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    if n <= 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32, copy=False)


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


def _convert_y_up_to_z_up_qpos(qpos: np.ndarray) -> np.ndarray:
    out = np.array(qpos, dtype=np.float32, copy=True)
    out[:, :3] = (R_YUP_TO_ZUP @ out[:, :3].T).T
    for i in range(out.shape[0]):
        q_old = _normalize_quat_wxyz(out[i, 3:7])
        out[i, 3:7] = _normalize_quat_wxyz(_quat_mul_wxyz(Q_YUP_TO_ZUP, q_old))
    return out


def _convert_y_up_to_z_up_points(points: np.ndarray) -> np.ndarray:
    flat = points.reshape(-1, 3)
    converted = (R_YUP_TO_ZUP @ flat.T).T
    return converted.reshape(points.shape).astype(np.float32, copy=False)


def _sphere_body_name(prefix: str, joint_name: str) -> str:
    return f"{prefix}sphere_{joint_name}"


def _capsule_body_name(prefix: str, capsule_name: str) -> str:
    return f"{prefix}capsule_{capsule_name}"


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _quat_wxyz_from_segment(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    z_axis = _normalize(np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rot = np.column_stack([x_axis, y_axis, z_axis])
    quat_xyzw = Rotation.from_matrix(rot).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _load_camera_preset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    position = np.asarray(payload["position"], dtype=np.float32)
    look_at = np.asarray(payload["look_at"], dtype=np.float32)
    up_direction = np.asarray(payload["up_direction"], dtype=np.float32)
    forward = look_at - position
    forward /= float(np.linalg.norm(forward))
    return position, forward, up_direction


def _load_mixed_tracks(
    npz_path: Path,
    frame_stride: int,
    max_frames: int | None,
) -> tuple[np.ndarray, np.ndarray, float, str | None, Path, str]:
    data = np.load(str(npz_path), allow_pickle=True)
    stride = max(1, int(frame_stride))

    target_a = _coerce_scalar_string(data["target_A"]) if "target_A" in data else None
    target_b = _coerce_scalar_string(data["target_B"]) if "target_B" in data else None
    target_a = (target_a or "").lower()
    target_b = (target_b or "").lower()

    if target_a == "robot" and target_b == "human":
        qpos_key = "qpos_A"
        joints_key = "human_joints_B"
    elif target_b == "robot" and target_a == "human":
        qpos_key = "qpos_B"
        joints_key = "human_joints_A"
    else:
        raise ValueError(
            f"{npz_path} does not look like a mixed robot-human output. "
            f"Expected one robot target and one human target, got A={target_a!r}, B={target_b!r}."
        )

    if qpos_key not in data or joints_key not in data:
        raise ValueError(f"{npz_path} must contain {qpos_key} and {joints_key}.")

    qpos = np.asarray(data[qpos_key], dtype=np.float32)[::stride]
    human_joints = np.asarray(data[joints_key], dtype=np.float32)[::stride]
    n_frames = min(qpos.shape[0], human_joints.shape[0])
    if max_frames is not None:
        n_frames = min(n_frames, max(0, int(max_frames)))
    qpos = qpos[:n_frames]
    human_joints = human_joints[:n_frames]

    fps = float(data["fps"]) if "fps" in data else 30.0
    if stride > 1:
        fps = fps / stride
    qpos_coordinate_frame = _parse_coordinate_frame(data["qpos_coordinate_frame"]) if "qpos_coordinate_frame" in data else None
    mixed_scene_xml = Path(_coerce_scalar_string(data["mixed_scene_xml"]) or "")
    human_prefix = _coerce_scalar_string(data["human_proxy_prefix"]) or "H_"
    if not mixed_scene_xml:
        raise ValueError(f"{npz_path} is missing mixed_scene_xml metadata.")
    return qpos, human_joints, fps, qpos_coordinate_frame, mixed_scene_xml, human_prefix


def _resolve_robot_qpos_block(model: mujoco.MjModel) -> np.ndarray:
    free_joint_ids = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    if not free_joint_ids:
        return np.arange(model.nq, dtype=int)

    qadr_all_free = sorted(int(model.jnt_qposadr[j]) for j in free_joint_ids)
    qadr = qadr_all_free[0]
    qend = next((q for q in qadr_all_free if q > qadr), int(model.nq))
    if qend <= qadr:
        raise ValueError("Invalid robot qpos layout in mixed scene model.")
    return np.arange(qadr, qend, dtype=int)


def _resolve_human_proxy_handles(
    model: mujoco.MjModel,
    human_prefix: str,
) -> tuple[dict[str, int], dict[str, int]]:
    joint_mocaps: dict[str, int] = {}
    capsule_mocaps: dict[str, int] = {}
    for joint_name in HUMAN_PROXY_SPHERES:
        body_name = _sphere_body_name(human_prefix, joint_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id >= 0:
                joint_mocaps[joint_name] = mocap_id
    for capsule_name, _, _ in HUMAN_PROXY_CAPSULES:
        body_name = _capsule_body_name(human_prefix, capsule_name)
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id != -1:
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id >= 0:
                capsule_mocaps[capsule_name] = mocap_id
    return joint_mocaps, capsule_mocaps


def _set_human_proxy_pose(
    data: mujoco.MjData,
    human_joints: np.ndarray,
    joint_mocaps: dict[str, int],
    capsule_mocaps: dict[str, int],
) -> None:
    joint_to_idx = {name: i for i, name in enumerate(SMPLX22_JOINTS[: human_joints.shape[0]])}

    for joint_name, mocap_id in joint_mocaps.items():
        idx = joint_to_idx.get(joint_name)
        if idx is None:
            continue
        data.mocap_pos[mocap_id] = np.asarray(human_joints[idx], dtype=float)
        data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    for capsule_name, joint_a, joint_b in HUMAN_PROXY_CAPSULES:
        mocap_id = capsule_mocaps.get(capsule_name)
        idx_a = joint_to_idx.get(joint_a)
        idx_b = joint_to_idx.get(joint_b)
        if mocap_id is None or idx_a is None or idx_b is None:
            continue
        p0 = np.asarray(human_joints[idx_a], dtype=float)
        p1 = np.asarray(human_joints[idx_b], dtype=float)
        data.mocap_pos[mocap_id] = 0.5 * (p0 + p1)
        data.mocap_quat[mocap_id] = _quat_wxyz_from_segment(p0, p1)


def main(cfg: Config) -> None:
    npz_path = Path(cfg.qpos_npz)
    output_mp4_path = (
        Path(cfg.output_mp4)
        if cfg.output_mp4 is not None
        else npz_path.with_name(f"{npz_path.stem}_proxy.mp4")
    )
    output_mp4_path.parent.mkdir(parents=True, exist_ok=True)

    qpos, human_joints, fps_from_data, qpos_coordinate_frame, mixed_scene_xml_from_npz, human_prefix = _load_mixed_tracks(
        npz_path=npz_path,
        frame_stride=cfg.frame_stride,
        max_frames=cfg.max_frames,
    )
    if qpos.shape[0] == 0:
        raise ValueError("No frames available after applying frame_stride/max_frames.")

    mixed_scene_xml = Path(cfg.mixed_scene_xml) if cfg.mixed_scene_xml is not None else mixed_scene_xml_from_npz
    if not mixed_scene_xml.exists():
        raise FileNotFoundError(f"Mixed scene XML not found: {mixed_scene_xml}")

    should_convert = bool(cfg.y_up_to_z_up)
    if qpos_coordinate_frame == "z_up":
        should_convert = False
    if should_convert:
        qpos = _convert_y_up_to_z_up_qpos(qpos)
        human_joints = _convert_y_up_to_z_up_points(human_joints)

    fps = float(cfg.fps_override) if cfg.fps_override is not None else fps_from_data
    fps = max(1.0e-6, fps)

    print(
        f"[mixed_proxy_mp4_export] frames={qpos.shape[0]} fps={fps:.2f} size={cfg.width}x{cfg.height}"
    )
    print(f"[mixed_proxy_mp4_export] qpos_npz={npz_path}")
    print(f"[mixed_proxy_mp4_export] mixed_scene_xml={mixed_scene_xml}")
    print(
        "[mixed_proxy_mp4_export] qpos_coordinate_frame="
        f"{qpos_coordinate_frame or 'legacy_assumed_y_up'} | qpos_y_up_to_z_up_applied={should_convert}"
    )
    print(
        "[mixed_proxy_mp4_export] camera_mode="
        f"{cfg.camera_mode} | lock_camera={cfg.lock_camera} | preset={cfg.camera_preset_json}"
    )

    model = mujoco.MjModel.from_xml_path(str(mixed_scene_xml))
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), int(cfg.width))
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), int(cfg.height))
    _apply_floor_style(model, cfg)
    data = mujoco.MjData(model)
    qpos_block = _resolve_robot_qpos_block(model)
    joint_mocaps, capsule_mocaps = _resolve_human_proxy_handles(model, human_prefix)

    if qpos.shape[1] < qpos_block.shape[0]:
        raise ValueError(
            f"qpos width {qpos.shape[1]} is smaller than mixed scene robot block {qpos_block.shape[0]}."
        )

    static_camera_state = None
    if cfg.camera_mode == "preset":
        if cfg.camera_preset_json is None:
            raise ValueError("--camera-mode preset requires --camera-preset-json.")
        pos, forward, up = _load_camera_preset(Path(cfg.camera_preset_json))
        static_camera_state = type(
            "_CameraStateProxy",
            (),
            {"pos": pos.astype(np.float32), "forward": forward.astype(np.float32), "up": up.astype(np.float32)},
        )()
    elif cfg.lock_camera:
        all_focus_points: list[np.ndarray] = []
        for frame_idx in range(qpos.shape[0]):
            data.qpos[:] = np.array(model.qpos0, copy=True)
            data.qpos[qpos_block] = qpos[frame_idx, : qpos_block.shape[0]]
            _set_human_proxy_pose(data, human_joints[frame_idx, :22], joint_mocaps, capsule_mocaps)
            mujoco.mj_forward(model, data)
            all_focus_points.append(_collect_focus_points(model, data, prefixes=()))
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
            for frame_idx in range(qpos.shape[0]):
                data.qpos[:] = np.array(model.qpos0, copy=True)
                data.qpos[qpos_block] = qpos[frame_idx, : qpos_block.shape[0]]
                _set_human_proxy_pose(data, human_joints[frame_idx, :22], joint_mocaps, capsule_mocaps)
                mujoco.mj_forward(model, data)

                camera_state = static_camera_state
                if camera_state is None:
                    camera_state = _calculate_camera_state(
                        _collect_focus_points(model, data, prefixes=()),
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

                if frame_idx == 0 or frame_idx == qpos.shape[0] - 1 or (frame_idx + 1) % 50 == 0:
                    print(f"[mixed_proxy_mp4_export] rendered frame {frame_idx + 1}/{qpos.shape[0]}")
        finally:
            writer.release()

    print(f"[mixed_proxy_mp4_export] wrote {output_mp4_path}")


if __name__ == "__main__":
    main(tyro.cli(Config))
