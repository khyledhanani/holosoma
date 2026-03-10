#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx  # type: ignore[import-untyped]
import torch
import tyro
import viser  # type: ignore[import-not-found]
import yourdfpy  # type: ignore[import-untyped]
from viser.extras import ViserUrdf  # type: ignore[import-not-found]


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
    "g1": "src/holosoma_retargeting/holosoma_retargeting/models/g1/g1_29dof.urdf",
    "t1": "src/holosoma_retargeting/holosoma_retargeting/models/t1/t1_23dof.urdf",
}


@dataclass
class Config:
    data_npz: str
    """Path to an Inter-X sequence npz (used for sequence id/fps and optional direct mesh arrays)."""

    sequence_id: str | None = None
    """Optional override for sequence id. If None, uses npz sequence_id field or file stem."""

    interx_motion_root: str = "DATA/motions"
    """Root containing raw Inter-X per-sequence folders with P1.npz/P2.npz."""

    smplx_model_root: str = "DATA/smpl_all_models"
    """Root for SMPL-X model files. Supports either <root>/smplx/SMPLX_*.npz or <root>/SMPLX_*.npz."""

    prefer_neutral_gender: bool = True
    """If True, force neutral SMPL-X model for both persons."""

    device: str = "cpu"
    """Torch device for SMPL-X inference ('cpu', 'cuda', or 'auto')."""

    batch_size: int = 256
    """Chunk size for SMPL-X forward passes."""

    fps_override: float | None = None
    """Optional FPS override. If None, use fps from data_npz (default 30 if absent)."""

    frame_stride: int = 1
    """Subsample frames for faster playback."""

    mesh_opacity: float = 0.9
    """Mesh opacity in [0, 1]."""

    loop: bool = True
    """Loop playback."""

    y_up_to_z_up: bool = True
    """Convert incoming points from y-up to z-up before rendering."""

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


def _normalize_gender(raw_gender: object) -> str:
    g = str(raw_gender).strip().lower()
    if g.startswith("m"):
        return "male"
    if g.startswith("f"):
        return "female"
    return "neutral"


def _as_aa_triplets(x: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[1] % 3 == 0:
        return arr.reshape(arr.shape[0], -1, 3)
    raise ValueError(f"{name} must be shape (T, J, 3) or (T, 3*J), got {arr.shape}")


def _choose_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _infer_sequence_id(npz_path: Path, explicit: str | None = None) -> str:
    if explicit is not None:
        return explicit
    data = np.load(str(npz_path), allow_pickle=True)
    if "sequence_id" in data:
        return str(data["sequence_id"])
    return npz_path.stem


def _load_reference_timeline(npz_path: Path, frame_stride: int) -> tuple[str, float, int | None]:
    data = np.load(str(npz_path), allow_pickle=True)
    stride = max(1, int(frame_stride))
    sequence_id = str(data["sequence_id"]) if "sequence_id" in data else npz_path.stem

    fps = float(data["fps"]) if "fps" in data else 30.0
    if stride > 1:
        fps = fps / stride

    frame_keys = [
        ("human_joints_A", "human_joints_B"),
        ("global_joint_positions_A", "global_joint_positions_B"),
        ("human_verts_A", "human_verts_B"),
        ("global_joint_verts_A", "global_joint_verts_B"),
        ("verts_A", "verts_B"),
        ("vertices_A", "vertices_B"),
        ("qpos_A", "qpos_B"),
    ]
    target_frames: int | None = None
    for ka, kb in frame_keys:
        if ka in data and kb in data:
            n = min(np.asarray(data[ka])[::stride].shape[0], np.asarray(data[kb])[::stride].shape[0])
            target_frames = int(n)
            break
    return sequence_id, fps, target_frames


def _resample_frames(frames: np.ndarray, target_frames: int) -> np.ndarray:
    if target_frames <= 0:
        raise ValueError(f"target_frames must be > 0, got {target_frames}")
    if frames.shape[0] == target_frames:
        return frames
    idx = np.rint(np.linspace(0, frames.shape[0] - 1, num=target_frames)).astype(np.int64)
    return frames[idx]


def _extract_faces(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    face_keys = [
        "mesh_faces",
        "faces",
        "smplx_faces",
        "human_faces",
    ]
    for key in face_keys:
        if key in data:
            faces = np.asarray(data[key], dtype=np.int32)
            if faces.ndim != 2 or faces.shape[1] != 3:
                raise ValueError(f"Faces in key '{key}' must have shape (F, 3), got {faces.shape}")
            return faces
    return None


def _load_direct_mesh_from_npz(npz_path: Path, frame_stride: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    data = np.load(str(npz_path), allow_pickle=True)
    stride = max(1, int(frame_stride))
    mesh_key_pairs = [
        ("human_verts_A", "human_verts_B"),
        ("global_joint_verts_A", "global_joint_verts_B"),
        ("verts_A", "verts_B"),
        ("vertices_A", "vertices_B"),
        ("human_vertices_A", "human_vertices_B"),
        ("smplx_vertices_A", "smplx_vertices_B"),
    ]

    for key_a, key_b in mesh_key_pairs:
        if key_a in data and key_b in data:
            verts_a = np.asarray(data[key_a], dtype=np.float32)[::stride]
            verts_b = np.asarray(data[key_b], dtype=np.float32)[::stride]
            if verts_a.ndim != 3 or verts_a.shape[-1] != 3:
                raise ValueError(f"{key_a} must have shape (T, V, 3), got {verts_a.shape}")
            if verts_b.ndim != 3 or verts_b.shape[-1] != 3:
                raise ValueError(f"{key_b} must have shape (T, V, 3), got {verts_b.shape}")
            n = min(verts_a.shape[0], verts_b.shape[0])
            faces = _extract_faces(data)
            if faces is None:
                raise KeyError(
                    f"Found mesh vertices in '{key_a}/{key_b}' but no faces key. "
                    "Expected one of: mesh_faces, faces, smplx_faces, human_faces."
                )
            return verts_a[:n], verts_b[:n], faces
    return None


def _load_smplx_model(
    model_root: Path,
    gender: str,
    num_betas: int,
    device: torch.device,
    model_cache: dict[tuple[str, str, int, str], torch.nn.Module],
) -> torch.nn.Module:
    key = (str(model_root.resolve()), gender, int(num_betas), str(device))
    if key in model_cache:
        return model_cache[key]

    candidates = [model_root, model_root / "smplx"]
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            model = smplx.create(
                model_path=str(candidate),
                model_type="smplx",
                gender=gender,
                use_pca=False,
                ext="npz",
                num_betas=int(num_betas),
            )
            model = model.to(device)
            model.eval()
            model_cache[key] = model
            return model
        except Exception as exc:  # pragma: no cover - diagnostic path
            last_exc = exc

    expected_paths = [
        model_root / "smplx" / f"SMPLX_{gender.upper()}.npz",
        model_root / f"SMPLX_{gender.upper()}.npz",
    ]
    raise FileNotFoundError(
        "Failed to load SMPL-X model. Checked paths:\n"
        + "\n".join(f"  - {p}" for p in expected_paths)
        + (f"\nLast error: {last_exc}" if last_exc is not None else "")
    )


def _load_person_mesh_from_motion(
    person_npz: Path,
    model_root: Path,
    prefer_neutral_gender: bool,
    device: torch.device,
    batch_size: int,
    model_cache: dict[tuple[str, str, int, str], torch.nn.Module],
) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(str(person_npz), allow_pickle=True)

    required = ["pose_body", "root_orient", "trans", "betas"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{person_npz} missing keys: {missing}")

    pose_body = _as_aa_triplets(np.asarray(data["pose_body"]), "pose_body")
    root_orient = _as_aa_triplets(np.asarray(data["root_orient"]), "root_orient")
    trans = np.asarray(data["trans"], dtype=np.float32)

    if "pose_lhand" in data and "pose_rhand" in data:
        pose_lhand = _as_aa_triplets(np.asarray(data["pose_lhand"]), "pose_lhand")
        pose_rhand = _as_aa_triplets(np.asarray(data["pose_rhand"]), "pose_rhand")
    elif "pose_hand" in data:
        pose_hand = _as_aa_triplets(np.asarray(data["pose_hand"]), "pose_hand")
        if pose_hand.shape[1] < 30:
            raise ValueError(f"pose_hand must contain at least 30 joints (L+R), got {pose_hand.shape}")
        pose_lhand = pose_hand[:, :15, :]
        pose_rhand = pose_hand[:, 15:30, :]
    else:
        pose_lhand = np.zeros((pose_body.shape[0], 15, 3), dtype=np.float32)
        pose_rhand = np.zeros((pose_body.shape[0], 15, 3), dtype=np.float32)

    if root_orient.shape[1] != 1:
        raise ValueError(f"root_orient must have one joint axis-angle per frame, got {root_orient.shape}")
    if pose_body.shape[1] != 21:
        raise ValueError(f"pose_body must have 21 joints, got {pose_body.shape}")
    if pose_lhand.shape[1] != 15 or pose_rhand.shape[1] != 15:
        raise ValueError(
            "hand poses must be 15 joints each; "
            f"got left={pose_lhand.shape}, right={pose_rhand.shape}"
        )
    if trans.ndim != 2 or trans.shape[1] != 3:
        raise ValueError(f"trans must have shape (T, 3), got {trans.shape}")

    t = min(trans.shape[0], pose_body.shape[0], pose_lhand.shape[0], pose_rhand.shape[0], root_orient.shape[0])
    trans = trans[:t]
    pose_body = pose_body[:t]
    pose_lhand = pose_lhand[:t]
    pose_rhand = pose_rhand[:t]
    root_orient = root_orient[:t]

    betas = np.asarray(data["betas"], dtype=np.float32).reshape(-1)
    if betas.size == 0:
        raise ValueError(f"betas in {person_npz} is empty")
    num_betas = int(betas.shape[0])
    betas = np.broadcast_to(betas[None, :], (t, num_betas)).copy()

    if prefer_neutral_gender:
        gender = "neutral"
    else:
        raw_gender = data["gender"] if "gender" in data else "neutral"
        if isinstance(raw_gender, np.ndarray):
            raw_gender = raw_gender.item()
        gender = _normalize_gender(raw_gender)

    model = _load_smplx_model(
        model_root=model_root,
        gender=gender,
        num_betas=num_betas,
        device=device,
        model_cache=model_cache,
    )
    faces = np.asarray(model.faces, dtype=np.int32)

    body_pose = pose_body.reshape(t, 63)
    left_hand_pose = pose_lhand.reshape(t, 45)
    right_hand_pose = pose_rhand.reshape(t, 45)
    root_orient_flat = root_orient.reshape(t, 3)

    vertices_chunks: list[np.ndarray] = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        for i in range(0, t, bs):
            j = min(i + bs, t)
            chunk = j - i
            # Keep face-related pose inputs batched to avoid SMPL-X cat() size mismatch
            # when model defaults store (1, *) parameters.
            expr_dim = int(getattr(model, "num_expression_coeffs", 10))
            out = model(
                betas=torch.from_numpy(betas[i:j]).to(device=device, dtype=torch.float32),
                body_pose=torch.from_numpy(body_pose[i:j]).to(device=device, dtype=torch.float32),
                left_hand_pose=torch.from_numpy(left_hand_pose[i:j]).to(device=device, dtype=torch.float32),
                right_hand_pose=torch.from_numpy(right_hand_pose[i:j]).to(device=device, dtype=torch.float32),
                global_orient=torch.from_numpy(root_orient_flat[i:j]).to(device=device, dtype=torch.float32),
                transl=torch.from_numpy(trans[i:j]).to(device=device, dtype=torch.float32),
                expression=torch.zeros((chunk, expr_dim), device=device, dtype=torch.float32),
                jaw_pose=torch.zeros((chunk, 3), device=device, dtype=torch.float32),
                leye_pose=torch.zeros((chunk, 3), device=device, dtype=torch.float32),
                reye_pose=torch.zeros((chunk, 3), device=device, dtype=torch.float32),
            )
            vertices_chunks.append(out.vertices.detach().cpu().numpy().astype(np.float32))

    vertices = np.concatenate(vertices_chunks, axis=0)
    return vertices, faces


def _load_mesh_from_motion_folder(
    sequence_id: str,
    motion_root: Path,
    model_root: Path,
    prefer_neutral_gender: bool,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    seq_dir = motion_root / sequence_id
    p1 = seq_dir / "P1.npz"
    p2 = seq_dir / "P2.npz"
    if not p1.exists() or not p2.exists():
        raise FileNotFoundError(
            f"Could not find Inter-X motion files for sequence '{sequence_id}'. "
            f"Expected: {p1} and {p2}"
        )

    model_cache: dict[tuple[str, str, int, str], torch.nn.Module] = {}
    verts_a, faces_a = _load_person_mesh_from_motion(
        person_npz=p1,
        model_root=model_root,
        prefer_neutral_gender=prefer_neutral_gender,
        device=device,
        batch_size=batch_size,
        model_cache=model_cache,
    )
    verts_b, faces_b = _load_person_mesh_from_motion(
        person_npz=p2,
        model_root=model_root,
        prefer_neutral_gender=prefer_neutral_gender,
        device=device,
        batch_size=batch_size,
        model_cache=model_cache,
    )

    if faces_a.shape != faces_b.shape or np.any(faces_a != faces_b):
        # Keep A topology for rendering if they differ for any reason.
        print("[dual_interx_mesh_renderer] warning: A/B faces differ; using faces from person A.")

    n = min(verts_a.shape[0], verts_b.shape[0])
    return verts_a[:n], verts_b[:n], faces_a


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
    stride = max(1, int(frame_stride))
    qpos_frame = _parse_coordinate_frame(data["qpos_coordinate_frame"]) if "qpos_coordinate_frame" in data else None
    qpos_a = np.asarray(data["qpos_A"], dtype=np.float32)[::stride] if "qpos_A" in data else None
    qpos_b = np.asarray(data["qpos_B"], dtype=np.float32)[::stride] if "qpos_B" in data else None
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


def _resolve_robot_urdf_paths(cfg: Config, qpos_npz_path: Path, has_b: bool) -> tuple[Path, Path | None]:
    data = np.load(str(qpos_npz_path), allow_pickle=True)
    robot_type_a = _infer_robot_type(data, "robot_A")
    robot_type_b = _infer_robot_type(data, "robot_B")

    urdf_a = cfg.robot_urdf_a or cfg.robot_urdf
    if urdf_a is None:
        urdf_a = DEFAULT_URDF_BY_ROBOT.get(robot_type_a or "", DEFAULT_URDF_BY_ROBOT["g1"])

    urdf_b_path: Path | None = None
    if has_b:
        urdf_b = cfg.robot_urdf_b or cfg.robot_urdf
        if urdf_b is None:
            urdf_b = DEFAULT_URDF_BY_ROBOT.get(robot_type_b or "", urdf_a)
        urdf_b_path = Path(urdf_b)

    return Path(urdf_a), urdf_b_path


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
        q_new = _quat_mul_wxyz(Q_YUP_TO_ZUP, q_old)
        out[i, 3:7] = _normalize_quat_wxyz(q_new)
    return out


def _update_mesh(
    server: viser.ViserServer,
    handle: object | None,
    path: str,
    vertices: np.ndarray,
    faces: np.ndarray,
    color: tuple[int, int, int],
    opacity: float,
) -> object:
    if handle is not None:
        try:
            setattr(handle, "vertices", vertices)
            setattr(handle, "faces", faces)
            if hasattr(handle, "opacity"):
                setattr(handle, "opacity", float(opacity))
            return handle
        except Exception:
            try:
                handle.remove()
            except Exception:
                pass
    return server.scene.add_mesh_simple(
        path,
        vertices=vertices,
        faces=faces,
        color=color,
        opacity=float(opacity),
    )


def main(cfg: Config) -> None:
    npz_path = Path(cfg.data_npz)
    if not npz_path.exists():
        raise FileNotFoundError(f"data_npz does not exist: {npz_path}")

    sequence_id_ref, fps_ref, target_frames = _load_reference_timeline(npz_path, cfg.frame_stride)
    sequence_id = cfg.sequence_id if cfg.sequence_id is not None else sequence_id_ref
    fps = float(cfg.fps_override) if cfg.fps_override is not None else float(fps_ref)
    fps = max(1e-6, fps)

    direct_mesh = _load_direct_mesh_from_npz(npz_path=npz_path, frame_stride=cfg.frame_stride)
    if direct_mesh is not None:
        verts_a, verts_b, faces = direct_mesh
        source_mode = "direct_npz_mesh"
    else:
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
            verts_a = _resample_frames(verts_a, target_frames)
            verts_b = _resample_frames(verts_b, target_frames)
        else:
            stride = max(1, int(cfg.frame_stride))
            verts_a = verts_a[::stride]
            verts_b = verts_b[::stride]
        source_mode = "smplx_reconstructed"

    if cfg.y_up_to_z_up:
        verts_a = _convert_y_up_to_z_up_points(verts_a)
        verts_b = _convert_y_up_to_z_up_points(verts_b)

    n_frames = min(verts_a.shape[0], verts_b.shape[0])
    verts_a = verts_a[:n_frames]
    verts_b = verts_b[:n_frames]

    print(
        f"[dual_interx_mesh_renderer] sequence={sequence_id} frames={n_frames} "
        f"verts={verts_a.shape[1]} faces={faces.shape[0]} fps={fps:.2f} source={source_mode}"
    )
    print(f"[dual_interx_mesh_renderer] y_up_to_z_up={cfg.y_up_to_z_up}")

    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    qpos_a, qpos_b, qpos_coordinate_frame = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    if cfg.show_robots:
        if qpos_a is None or qpos_b is None:
            raise ValueError(
                f"show_robots=True but qpos_A/qpos_B were not found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a dual retarget output."
            )
        should_convert_qpos = bool(cfg.y_up_to_z_up)
        if qpos_coordinate_frame == "z_up":
            should_convert_qpos = False
        if should_convert_qpos:
            qpos_a = _convert_y_up_to_z_up_qpos(qpos_a)
            qpos_b = _convert_y_up_to_z_up_qpos(qpos_b)
        print(
            "[dual_interx_mesh_renderer] qpos_coordinate_frame="
            f"{qpos_coordinate_frame or 'legacy_assumed_y_up'} | qpos_y_up_to_z_up_applied={should_convert_qpos}"
        )

    server = viser.ViserServer()
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))

    mesh_a = server.scene.add_mesh_simple(
        "/humans/A/mesh",
        vertices=verts_a[0],
        faces=faces,
        color=(0, 200, 255),
        opacity=float(cfg.mesh_opacity),
    )
    mesh_b = server.scene.add_mesh_simple(
        "/humans/B/mesh",
        vertices=verts_b[0],
        faces=faces,
        color=(255, 160, 0),
        opacity=float(cfg.mesh_opacity),
    )

    robot_a = None
    robot_b = None
    robot_a_root = None
    robot_b_root = None
    robot_dof_a = None
    robot_dof_b = None
    if cfg.show_robots:
        urdf_a_path, urdf_b_path = _resolve_robot_urdf_paths(cfg, qpos_npz_path, has_b=True)
        robot_urdf_a = yourdfpy.URDF.load(str(urdf_a_path), load_meshes=True, build_scene_graph=True)

        robot_a_root = server.scene.add_frame("/robots/A", show_axes=False)
        robot_a = ViserUrdf(server, urdf_or_path=robot_urdf_a, root_node_name="/robots/A")
        robot_a.show_visual = True
        robot_dof_a = len(robot_a.get_actuated_joint_limits())

        if urdf_b_path is None:
            raise ValueError("Could not resolve robot_urdf for B.")
        robot_urdf_b = yourdfpy.URDF.load(str(urdf_b_path), load_meshes=True, build_scene_graph=True)
        robot_b_root = server.scene.add_frame("/robots/B", show_axes=False)
        robot_b = ViserUrdf(server, urdf_or_path=robot_urdf_b, root_node_name="/robots/B")
        robot_b.show_visual = True
        robot_dof_b = len(robot_b.get_actuated_joint_limits())

        print(
            f"[dual_interx_mesh_renderer] robot overlay enabled | urdf_A={urdf_a_path} | dof_A={robot_dof_a}"
            f" | urdf_B={urdf_b_path} | dof_B={robot_dof_b}"
        )

        n_frames = min(n_frames, qpos_a.shape[0], qpos_b.shape[0])  # type: ignore[union-attr]
        verts_a = verts_a[:n_frames]
        verts_b = verts_b[:n_frames]

    with server.gui.add_folder("Playback"):
        playing_cb = server.gui.add_checkbox("Playing", initial_value=True)
        frame_slider = server.gui.add_slider("Frame", min=0, max=max(0, n_frames - 1), step=1, initial_value=0)
        fps_in = server.gui.add_number("FPS", initial_value=float(fps), min=1.0, max=240.0, step=1.0)

    with server.gui.add_folder("Display"):
        opacity_slider = server.gui.add_slider(
            "Mesh opacity",
            min=0.05,
            max=1.0,
            step=0.01,
            initial_value=float(cfg.mesh_opacity),
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
        nonlocal mesh_a, mesh_b
        i = int(np.clip(idx, 0, n_frames - 1))
        opacity = float(opacity_slider.value)
        mesh_a = _update_mesh(
            server=server,
            handle=mesh_a,
            path="/humans/A/mesh",
            vertices=verts_a[i],
            faces=faces,
            color=(0, 200, 255),
            opacity=opacity,
        )
        mesh_b = _update_mesh(
            server=server,
            handle=mesh_b,
            path="/humans/B/mesh",
            vertices=verts_b[i],
            faces=faces,
            color=(255, 160, 0),
            opacity=opacity,
        )

        if cfg.show_robots and qpos_a is not None and robot_a is not None and robot_a_root is not None and robot_dof_a is not None:
            qa = qpos_a[i]
            if qa.shape[0] < 7 + robot_dof_a:
                raise ValueError(f"qpos_A frame has width {qa.shape[0]}, expected at least {7 + robot_dof_a}")
            robot_a_root.position = qa[0:3]
            robot_a_root.wxyz = qa[3:7]
            robot_a.update_cfg(qa[7 : 7 + robot_dof_a])
        if cfg.show_robots and qpos_b is not None and robot_b is not None and robot_b_root is not None and robot_dof_b is not None:
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

    print("[dual_interx_mesh_renderer] Open the Viser URL above. Ctrl+C to exit.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main(tyro.cli(Config))
