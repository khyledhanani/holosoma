#!/usr/bin/env python3
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import smplx  # type: ignore[import-untyped]
import torch
import trimesh
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

HUMAN_PROXY_SPHERES: tuple[tuple[str, float], ...] = (
    ("Pelvis", 0.10),
    ("Head", 0.09),
    ("L_Wrist", 0.04),
    ("R_Wrist", 0.04),
    ("L_Foot", 0.05),
    ("R_Foot", 0.05),
)

HUMAN_PROXY_CAPSULES: tuple[tuple[str, str, str, float], ...] = (
    ("pelvis_spine1", "Pelvis", "Spine1", 0.08),
    ("spine1_spine2", "Spine1", "Spine2", 0.07),
    ("spine2_spine3", "Spine2", "Spine3", 0.07),
    ("spine3_neck", "Spine3", "Neck", 0.06),
    ("neck_head", "Neck", "Head", 0.06),
    ("l_collar_shoulder", "L_Collar", "L_Shoulder", 0.045),
    ("r_collar_shoulder", "R_Collar", "R_Shoulder", 0.045),
    ("l_shoulder_elbow", "L_Shoulder", "L_Elbow", 0.045),
    ("r_shoulder_elbow", "R_Shoulder", "R_Elbow", 0.045),
    ("l_elbow_wrist", "L_Elbow", "L_Wrist", 0.040),
    ("r_elbow_wrist", "R_Elbow", "R_Wrist", 0.040),
    ("l_hip_knee", "L_Hip", "L_Knee", 0.055),
    ("r_hip_knee", "R_Hip", "R_Knee", 0.055),
    ("l_knee_ankle", "L_Knee", "L_Ankle", 0.050),
    ("r_knee_ankle", "R_Knee", "R_Ankle", 0.050),
    ("l_ankle_foot", "L_Ankle", "L_Foot", 0.040),
    ("r_ankle_foot", "R_Ankle", "R_Foot", 0.040),
    ("l_palm_wrist_index", "L_Wrist", "L_hand_index_mcp", 0.020),
    ("l_palm_wrist_middle", "L_Wrist", "L_hand_middle_mcp", 0.020),
    ("l_palm_wrist_pinky", "L_Wrist", "L_hand_pinky_mcp", 0.018),
    ("l_index_finger", "L_hand_index_mcp", "L_hand_index_tip", 0.014),
    ("l_thumb_finger", "L_Wrist", "L_hand_thumb_tip", 0.014),
    ("r_palm_wrist_index", "R_Wrist", "R_hand_index_mcp", 0.020),
    ("r_palm_wrist_middle", "R_Wrist", "R_hand_middle_mcp", 0.020),
    ("r_palm_wrist_pinky", "R_Wrist", "R_hand_pinky_mcp", 0.018),
    ("r_index_finger", "R_hand_index_mcp", "R_hand_index_tip", 0.014),
    ("r_thumb_finger", "R_Wrist", "R_hand_thumb_tip", 0.014),
)

HUMAN_PROXY_HAND_SPHERES: tuple[tuple[str, float], ...] = (
    ("L_hand_index_mcp", 0.028),
    ("L_hand_middle_mcp", 0.028),
    ("L_hand_pinky_mcp", 0.026),
    ("L_hand_index_tip", 0.022),
    ("L_hand_thumb_tip", 0.022),
    ("R_hand_index_mcp", 0.028),
    ("R_hand_middle_mcp", 0.028),
    ("R_hand_pinky_mcp", 0.026),
    ("R_hand_index_tip", 0.022),
    ("R_hand_thumb_tip", 0.022),
)


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

    port: int = 8080
    """Port for the Viser server."""

    mesh_opacity: float = 0.9
    """Mesh opacity in [0, 1]."""

    show_human_proxy: bool = False
    """Render the MuJoCo human collision proxy overlay when human joints are available."""

    human_proxy_opacity: float = 1.0
    """Opacity for the MuJoCo human proxy overlay."""

    hide_human_mesh_on_robot_side: bool = True
    """In mixed mode, hide the SMPL-X mesh on the robot-retargeted side."""

    loop: bool = True
    """Loop playback."""

    y_up_to_z_up: bool = True
    """Convert incoming points from y-up to z-up before rendering."""

    align_mesh_to_proxy: bool = True
    """Align reconstructed mesh to proxy joints via per-frame translation for visual overlay."""

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


def _load_proxy_joints_from_npz(
    npz_path: Path,
    frame_stride: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, list[str] | None, list[str] | None]:
    data = np.load(str(npz_path), allow_pickle=True)
    stride = max(1, int(frame_stride))
    proxy_frame = _parse_coordinate_frame(data["qpos_coordinate_frame"]) if "qpos_coordinate_frame" in data else None
    joints_a_key = "human_joints_A_rich" if "human_joints_A_rich" in data else "human_joints_A"
    joints_b_key = "human_joints_B_rich" if "human_joints_B_rich" in data else "human_joints_B"
    joints_a = np.asarray(data[joints_a_key], dtype=np.float32)[::stride] if joints_a_key in data else None
    joints_b = np.asarray(data[joints_b_key], dtype=np.float32)[::stride] if joints_b_key in data else None
    names_a = [str(x) for x in np.asarray(data["human_joint_names_A"]).tolist()] if "human_joint_names_A" in data else None
    names_b = [str(x) for x in np.asarray(data["human_joint_names_B"]).tolist()] if "human_joint_names_B" in data else None
    if joints_a is not None and joints_b is not None:
        n = min(joints_a.shape[0], joints_b.shape[0])
        joints_a = joints_a[:n]
        joints_b = joints_b[:n]
    return joints_a, joints_b, proxy_frame, names_a, names_b


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


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)


def _quat_wxyz_from_segment(p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    z_axis = _normalize(np.asarray(p1, dtype=np.float32) - np.asarray(p0, dtype=np.float32))
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(z_axis, up))) > 0.95:
        up = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    x_axis = _normalize(np.cross(up, z_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rot = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float32, copy=False)
    quat_xyzw = trimesh.transformations.quaternion_from_matrix(
        np.vstack([np.column_stack([rot, np.zeros(3)]), np.array([0.0, 0.0, 0.0, 1.0])])
    )
    return np.asarray([quat_xyzw[0], quat_xyzw[1], quat_xyzw[2], quat_xyzw[3]], dtype=np.float32)


def _transform_mesh(mesh: trimesh.Trimesh, *, quat_wxyz: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transform = trimesh.transformations.quaternion_matrix(np.asarray(quat_wxyz, dtype=float))
    transform[:3, 3] = np.asarray(translation, dtype=float)
    verts = trimesh.transform_points(mesh.vertices, transform).astype(np.float32, copy=False)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    return verts, faces


def _build_proxy_mesh_frame(joints: np.ndarray, joint_names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    if joint_names is None:
        names = SMPLX22_JOINT_NAMES[: min(len(SMPLX22_JOINT_NAMES), joints.shape[0])]
    else:
        names = joint_names[: joints.shape[0]]
    joint_map = {name: np.asarray(joints[idx], dtype=np.float32) for idx, name in enumerate(names)}
    vertices_parts: list[np.ndarray] = []
    faces_parts: list[np.ndarray] = []
    face_offset = 0

    for joint_name, radius in HUMAN_PROXY_SPHERES + HUMAN_PROXY_HAND_SPHERES:
        pos = joint_map.get(joint_name)
        if pos is None:
            continue
        sphere = trimesh.creation.icosphere(subdivisions=2, radius=float(radius))
        verts = np.asarray(sphere.vertices, dtype=np.float32) + pos.reshape(1, 3)
        faces = np.asarray(sphere.faces, dtype=np.uint32) + face_offset
        vertices_parts.append(verts)
        faces_parts.append(faces)
        face_offset += verts.shape[0]

    for _, joint_a, joint_b, radius in HUMAN_PROXY_CAPSULES:
        p0 = joint_map.get(joint_a)
        p1 = joint_map.get(joint_b)
        if p0 is None or p1 is None:
            continue
        segment = np.asarray(p1 - p0, dtype=np.float32)
        height = max(1e-3, float(np.linalg.norm(segment)))
        capsule = trimesh.creation.capsule(radius=float(radius), height=height, count=[10, 16])
        quat_wxyz = _quat_wxyz_from_segment(p0, p1)
        midpoint = 0.5 * (p0 + p1)
        verts, faces = _transform_mesh(capsule, quat_wxyz=quat_wxyz, translation=midpoint)
        faces = faces + face_offset
        vertices_parts.append(verts)
        faces_parts.append(faces)
        face_offset += verts.shape[0]

    if not vertices_parts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)
    return np.vstack(vertices_parts), np.vstack(faces_parts)


def _parse_coordinate_frame(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    frame = str(value).strip().lower()
    if not frame:
        return None
    if frame not in {"y_up", "z_up"}:
        raise ValueError(f"Unsupported coordinate frame metadata: {frame}")
    return frame


def _proxy_floor_height(
    proxy_joints: np.ndarray,
    joint_names: list[str] | None,
    vertical_axis: int,
) -> np.ndarray:
    if joint_names is not None:
        name_to_idx = {str(name): i for i, name in enumerate(joint_names[: proxy_joints.shape[1]])}
        foot_idx = [name_to_idx[n] for n in ("L_Foot", "R_Foot", "L_Ankle", "R_Ankle") if n in name_to_idx]
    else:
        foot_idx = []

    if foot_idx:
        return np.min(proxy_joints[:, foot_idx, vertical_axis], axis=1)
    # Fallback: use global minimum joint height if foot names are unavailable.
    return np.min(proxy_joints[:, :, vertical_axis], axis=1)


def _infer_proxy_vertical_axis(
    proxy_joints_a: np.ndarray | None,
    proxy_joints_b: np.ndarray | None,
) -> int:
    candidates: list[np.ndarray] = []
    if proxy_joints_a is not None and proxy_joints_a.size > 0:
        candidates.append(proxy_joints_a.reshape(-1, 3))
    if proxy_joints_b is not None and proxy_joints_b.size > 0:
        candidates.append(proxy_joints_b.reshape(-1, 3))
    if not candidates:
        return 2

    points = np.concatenate(candidates, axis=0)
    # Vertical axis should usually have larger span than lateral depth for
    # standing/walking sequences; this works well for our human tracks.
    span_y = float(np.percentile(points[:, 1], 95.0) - np.percentile(points[:, 1], 5.0))
    span_z = float(np.percentile(points[:, 2], 95.0) - np.percentile(points[:, 2], 5.0))
    return 2 if span_z >= span_y else 1


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


def _coerce_scalar_string(value: object) -> str | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    text = str(value).strip()
    return text or None


def _coerce_scalar_float(value: object) -> float | None:
    if isinstance(value, np.ndarray):
        value = value.item()
    try:
        return float(value)
    except Exception:
        return None


def _resolve_mesh_scales(data_npz_path: Path, qpos_npz_path: Path) -> tuple[float, float]:
    sources: list[np.lib.npyio.NpzFile] = []
    seen: set[Path] = set()
    for path in (qpos_npz_path, data_npz_path):
        resolved = path.resolve()
        if resolved in seen or not path.exists():
            continue
        seen.add(resolved)
        sources.append(np.load(str(path), allow_pickle=True))

    def _read_scale(key: str) -> float | None:
        for data in sources:
            if key not in data:
                continue
            value = _coerce_scalar_float(data[key])
            if value is not None:
                return value
        return None

    return _read_scale("scale_A") or 1.0, _read_scale("scale_B") or 1.0


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
        urdf_a_path = Path(urdf_a)

    urdf_b_path: Path | None = None
    if include_b:
        urdf_b = cfg.robot_urdf_b or cfg.robot_urdf
        if urdf_b is None:
            fallback = str(urdf_a_path) if urdf_a_path is not None else DEFAULT_URDF_BY_ROBOT["g1"]
            urdf_b = DEFAULT_URDF_BY_ROBOT.get(robot_type_b or "", fallback)
        urdf_b_path = Path(urdf_b)

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
    proxy_joints_a, proxy_joints_b, proxy_coordinate_frame, proxy_names_a, proxy_names_b = _load_proxy_joints_from_npz(npz_path, cfg.frame_stride)

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
                verts_a = _resample_frames(verts_a, target_frames)
                verts_b = _resample_frames(verts_b, target_frames)
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

    mesh_scale_a, mesh_scale_b = _resolve_mesh_scales(npz_path, Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path)
    if mesh_available:
        verts_a = verts_a * float(mesh_scale_a)
        verts_b = verts_b * float(mesh_scale_b)

    n_frames = min(verts_a.shape[0], verts_b.shape[0])
    verts_a = verts_a[:n_frames]
    verts_b = verts_b[:n_frames]

    print(
        f"[dual_interx_mesh_renderer] sequence={sequence_id} frames={n_frames} "
        f"verts={verts_a.shape[1]} faces={faces.shape[0]} fps={fps:.2f} source={source_mode}"
    )
    print(f"[dual_interx_mesh_renderer] y_up_to_z_up={cfg.y_up_to_z_up}")
    print(f"[dual_interx_mesh_renderer] mesh_scale_A={mesh_scale_a:.4f} | mesh_scale_B={mesh_scale_b:.4f}")
    if mesh_warning is not None:
        print(f"[dual_interx_mesh_renderer] mesh_fallback={mesh_warning}")

    qpos_npz_path = Path(cfg.qpos_npz) if cfg.qpos_npz is not None else npz_path
    qpos_a, qpos_b, qpos_coordinate_frame = _load_qpos_tracks(qpos_npz_path, cfg.frame_stride)
    proxy_vertical_axis = _infer_proxy_vertical_axis(proxy_joints_a, proxy_joints_b)
    proxy_frame_effective = proxy_coordinate_frame
    if proxy_frame_effective is None and qpos_coordinate_frame in {"y_up", "z_up"}:
        proxy_frame_effective = qpos_coordinate_frame
    if proxy_frame_effective is None:
        proxy_frame_effective = "z_up" if proxy_vertical_axis == 2 else "y_up"
    target_a, target_b = _resolve_target_types(npz_path, qpos_npz_path)
    show_robot_a = bool(cfg.show_robots and target_a != "human")
    show_robot_b = bool(cfg.show_robots and target_b != "human")
    show_proxy_a = bool(cfg.show_human_proxy and proxy_joints_a is not None and target_a == "human")
    show_proxy_b = bool(cfg.show_human_proxy and proxy_joints_b is not None and target_b == "human")
    hide_human_mesh_a = bool(cfg.hide_human_mesh_on_robot_side and target_a == "robot" and target_b == "human")
    hide_human_mesh_b = bool(cfg.hide_human_mesh_on_robot_side and target_b == "robot" and target_a == "human")
    print(f"[dual_interx_mesh_renderer] target_A={target_a} | target_B={target_b}")
    if cfg.show_human_proxy:
        should_convert_proxy = bool(cfg.y_up_to_z_up)
        if proxy_frame_effective == "z_up":
            should_convert_proxy = False
        if should_convert_proxy:
            if proxy_joints_a is not None:
                proxy_joints_a = _convert_y_up_to_z_up_points(proxy_joints_a)
            if proxy_joints_b is not None:
                proxy_joints_b = _convert_y_up_to_z_up_points(proxy_joints_b)
        print(
            "[dual_interx_mesh_renderer] human_proxy="
            f"{'enabled' if (show_proxy_a or show_proxy_b) else 'unavailable'}"
            f" | proxy_coordinate_frame={proxy_frame_effective}"
            f" | proxy_y_up_to_z_up_applied={should_convert_proxy}"
        )
    if cfg.align_mesh_to_proxy and mesh_available:
        vertical_axis = 2 if proxy_frame_effective == "z_up" else 1
        horiz_axes = (0, 1) if vertical_axis == 2 else (0, 2)
        if proxy_joints_a is not None and verts_a.shape[0] == proxy_joints_a.shape[0] and verts_a.shape[1] > 0:
            proxy_centroid_a = np.mean(proxy_joints_a, axis=1)
            mesh_centroid_a = np.mean(verts_a, axis=1)
            delta_a = np.zeros_like(proxy_centroid_a)
            delta_a[:, horiz_axes[0]] = proxy_centroid_a[:, horiz_axes[0]] - mesh_centroid_a[:, horiz_axes[0]]
            delta_a[:, horiz_axes[1]] = proxy_centroid_a[:, horiz_axes[1]] - mesh_centroid_a[:, horiz_axes[1]]
            proxy_floor_a = _proxy_floor_height(proxy_joints_a, proxy_names_a, vertical_axis=vertical_axis)
            mesh_floor_a = np.min(verts_a[:, :, vertical_axis], axis=1)
            delta_a[:, vertical_axis] = proxy_floor_a - mesh_floor_a
            verts_a = verts_a + delta_a[:, None, :]
            print(
                "[dual_interx_mesh_renderer] align_A=feet_anchored"
                f" mean_offset={float(np.linalg.norm(np.mean(delta_a, axis=0))):.5f}"
            )
        if proxy_joints_b is not None and verts_b.shape[0] == proxy_joints_b.shape[0] and verts_b.shape[1] > 0:
            proxy_centroid_b = np.mean(proxy_joints_b, axis=1)
            mesh_centroid_b = np.mean(verts_b, axis=1)
            delta_b = np.zeros_like(proxy_centroid_b)
            delta_b[:, horiz_axes[0]] = proxy_centroid_b[:, horiz_axes[0]] - mesh_centroid_b[:, horiz_axes[0]]
            delta_b[:, horiz_axes[1]] = proxy_centroid_b[:, horiz_axes[1]] - mesh_centroid_b[:, horiz_axes[1]]
            proxy_floor_b = _proxy_floor_height(proxy_joints_b, proxy_names_b, vertical_axis=vertical_axis)
            mesh_floor_b = np.min(verts_b[:, :, vertical_axis], axis=1)
            delta_b[:, vertical_axis] = proxy_floor_b - mesh_floor_b
            verts_b = verts_b + delta_b[:, None, :]
            print(
                "[dual_interx_mesh_renderer] align_B=feet_anchored"
                f" mean_offset={float(np.linalg.norm(np.mean(delta_b, axis=0))):.5f}"
            )
    if cfg.show_robots:
        if show_robot_a and qpos_a is None:
            raise ValueError(
                f"show_robots=True but qpos_A was not found in {qpos_npz_path}. "
                "Provide --qpos-npz pointing to a retarget output with qpos_A."
            )
        if show_robot_b and qpos_b is None:
            raise ValueError(
                f"show_robots=True but qpos_B was not found in {qpos_npz_path}. "
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
            "[dual_interx_mesh_renderer] qpos_coordinate_frame="
            f"{qpos_coordinate_frame or 'legacy_assumed_y_up'} | qpos_y_up_to_z_up_applied={should_convert_qpos}"
        )

    server = viser.ViserServer(port=cfg.port)
    server.scene.add_grid("/grid", width=4.0, height=4.0, position=(0.0, 0.0, 0.0))

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

        robot_overlay_parts: list[str] = ["[dual_interx_mesh_renderer] robot overlay enabled"]
        if show_robot_a:
            robot_overlay_parts.append(f"urdf_A={urdf_a_path} | dof_A={robot_dof_a}")
        if show_robot_b:
            robot_overlay_parts.append(f"urdf_B={urdf_b_path} | dof_B={robot_dof_b}")
        print(" | ".join(robot_overlay_parts))

        if show_robot_a and qpos_a is not None:
            n_frames = min(n_frames, qpos_a.shape[0])
        if show_robot_b and qpos_b is not None:
            n_frames = min(n_frames, qpos_b.shape[0])
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
        show_human_meshes_cb = server.gui.add_checkbox("Show human meshes", initial_value=True)
        if show_proxy_a or show_proxy_b:
            proxy_opacity_slider = server.gui.add_slider(
                "Proxy opacity",
                min=0.05,
                max=1.0,
                step=0.01,
                initial_value=float(cfg.human_proxy_opacity),
            )
            show_proxy_meshes_cb = server.gui.add_checkbox("Show proxy meshes", initial_value=True)
        if show_robot_a or show_robot_b:
            show_robot_meshes_cb = server.gui.add_checkbox("Show robot meshes", initial_value=True)

            @show_robot_meshes_cb.on_update
            def _(_event) -> None:
                if robot_a is not None:
                    robot_a.show_visual = bool(show_robot_meshes_cb.value)
                if robot_b is not None:
                    robot_b.show_visual = bool(show_robot_meshes_cb.value)

    def _apply_frame(idx: int) -> None:
        nonlocal mesh_a, mesh_b, proxy_a_handle, proxy_b_handle
        i = int(np.clip(idx, 0, n_frames - 1))
        opacity = float(opacity_slider.value)
        if mesh_available and mesh_a is not None:
            mesh_a = _update_mesh(
                server=server,
                handle=mesh_a,
                path="/humans/A/mesh",
                vertices=verts_a[i],
                faces=faces,
                color=(0, 200, 255),
                opacity=opacity,
            )
            if hasattr(mesh_a, "visible"):
                mesh_a.visible = bool(show_human_meshes_cb.value) and (not hide_human_mesh_a)
        if mesh_available and mesh_b is not None:
            mesh_b = _update_mesh(
                server=server,
                handle=mesh_b,
                path="/humans/B/mesh",
                vertices=verts_b[i],
                faces=faces,
                color=(255, 160, 0),
                opacity=opacity,
            )
            if hasattr(mesh_b, "visible"):
                mesh_b.visible = bool(show_human_meshes_cb.value) and (not hide_human_mesh_b)
        if show_proxy_a and proxy_joints_a is not None:
            proxy_a_verts, proxy_a_faces = _build_proxy_mesh_frame(proxy_joints_a[i], joint_names=proxy_names_a)
            proxy_a_handle = _update_mesh(
                server=server,
                handle=proxy_a_handle,
                path="/humans/A/proxy",
                vertices=proxy_a_verts,
                faces=proxy_a_faces,
                color=(215, 85, 70),
                opacity=float(proxy_opacity_slider.value),
            )
            if hasattr(proxy_a_handle, "visible"):
                proxy_a_handle.visible = bool(show_proxy_meshes_cb.value)
        if show_proxy_b and proxy_joints_b is not None:
            proxy_b_verts, proxy_b_faces = _build_proxy_mesh_frame(proxy_joints_b[i], joint_names=proxy_names_b)
            proxy_b_handle = _update_mesh(
                server=server,
                handle=proxy_b_handle,
                path="/humans/B/proxy",
                vertices=proxy_b_verts,
                faces=proxy_b_faces,
                color=(215, 85, 70),
                opacity=float(proxy_opacity_slider.value),
            )
            if hasattr(proxy_b_handle, "visible"):
                proxy_b_handle.visible = bool(show_proxy_meshes_cb.value)

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
