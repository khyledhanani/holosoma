from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial import Delaunay  # type: ignore[import-untyped]
from scipy.spatial.transform import Rotation  # type: ignore[import-untyped]

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
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


@dataclass(frozen=True)
class _BodyPointTarget:
    body_name: str
    point_offset: np.ndarray


@dataclass(frozen=True)
class _ContactLinearization:
    phi: float
    j_dqa: np.ndarray
    geom1_name: str
    geom2_name: str
    owner1: str
    owner2: str


@dataclass(frozen=True)
class _ProxySphere:
    joint_name: str
    radius: float


@dataclass(frozen=True)
class _ProxyCapsule:
    name: str
    joint_a: str
    joint_b: str
    radius: float


HUMAN_PROXY_SPHERES = (
    # Slightly fuller body radii to better match SMPL-X mesh occupancy.
    _ProxySphere("Pelvis", 0.12),
    _ProxySphere("Head", 0.10),
    _ProxySphere("L_Wrist", 0.05),
    _ProxySphere("R_Wrist", 0.05),
    _ProxySphere("L_Foot", 0.06),
    _ProxySphere("R_Foot", 0.06),
)

HUMAN_PROXY_CAPSULES = (
    _ProxyCapsule("pelvis_spine1", "Pelvis", "Spine1", 0.10),
    _ProxyCapsule("spine1_spine2", "Spine1", "Spine2", 0.09),
    _ProxyCapsule("spine2_spine3", "Spine2", "Spine3", 0.09),
    _ProxyCapsule("spine3_neck", "Spine3", "Neck", 0.08),
    _ProxyCapsule("neck_head", "Neck", "Head", 0.07),
    _ProxyCapsule("pelvis_l_hip", "Pelvis", "L_Hip", 0.08),
    _ProxyCapsule("pelvis_r_hip", "Pelvis", "R_Hip", 0.08),
    _ProxyCapsule("hip_span", "L_Hip", "R_Hip", 0.08),
    _ProxyCapsule("collar_span", "L_Collar", "R_Collar", 0.07),
    _ProxyCapsule("shoulder_span", "L_Shoulder", "R_Shoulder", 0.07),
    _ProxyCapsule("l_collar_shoulder", "L_Collar", "L_Shoulder", 0.055),
    _ProxyCapsule("r_collar_shoulder", "R_Collar", "R_Shoulder", 0.055),
    _ProxyCapsule("l_shoulder_elbow", "L_Shoulder", "L_Elbow", 0.055),
    _ProxyCapsule("r_shoulder_elbow", "R_Shoulder", "R_Elbow", 0.055),
    _ProxyCapsule("l_elbow_wrist", "L_Elbow", "L_Wrist", 0.050),
    _ProxyCapsule("r_elbow_wrist", "R_Elbow", "R_Wrist", 0.050),
    _ProxyCapsule("l_hip_knee", "L_Hip", "L_Knee", 0.065),
    _ProxyCapsule("r_hip_knee", "R_Hip", "R_Knee", 0.065),
    _ProxyCapsule("l_knee_ankle", "L_Knee", "L_Ankle", 0.058),
    _ProxyCapsule("r_knee_ankle", "R_Knee", "R_Ankle", 0.058),
    _ProxyCapsule("l_ankle_foot", "L_Ankle", "L_Foot", 0.050),
    _ProxyCapsule("r_ankle_foot", "R_Ankle", "R_Foot", 0.050),
)

HUMAN_PROXY_HAND_SPHERES = (
    _ProxySphere("L_hand_index_mcp", 0.032),
    _ProxySphere("L_hand_middle_mcp", 0.032),
    _ProxySphere("L_hand_pinky_mcp", 0.030),
    _ProxySphere("L_hand_index_tip", 0.024),
    _ProxySphere("L_hand_thumb_tip", 0.024),
    _ProxySphere("R_hand_index_mcp", 0.032),
    _ProxySphere("R_hand_middle_mcp", 0.032),
    _ProxySphere("R_hand_pinky_mcp", 0.030),
    _ProxySphere("R_hand_index_tip", 0.024),
    _ProxySphere("R_hand_thumb_tip", 0.024),
)

HUMAN_PROXY_HAND_CAPSULES = (
    _ProxyCapsule("l_palm_wrist_index", "L_Wrist", "L_hand_index_mcp", 0.028),
    _ProxyCapsule("l_palm_wrist_middle", "L_Wrist", "L_hand_middle_mcp", 0.028),
    _ProxyCapsule("l_palm_wrist_pinky", "L_Wrist", "L_hand_pinky_mcp", 0.026),
    _ProxyCapsule("l_palm_index_middle", "L_hand_index_mcp", "L_hand_middle_mcp", 0.024),
    _ProxyCapsule("l_palm_middle_pinky", "L_hand_middle_mcp", "L_hand_pinky_mcp", 0.023),
    _ProxyCapsule("l_palm_index_pinky", "L_hand_index_mcp", "L_hand_pinky_mcp", 0.021),
    _ProxyCapsule("l_index_finger", "L_hand_index_mcp", "L_hand_index_tip", 0.017),
    _ProxyCapsule("l_thumb_finger", "L_Wrist", "L_hand_thumb_tip", 0.017),
    _ProxyCapsule("r_palm_wrist_index", "R_Wrist", "R_hand_index_mcp", 0.028),
    _ProxyCapsule("r_palm_wrist_middle", "R_Wrist", "R_hand_middle_mcp", 0.028),
    _ProxyCapsule("r_palm_wrist_pinky", "R_Wrist", "R_hand_pinky_mcp", 0.026),
    _ProxyCapsule("r_palm_index_middle", "R_hand_index_mcp", "R_hand_middle_mcp", 0.024),
    _ProxyCapsule("r_palm_middle_pinky", "R_hand_middle_mcp", "R_hand_pinky_mcp", 0.023),
    _ProxyCapsule("r_palm_index_pinky", "R_hand_index_mcp", "R_hand_pinky_mcp", 0.021),
    _ProxyCapsule("r_index_finger", "R_hand_index_mcp", "R_hand_index_tip", 0.017),
    _ProxyCapsule("r_thumb_finger", "R_Wrist", "R_hand_thumb_tip", 0.017),
)

ALL_HUMAN_PROXY_SPHERES = HUMAN_PROXY_SPHERES + HUMAN_PROXY_HAND_SPHERES
ALL_HUMAN_PROXY_CAPSULES = HUMAN_PROXY_CAPSULES + HUMAN_PROXY_HAND_CAPSULES


def _joint_name_to_index(joint_names: list[str]) -> dict[str, int]:
    return {name: i for i, name in enumerate(joint_names)}


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


def _mean_segment_length(
    joints: np.ndarray,
    idx_a: int,
    idx_b: int,
) -> float:
    seg = np.asarray(joints[:, idx_b] - joints[:, idx_a], dtype=float)
    lengths = np.linalg.norm(seg, axis=1)
    return max(1e-3, float(np.mean(lengths)))


def _append_proxy_body(
    worldbody: ET.Element,
    *,
    body_name: str,
    geom_name: str,
    geom_type: str,
    pos: np.ndarray,
    quat: np.ndarray,
    size: tuple[float, ...],
    rgba: tuple[float, float, float, float] = (0.9, 0.4, 0.3, 0.25),
) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        attrib={
            "name": body_name,
            "mocap": "true",
            "pos": f"{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}",
            "quat": f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
        },
    )
    ET.SubElement(
        body,
        "geom",
        attrib={
            "name": geom_name,
            "type": geom_type,
            "size": " ".join(f"{x:.6f}" for x in size),
            "contype": "1",
            "conaffinity": "1",
            "density": "0",
            "group": "4",
            "rgba": " ".join(f"{x:.6f}" for x in rgba),
        },
    )


def _resolve_mesh_base_dir(robot_xml_path: Path, root: ET.Element) -> Path:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir") if compiler is not None else None
    if meshdir:
        mesh_base = Path(meshdir)
        if not mesh_base.is_absolute():
            mesh_base = (robot_xml_path.parent / mesh_base).resolve()
        return mesh_base
    assets_dir = robot_xml_path.parent / "assets"
    meshes_dir = robot_xml_path.parent / "meshes"
    if assets_dir.exists():
        return assets_dir.resolve()
    if meshes_dir.exists():
        return meshes_dir.resolve()
    return robot_xml_path.parent.resolve()


def _absolutize_file_paths(root: ET.Element, robot_xml_path: Path) -> None:
    mesh_base = _resolve_mesh_base_dir(robot_xml_path, root)
    for node in root.iter():
        file_attr = node.attrib.get("file")
        if file_attr is None:
            continue
        file_path = Path(file_attr)
        if file_path.is_absolute():
            continue
        if node.tag in {"mesh", "texture", "hfield", "heightfield"}:
            abs_path = (mesh_base / file_path).resolve()
        else:
            abs_path = (robot_xml_path.parent / file_path).resolve()
        node.set("file", str(abs_path))


def build_human_proxy_scene_xml(
    *,
    robot_xml_path: Path,
    out_path: Path,
    human_joint_trajectory: np.ndarray,
    joint_names: list[str],
    human_prefix: str = "H_",
) -> Path:
    tree = ET.parse(str(robot_xml_path))
    root = tree.getroot()
    _absolutize_file_paths(root, robot_xml_path)
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{robot_xml_path} has no worldbody section.")

    joint_to_idx = _joint_name_to_index(joint_names)
    initial = np.asarray(human_joint_trajectory[0], dtype=float)

    for sphere in ALL_HUMAN_PROXY_SPHERES:
        if sphere.joint_name not in joint_to_idx:
            continue
        idx = joint_to_idx[sphere.joint_name]
        pos = initial[idx]
        _append_proxy_body(
            worldbody,
            body_name=_sphere_body_name(human_prefix, sphere.joint_name),
            geom_name=_sphere_body_name(human_prefix, sphere.joint_name),
            geom_type="sphere",
            pos=pos,
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            size=(sphere.radius,),
        )

    for capsule in ALL_HUMAN_PROXY_CAPSULES:
        if capsule.joint_a not in joint_to_idx or capsule.joint_b not in joint_to_idx:
            continue
        idx_a = joint_to_idx[capsule.joint_a]
        idx_b = joint_to_idx[capsule.joint_b]
        p0 = initial[idx_a]
        p1 = initial[idx_b]
        midpoint = 0.5 * (p0 + p1)
        quat = _quat_wxyz_from_segment(p0, p1)
        half_length = 0.5 * _mean_segment_length(human_joint_trajectory, idx_a, idx_b)
        _append_proxy_body(
            worldbody,
            body_name=_capsule_body_name(human_prefix, capsule.name),
            geom_name=_capsule_body_name(human_prefix, capsule.name),
            geom_type="capsule",
            pos=midpoint,
            quat=quat,
            size=(capsule.radius, half_length),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(str(out_path), encoding="utf-8", xml_declaration=False)
    return out_path


class MixedHumanRobotRetargeter:
    """Mixed robot-human SQP retargeter with a kinematic MuJoCo human proxy.

    The robot is optimized. The human remains fixed in the objective and is
    inserted into MuJoCo as mocap-driven collision geoms.
    """

    def __init__(
        self,
        *,
        task_constants: ModuleType,
        mixed_scene_xml_path: str,
        human_joint_names: list[str] | None = None,
        human_prefix: str = "H_",
        q_a_init_idx: int = -7,
        activate_foot_sticking: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.08,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        human_contact_slack_weight: float = 300.0,
        allow_human_contact_slack: bool = True,
        human_clearance_margin: float = 0.0,
        collision_polish_passes: int = 0,
        collision_polish_step_size: float = 0.03,
        collision_polish_slack_weight: float = 5e4,
        collision_polish_target_margin: float = 0.0,
        w_self: float = 2.0,
        w_inter: float = 10.0,
        w_reg: float = 0.1,
        w_pelvis_abs: float = 4.0,
        w_nominal_tracking_init: float = 0.0,
        nominal_tracking_tau: float = 1e6,
        max_source_edge_len: float = 0.45,
        cross_agent_rescue_k: int = 3,
        debug: bool = False,
    ):
        self.task_constants = task_constants
        self.debug = bool(debug)

        self.human_prefix = human_prefix
        self.collision_detection_threshold = float(collision_detection_threshold)
        self.activate_foot_sticking = bool(activate_foot_sticking)
        self.activate_joint_limits = bool(activate_joint_limits)
        self.penetration_tolerance = float(penetration_tolerance)
        self.foot_sticking_tolerance = float(foot_sticking_tolerance)
        self.human_contact_slack_weight = float(human_contact_slack_weight)
        self.allow_human_contact_slack = bool(allow_human_contact_slack)
        self.human_clearance_margin = float(human_clearance_margin)
        self.step_size = float(step_size)
        self.collision_polish_passes = int(collision_polish_passes)
        self.collision_polish_step_size = float(collision_polish_step_size)
        self.collision_polish_slack_weight = float(collision_polish_slack_weight)
        self.collision_polish_target_margin = float(collision_polish_target_margin)

        self.w_self = float(w_self)
        self.w_inter = float(w_inter)
        self.w_reg = float(w_reg)
        self.w_pelvis_abs = float(w_pelvis_abs)
        self.w_nominal_tracking_init = float(w_nominal_tracking_init)
        self.nominal_tracking_tau = float(nominal_tracking_tau)
        self.max_source_edge_len = float(max_source_edge_len)
        self.cross_agent_rescue_k = int(cross_agent_rescue_k)

        self.robot_model = mujoco.MjModel.from_xml_path(mixed_scene_xml_path)
        self.robot_data = mujoco.MjData(self.robot_model)

        self.demo_joints = list(self.task_constants.DEMO_JOINTS)
        if len(self.demo_joints) < 22:
            raise ValueError(f"Expected at least 22 demo joints, got {len(self.demo_joints)}")
        self.source_joint_names_a = self.demo_joints[:22]
        # Use all provided human proxy joints (e.g., richer hand landmarks) instead
        # of truncating to the base 22-body set.
        self.source_joint_names_b = list(human_joint_names if human_joint_names is not None else SMPLX22_JOINTS)
        self.num_source_a = len(self.source_joint_names_a)
        self.num_source_b = len(self.source_joint_names_b)
        self.num_source_total = self.num_source_a + self.num_source_b

        self.laplacian_match_links = dict(self.task_constants.JOINTS_MAPPING)
        self.base_foot_links = list(self.task_constants.FOOT_STICKING_LINKS)
        self.robot_dof_expected = int(self.task_constants.ROBOT_DOF)

        self.q_a_init_idx = int(q_a_init_idx)
        self._setup_joint_layout()
        self._setup_mapped_links()
        self._setup_bounds_and_costs()
        self._setup_source_graph_indices()
        self._setup_human_proxy_handles()

    def _setup_joint_layout(self) -> None:
        free_joint_ids = [
            j for j in range(self.robot_model.njnt)
            if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if not free_joint_ids:
            raise ValueError("MixedHumanRobotRetargeter expects one free joint for the robot root.")

        self.free_joint_id = int(sorted(free_joint_ids, key=lambda j: int(self.robot_model.jnt_qposadr[j]))[0])
        self.qadr = int(self.robot_model.jnt_qposadr[self.free_joint_id])

        qadr_all_free = sorted(int(self.robot_model.jnt_qposadr[j]) for j in free_joint_ids)
        qend = next((q for q in qadr_all_free if q > self.qadr), int(self.robot_model.nq))
        self.q_block = np.arange(self.qadr, qend, dtype=int)

        if len(self.q_block) < 7:
            raise ValueError("Robot qpos block must contain the free joint state.")
        self.robot_dof = len(self.q_block) - 7
        if self.robot_dof != self.robot_dof_expected:
            raise ValueError(
                f"Robot DOF mismatch: constants={self.robot_dof_expected}, scene={self.robot_dof}"
            )

        self.base_quat_slice = slice(self.qadr + 3, self.qadr + 7)
        self.start_local = 7 + self.q_a_init_idx
        if self.start_local < 0 or self.start_local >= len(self.q_block):
            raise ValueError("q_a_init_idx must be >= -7.")
        self.q_a_indices = self.q_block[self.start_local:]
        self.nq_a = len(self.q_a_indices)

        self._natural_standing_z = self._compute_natural_standing_z()

    def _compute_natural_standing_z(self) -> float:
        q_default = self.robot_model.qpos0.copy()
        self.robot_data.qpos[:] = q_default
        mujoco.mj_forward(self.robot_model, self.robot_data)

        root_z = float(self.robot_data.qpos[self.qadr + 2])
        min_geom_z = np.inf
        for g in range(self.robot_model.ngeom):
            gname = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gname.startswith(self.human_prefix):
                continue
            if "ground" in gname.lower() or "floor" in gname.lower():
                continue
            min_geom_z = min(min_geom_z, float(self.robot_data.geom_xpos[g, 2]))
        if np.isinf(min_geom_z):
            return root_z
        return float(root_z - min_geom_z)

    def _setup_mapped_links(self) -> None:
        self.laplacian_match_links_prefixed = {
            joint_name: self._resolve_body_point_target(link_target)
            for joint_name, link_target in self.laplacian_match_links.items()
        }
        self.foot_links = {name: name for name in self.base_foot_links}

    @staticmethod
    def _resolve_body_point_target(
        target: str | tuple[str, np.ndarray] | _BodyPointTarget,
    ) -> str | _BodyPointTarget:
        if isinstance(target, _BodyPointTarget):
            return target
        if isinstance(target, tuple):
            body_name, point_offset = target
            return _BodyPointTarget(
                body_name=body_name,
                point_offset=np.asarray(point_offset, dtype=float).reshape(3),
            )
        return target

    def _setup_bounds_and_costs(self) -> None:
        nq = self.robot_model.nq
        large = 1e6
        lower = -large * np.ones(nq, dtype=float)
        upper = large * np.ones(nq, dtype=float)
        for j in range(self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                qadr = int(self.robot_model.jnt_qposadr[j])
                lower[qadr] = float(self.robot_model.jnt_range[j, 0])
                upper[qadr] = float(self.robot_model.jnt_range[j, 1])
        self.q_a_lb = lower[self.q_a_indices].copy()
        self.q_a_ub = upper[self.q_a_indices].copy()

        for k, v in getattr(self.task_constants, "MANUAL_LB", {}).items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.q_a_lb[idx] = float(v)
        for k, v in getattr(self.task_constants, "MANUAL_UB", {}).items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.q_a_ub[idx] = float(v)

        self.Q_diag = np.zeros(self.nq_a, dtype=float)
        for k, v in getattr(self.task_constants, "MANUAL_COST", {}).items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.Q_diag[idx] = float(v)

        self.track_nominal_indices = np.arange(self.nq_a, dtype=int)

    def _setup_source_graph_indices(self) -> None:
        self.source_name_to_idx_a = {n: i for i, n in enumerate(self.source_joint_names_a)}
        self.source_name_to_idx_b = {n: i for i, n in enumerate(self.source_joint_names_b)}

        cross_critical = {
            "L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
        }
        self.cross_rescue_a = [
            self.source_name_to_idx_a[n] for n in cross_critical if n in self.source_name_to_idx_a
        ]
        self.cross_rescue_b_local = [
            self.source_name_to_idx_b[n] for n in cross_critical if n in self.source_name_to_idx_b
        ]

        critical_w = {
            "Pelvis", "Spine1", "Spine2", "Spine3", "Neck",
            "L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
        }
        self.laplacian_weights = np.ones(self.num_source_total, dtype=float)
        for i, name in enumerate(self.source_joint_names_a):
            if name in critical_w:
                self.laplacian_weights[i] = 2.5
        for i, name in enumerate(self.source_joint_names_b):
            if name in critical_w:
                self.laplacian_weights[self.num_source_a + i] = 2.5

    def _setup_human_proxy_handles(self) -> None:
        self._human_joint_body_mocap: dict[str, int] = {}
        self._human_capsule_body_mocap: dict[str, int] = {}
        for sphere in ALL_HUMAN_PROXY_SPHERES:
            body_name = _sphere_body_name(self.human_prefix, sphere.joint_name)
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                mocap_id = int(self.robot_model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self._human_joint_body_mocap[sphere.joint_name] = mocap_id
        for capsule in ALL_HUMAN_PROXY_CAPSULES:
            body_name = _capsule_body_name(self.human_prefix, capsule.name)
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id != -1:
                mocap_id = int(self.robot_model.body_mocapid[body_id])
                if mocap_id >= 0:
                    self._human_capsule_body_mocap[capsule.name] = mocap_id

    def _set_human_proxy_pose(self, human_joints: np.ndarray) -> None:
        joint_to_idx = self.source_name_to_idx_b
        for sphere in ALL_HUMAN_PROXY_SPHERES:
            mocap_id = self._human_joint_body_mocap.get(sphere.joint_name)
            idx = joint_to_idx.get(sphere.joint_name)
            if mocap_id is None or idx is None:
                continue
            self.robot_data.mocap_pos[mocap_id] = np.asarray(human_joints[idx], dtype=float)
            self.robot_data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

        for capsule in ALL_HUMAN_PROXY_CAPSULES:
            mocap_id = self._human_capsule_body_mocap.get(capsule.name)
            idx_a = joint_to_idx.get(capsule.joint_a)
            idx_b = joint_to_idx.get(capsule.joint_b)
            if mocap_id is None or idx_a is None or idx_b is None:
                continue
            p0 = np.asarray(human_joints[idx_a], dtype=float)
            p1 = np.asarray(human_joints[idx_b], dtype=float)
            self.robot_data.mocap_pos[mocap_id] = 0.5 * (p0 + p1)
            self.robot_data.mocap_quat[mocap_id] = _quat_wxyz_from_segment(p0, p1)

    def retarget_motion(
        self,
        *,
        robot_human_joints_raw: np.ndarray,
        human_joints_fixed: np.ndarray,
        robot_height_source: float,
        human_scale_fixed: float = 1.0,
        foot_sticking_sequences: list[dict[str, bool]] | None = None,
        q_init_a: np.ndarray | None = None,
        q_nominal_a: np.ndarray | None = None,
        dest_res_path: str | None = None,
        n_iter_initial: int = 50,
        n_iter_per_frame: int = 10,
    ) -> np.ndarray:
        if robot_human_joints_raw.shape[0] != human_joints_fixed.shape[0]:
            raise ValueError("Robot-source and human-fixed sequences must have the same frame count.")
        if robot_human_joints_raw.ndim != 3 or human_joints_fixed.ndim != 3:
            raise ValueError("Expected joint arrays with shape (T, J, 3).")

        num_frames = robot_human_joints_raw.shape[0]
        robot_height = float(getattr(self.task_constants, "ROBOT_HEIGHT", 1.78))
        scale_a = robot_height / float(robot_height_source)

        if foot_sticking_sequences is None:
            foot_sticking_sequences = [{} for _ in range(num_frames)]
        if len(foot_sticking_sequences) != num_frames:
            raise ValueError("Foot-sticking sequence length must match number of frames.")

        q = self.robot_model.qpos0.copy()
        if q_init_a is None:
            pelvis = (scale_a * robot_human_joints_raw[0, 0, :]).astype(float)
            q_local = np.zeros(7 + self.robot_dof, dtype=float)
            q_local[:2] = pelvis[:2]
            q_local[2] = self._natural_standing_z
            q_local[3] = 1.0
            self._write_local_robot_state(q, q_local)
        else:
            self._write_local_robot_state(q, q_init_a)

        q_prev = q.copy()
        qpos_list: list[np.ndarray] = []

        if q_nominal_a is not None:
            q_nominal_arr = np.asarray(q_nominal_a, dtype=float)
            if q_nominal_arr.shape[0] != num_frames:
                raise ValueError("Nominal sequence must have shape (T, D).")
        else:
            q_nominal_arr = None

        for t in range(num_frames):
            raw_a_t = np.asarray(robot_human_joints_raw[t, : self.num_source_a], dtype=float)
            raw_b_t = np.asarray(human_joints_fixed[t, : self.num_source_b], dtype=float)
            source_vertices = np.vstack([scale_a * raw_a_t, raw_b_t])
            self_adj, _, inter_edges = self.build_source_graph(source_vertices)
            spring_data = self.compute_inter_spring_data(source_vertices, inter_edges)

            nominal_t = None
            if q_nominal_arr is not None:
                nominal_local = q_nominal_arr[t]
                nominal_t = np.asarray(nominal_local[self.start_local : 7 + self.robot_dof], dtype=float)

            q, _ = self.iterate(
                q_n=q,
                q_t_last=q_prev,
                source_vertices=source_vertices,
                self_adj_list=self_adj,
                spring_data=spring_data,
                human_joints_fixed_t=raw_b_t,
                foot_sticking=foot_sticking_sequences[t],
                q_nominal_a=nominal_t,
                n_iter=int(n_iter_initial) if t == 0 else int(n_iter_per_frame),
                init_t=(t == 0),
            )
            if self.collision_polish_passes > 0:
                q = self._polish_human_collision_frame(
                    q=q,
                    human_joints_fixed_t=raw_b_t,
                    n_passes=self.collision_polish_passes,
                )
            q_prev = q.copy()
            qpos_list.append(self._extract_local_robot_state(q))

        qpos = np.asarray(qpos_list, dtype=np.float32)
        if dest_res_path is not None:
            np.savez(
                dest_res_path,
                qpos_A=qpos,
                human_joints_A=robot_human_joints_raw,
                human_joints_B=human_joints_fixed,
                scale_A=np.asarray(scale_a, dtype=np.float32),
                scale_B=np.asarray(human_scale_fixed, dtype=np.float32),
                target_A=np.asarray("robot"),
                target_B=np.asarray("human"),
            )
        return qpos

    def iterate(
        self,
        *,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        source_vertices: np.ndarray,
        self_adj_list: list[list[int]],
        spring_data: list[tuple[int, int, float, np.ndarray]],
        human_joints_fixed_t: np.ndarray,
        foot_sticking: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        n_iter: int,
        init_t: bool,
    ) -> tuple[np.ndarray, float]:
        last_cost = np.inf
        q_curr = q_n.copy()
        for _ in range(n_iter):
            q_curr, cost = self.solve_single_iteration(
                q_curr=q_curr,
                q_t_last=q_t_last,
                source_vertices=source_vertices,
                self_adj_list=self_adj_list,
                spring_data=spring_data,
                human_joints_fixed_t=human_joints_fixed_t,
                foot_sticking=foot_sticking,
                q_nominal_a=q_nominal_a,
                init_t=init_t,
            )
            if np.isclose(cost, last_cost):
                break
            last_cost = float(cost)
        return q_curr, float(last_cost)

    def solve_single_iteration(
        self,
        *,
        q_curr: np.ndarray,
        q_t_last: np.ndarray,
        source_vertices: np.ndarray,
        self_adj_list: list[list[int]],
        spring_data: list[tuple[int, int, float, np.ndarray]],
        human_joints_fixed_t: np.ndarray,
        foot_sticking: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        init_t: bool = False,
        verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        q_a_n = q_curr[self.q_a_indices]
        J_target, target_vertices = self._build_target_jacobian_and_vertices(
            q_curr=q_curr,
            source_vertices=source_vertices,
        )

        L_self = calculate_laplacian_matrix(target_vertices, self_adj_list)
        if not sp.issparse(L_self):
            L_self = sp.csr_matrix(L_self)
        L_self = L_self.tocsr()
        kron_self = sp.kron(L_self, sp.eye(3, format="csr"), format="csr")
        J_lap = kron_self @ J_target
        lap0 = (L_self @ target_vertices).reshape(-1)
        target_lap_vec = calculate_laplacian_coordinates(source_vertices, self_adj_list).reshape(-1)

        dqa = cp.Variable(self.nq_a, name="dqa")
        lap_slack = cp.Variable(3 * source_vertices.shape[0], name="lap_slack")
        constraints: list[cp.Constraint] = []
        constraints.append(cp.Constant(J_lap) @ dqa - lap_slack == -lap0)

        if spring_data:
            M_spring = np.zeros((3 * len(spring_data), self.nq_a), dtype=float)
            b_spring = np.zeros(3 * len(spring_data), dtype=float)
            for k, (vi, vj, omega_ij, r_hat_ij) in enumerate(spring_data):
                J_rel = J_target[3 * vi : 3 * (vi + 1), :] - J_target[3 * vj : 3 * (vj + 1), :]
                r_curr = target_vertices[vi] - target_vertices[vj]
                sqrt_w = float(np.sqrt(omega_ij))
                M_spring[3 * k : 3 * (k + 1), :] = sqrt_w * J_rel
                b_spring[3 * k : 3 * (k + 1)] = sqrt_w * (r_hat_ij - r_curr)
        else:
            M_spring = np.zeros((0, self.nq_a), dtype=float)
            b_spring = np.zeros(0, dtype=float)

        if self.activate_foot_sticking:
            constraints.extend(
                self._build_foot_constraints(
                    q_curr=q_curr,
                    q_t_last=q_t_last,
                    dqa=dqa,
                    flags=foot_sticking,
                )
            )

        contact_rows = self._linearize_contact_rows(q_curr, human_joints_fixed_t)
        constraints.extend(self._build_ground_constraints(contact_rows, dqa))
        human_constraints, human_slack = self._build_human_constraints(contact_rows, dqa)
        constraints.extend(human_constraints)

        if self.activate_joint_limits:
            constraints += [dqa >= (self.q_a_lb - q_a_n), dqa <= (self.q_a_ub - q_a_n)]

        constraints.append(cp.SOC(self.step_size, dqa))

        obj_terms: list[cp.Expression] = []
        sqrt_w_lap = np.sqrt(np.repeat(self.laplacian_weights, 3))
        obj_terms.append(
            self.w_self * cp.sum_squares(cp.multiply(sqrt_w_lap, lap_slack - target_lap_vec))
        )
        if spring_data:
            obj_terms.append(self.w_inter * cp.sum_squares(cp.Constant(M_spring) @ dqa - b_spring))

        dx_smooth = q_t_last[self.q_a_indices] - q_a_n
        obj_terms.append(self.w_reg * cp.sum_squares(dqa - dx_smooth))

        if q_nominal_a is not None:
            z = dqa - (q_nominal_a - q_a_n)
            obj_terms.append(self.w_nominal_tracking_init * cp.sum_squares(z))

        if self.w_pelvis_abs > 0:
            j_pelvis_z = J_target[2, :]
            target_pelvis_z = float(source_vertices[0, 2])
            curr_pelvis_z = float(target_vertices[0, 2])
            obj_terms.append(
                self.w_pelvis_abs * (cp.Constant(j_pelvis_z) @ dqa - (target_pelvis_z - curr_pelvis_z)) ** 2
            )

        if np.any(self.Q_diag > 0):
            obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(self.Q_diag), dqa + q_a_n)))

        if human_slack is not None and human_slack.size > 0:
            obj_terms.append(self.human_contact_slack_weight * cp.sum_squares(human_slack))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        self._solve_with_fallback(problem, verbose=verbose)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and init_t:
            constraints_no_soc = [
                c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)
            ]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints_no_soc)
            self._solve_with_fallback(problem, verbose=verbose)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"MixedHumanRobotRetargeter solve failed: {problem.status}")

        q_next = q_curr.copy()
        q_next[self.q_a_indices] = q_a_n + np.asarray(dqa.value, dtype=float)
        q_next[self.base_quat_slice] /= np.linalg.norm(q_next[self.base_quat_slice]) + 1e-12
        return q_next, float(problem.value)

    @staticmethod
    def _solve_with_fallback(problem: cp.Problem, verbose: bool = False) -> None:
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return
        except Exception:
            pass
        problem.solve(solver=cp.CLARABEL, verbose=verbose)

    def _build_target_jacobian_and_vertices(
        self,
        *,
        q_curr: np.ndarray,
        source_vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        j_dict, p_dict = self._calc_manipulator_jacobians(q_curr, self.laplacian_match_links_prefixed)
        n_v = source_vertices.shape[0]
        target_vertices = source_vertices.copy()
        J_target = np.zeros((3 * n_v, self.nq_a), dtype=float)

        for i, name in enumerate(self.source_joint_names_a):
            if name not in p_dict:
                continue
            row = slice(3 * i, 3 * (i + 1))
            target_vertices[i] = p_dict[name]
            J_target[row, :] = np.asarray(j_dict[name][:, self.q_a_indices], dtype=float)
        return J_target, target_vertices

    def _build_foot_constraints(
        self,
        *,
        q_curr: np.ndarray,
        q_t_last: np.ndarray,
        dqa: cp.Expression,
        flags: dict[str, bool],
    ) -> list[cp.Constraint]:
        if self.q_a_init_idx >= 12:
            return []
        j_now, p_now = self._calc_manipulator_jacobians(q_curr, self.foot_links)
        _, p_prev = self._calc_manipulator_jacobians(q_t_last, self.foot_links)
        left_flag, right_flag = self._extract_lr_flags(flags)
        constraints: list[cp.Constraint] = []
        for link_name, j_full in j_now.items():
            lnk = link_name.lower()
            apply = ("left" in lnk and left_flag) or ("right" in lnk and right_flag)
            if not apply:
                continue
            p_lb = p_prev[link_name] - p_now[link_name] - self.foot_sticking_tolerance
            p_ub = p_prev[link_name] - p_now[link_name] + self.foot_sticking_tolerance
            jxy = j_full[:2, self.q_a_indices]
            constraints += [jxy @ dqa >= p_lb[:2], jxy @ dqa <= p_ub[:2]]
        return constraints

    @staticmethod
    def _extract_lr_flags(flags: dict[str, bool]) -> tuple[bool, bool]:
        left = right = False
        for k, v in flags.items():
            lk = k.lower()
            if lk.startswith("l") or "left" in lk:
                left = bool(v)
            if lk.startswith("r") or "right" in lk:
                right = bool(v)
        return left, right

    def _build_ground_constraints(
        self,
        rows: list[_ContactLinearization],
        dqa: cp.Expression,
    ) -> list[cp.Constraint]:
        return [
            row.j_dqa @ dqa >= (-row.phi - self.penetration_tolerance)
            for row in rows
            if self._is_ground_pair(row.owner1, row.owner2)
        ]

    def _build_human_constraints(
        self,
        rows: list[_ContactLinearization],
        dqa: cp.Expression,
    ) -> tuple[list[cp.Constraint], cp.Variable | None]:
        human_rows = [row for row in rows if self._is_human_pair(row.owner1, row.owner2)]
        if not human_rows:
            return [], None
        if not self.allow_human_contact_slack:
            constraints = [
                row.j_dqa @ dqa >= (self.human_clearance_margin - row.phi - self.penetration_tolerance)
                for row in human_rows
            ]
            return constraints, None
        slack = cp.Variable(len(human_rows), nonneg=True, name="human_contact_slack")
        constraints: list[cp.Constraint] = []
        for i, row in enumerate(human_rows):
            constraints.append(
                row.j_dqa @ dqa
                >= (self.human_clearance_margin - row.phi - self.penetration_tolerance - slack[i])
            )
        return constraints, slack

    def _linearize_contact_rows(
        self,
        q: np.ndarray,
        human_joints_fixed_t: np.ndarray,
    ) -> list[_ContactLinearization]:
        self.robot_data.qpos[:] = q
        self._set_human_proxy_pose(human_joints_fixed_t)
        mujoco.mj_forward(self.robot_model, self.robot_data)

        candidates = self._prefilter_pairs_with_mj_collision(self.collision_detection_threshold)
        geom_names = [
            mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for g in range(self.robot_model.ngeom)
        ]
        rows: list[_ContactLinearization] = []
        fromto = np.zeros(6, dtype=float)
        for g1, g2 in candidates:
            name1 = geom_names[g1]
            name2 = geom_names[g2]
            owner1 = self._owner_from_geom(g1, name1)
            owner2 = self._owner_from_geom(g2, name2)
            if not (self._is_ground_pair(owner1, owner2) or self._is_human_pair(owner1, owner2)):
                continue
            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(
                self.robot_model,
                self.robot_data,
                int(g1),
                int(g2),
                float(self.collision_detection_threshold),
                fromto,
            )
            if dist > self.collision_detection_threshold:
                continue
            j_full = self._compute_jacobian_for_contact_relative(
                self.robot_model.geom(g1),
                self.robot_model.geom(g2),
                name1,
                name2,
                fromto,
                dist,
            )
            rows.append(
                _ContactLinearization(
                    phi=float(dist),
                    j_dqa=np.asarray(j_full[self.q_a_indices], dtype=float),
                    geom1_name=name1,
                    geom2_name=name2,
                    owner1=owner1,
                    owner2=owner2,
                )
            )
        return rows

    def _polish_human_collision_frame(
        self,
        *,
        q: np.ndarray,
        human_joints_fixed_t: np.ndarray,
        n_passes: int,
    ) -> np.ndarray:
        q_curr = q.copy()
        for _ in range(max(0, int(n_passes))):
            rows = self._linearize_contact_rows(q_curr, human_joints_fixed_t)
            human_rows = [row for row in rows if self._is_human_pair(row.owner1, row.owner2)]
            if not human_rows:
                break

            min_phi = min(float(row.phi) for row in human_rows)
            if min_phi >= self.collision_polish_target_margin:
                break

            q_a_n = q_curr[self.q_a_indices]
            dqa = cp.Variable(self.nq_a, name="dqa_polish")
            slack = cp.Variable(len(human_rows), nonneg=True, name="human_polish_slack")

            constraints: list[cp.Constraint] = []
            for i, row in enumerate(human_rows):
                constraints.append(
                    row.j_dqa @ dqa
                    >= (self.collision_polish_target_margin - row.phi - slack[i])
                )

            if self.activate_joint_limits:
                constraints += [dqa >= (self.q_a_lb - q_a_n), dqa <= (self.q_a_ub - q_a_n)]

            constraints.append(cp.SOC(self.collision_polish_step_size, dqa))

            objective = cp.sum_squares(dqa) + self.collision_polish_slack_weight * cp.sum_squares(slack)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            self._solve_with_fallback(problem, verbose=False)
            if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or dqa.value is None:
                break

            q_curr[self.q_a_indices] = q_a_n + np.asarray(dqa.value, dtype=float)
            q_curr[self.base_quat_slice] /= np.linalg.norm(q_curr[self.base_quat_slice]) + 1e-12
        return q_curr

    def _prefilter_pairs_with_mj_collision(self, threshold: float) -> set[tuple[int, int]]:
        m, d = self.robot_model, self.robot_data
        if not hasattr(self, "_saved_margins"):
            self._saved_margins = np.empty_like(m.geom_margin)
        self._saved_margins[:] = m.geom_margin
        m.geom_margin[:] = threshold
        mujoco.mj_collision(m, d)

        candidates: set[tuple[int, int]] = set()
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 < 0 or g2 < 0 or g1 == g2:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        m.geom_margin[:] = self._saved_margins
        return candidates

    def _owner_from_prefixed_name(self, name: str) -> str:
        lower = name.lower()
        if "ground" in lower or "floor" in lower:
            return "G"
        if name.startswith(self.human_prefix):
            return "H"
        return "A"

    def _owner_from_geom(self, geom_id: int, geom_name: str) -> str:
        owner = self._owner_from_prefixed_name(geom_name)
        if owner != "A":
            return owner

        body_id = int(self.robot_model.geom_bodyid[int(geom_id)])
        body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        owner = self._owner_from_prefixed_name(body_name)
        if owner != "A":
            return owner

        if int(self.robot_model.geom_type[int(geom_id)]) == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.robot_model.geom_dataid[int(geom_id)])
            if mesh_id >= 0:
                mesh_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
                owner = self._owner_from_prefixed_name(mesh_name)
                if owner != "A":
                    return owner

        return "A"

    @staticmethod
    def _is_ground_pair(o1: str, o2: str) -> bool:
        return (o1 == "A" and o2 == "G") or (o1 == "G" and o2 == "A")

    @staticmethod
    def _is_human_pair(o1: str, o2: str) -> bool:
        return (o1 == "A" and o2 == "H") or (o1 == "H" and o2 == "A")

    def _compute_jacobian_for_contact_relative(
        self,
        geom1,
        geom2,
        geom1_name: str,
        geom2_name: str,
        fromto: np.ndarray,
        dist: float,
    ) -> np.ndarray:
        pos1 = fromto[:3]
        pos2 = fromto[3:]
        v = pos1 - pos2
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            n_hat = np.sign(dist) * (v / norm_v)
        elif "ground" in geom2_name.lower():
            n_hat = np.array([0.0, 0.0, 1.0]) * (1.0 if dist >= 0 else -1.0)
        elif "ground" in geom1_name.lower():
            n_hat = np.array([0.0, 0.0, -1.0]) * (1.0 if dist >= 0 else -1.0)
        else:
            n_hat = np.zeros(3)
        j_a = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        j_b = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)
        return n_hat @ (j_a - j_b)

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str | _BodyPointTarget],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        J_dict: dict[str, np.ndarray] = {}
        p_dict: dict[str, np.ndarray] = {}
        for key, target in links.items():
            if isinstance(target, _BodyPointTarget):
                body_name = target.body_name
                point_offset = target.point_offset
            else:
                body_name = target
                point_offset = np.zeros(3, dtype=float)

            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                continue
            J = self._calc_contact_jacobian_from_point(body_id, point_offset)
            R_WB = self.robot_data.xmat[body_id].reshape(3, 3)
            pos = self.robot_data.xpos[body_id] + R_WB @ point_offset
            J_dict[key] = np.asarray(J, dtype=float, copy=True)
            p_dict[key] = np.asarray(pos, dtype=float, copy=True)
        return J_dict, p_dict

    def _calc_contact_jacobian_from_point(
        self,
        body_idx: int,
        p_body: np.ndarray,
        input_world: bool = False,
    ) -> np.ndarray:
        p_body = np.asarray(p_body, dtype=float).reshape(3)
        mujoco.mj_forward(self.robot_model, self.robot_data)
        R_WB = self.robot_data.xmat[body_idx].reshape(3, 3)
        p_WB = self.robot_data.xpos[body_idx]
        if input_world:
            p_W = p_body.astype(np.float64).reshape(3, 1)
        else:
            p_W = (p_WB + R_WB @ p_body).astype(np.float64).reshape(3, 1)
        Jp = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        Jr = np.zeros((3, self.robot_model.nv), dtype=np.float64, order="C")
        mujoco.mj_jac(self.robot_model, self.robot_data, Jp, Jr, p_W, int(body_idx))
        T = self._build_transform_qdot_to_qvel()
        return Jp @ T

    def _build_transform_qdot_to_qvel(self) -> np.ndarray:
        nq = self.robot_model.nq
        nv = self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        def _e_world(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
            return np.array([
                [-qx, qw, qz, -qy],
                [-qy, -qz, qw, qx],
                [-qz, qy, -qx, qw],
            ])

        for j in range(self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            qadr = int(self.robot_model.jnt_qposadr[j])
            dadr = int(self.robot_model.jnt_dofadr[j])
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)
                qw, qx, qy, qz = self.robot_data.qpos[qadr + 3 : qadr + 7]
                T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = 2.0 * _e_world(qw, qx, qy, qz)
            elif jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                T[dadr, qadr] = 1.0
        return T

    def _write_local_robot_state(self, q_full: np.ndarray, q_local: np.ndarray) -> None:
        min_width = 7 + self.robot_dof
        if q_local.shape[0] < min_width:
            raise ValueError(f"Expected local state length >= {min_width}, got {q_local.shape[0]}")
        q_full[self.q_block[:min_width]] = q_local[:min_width]

    def _extract_local_robot_state(self, q_full: np.ndarray) -> np.ndarray:
        return np.asarray(q_full[self.q_block[: 7 + self.robot_dof]], dtype=np.float32)

    def build_source_graph(
        self,
        source_vertices: np.ndarray,
    ) -> tuple[list[list[int]], list[list[int]], set[tuple[int, int]]]:
        all_edges = self._delaunay_edges(source_vertices)
        all_edges = self._prune_long_edges(source_vertices, all_edges, self.max_source_edge_len)
        all_edges = self._add_cross_agent_rescue_edges(source_vertices, all_edges)
        self_edges, inter_edges = self._partition_edges(all_edges)
        n = source_vertices.shape[0]
        return self._edges_to_adj_list(self_edges, n), self._edges_to_adj_list(all_edges, n), inter_edges

    def _partition_edges(
        self,
        edges: set[tuple[int, int]],
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        E_self: set[tuple[int, int]] = set()
        E_inter: set[tuple[int, int]] = set()
        for i, j in edges:
            if (i < self.num_source_a) == (j < self.num_source_a):
                E_self.add((i, j))
            else:
                E_inter.add((i, j))
        return E_self, E_inter

    def _add_cross_agent_rescue_edges(
        self,
        vertices: np.ndarray,
        edges: set[tuple[int, int]],
    ) -> set[tuple[int, int]]:
        a_ids = np.arange(0, self.num_source_a, dtype=int)
        b_ids = np.arange(self.num_source_a, self.num_source_total, dtype=int)
        dmat = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)

        def _cross_count(idx: int) -> int:
            count = 0
            for ei, ej in edges:
                if ei == idx and ej >= self.num_source_a:
                    count += 1
                elif ej == idx and ei >= self.num_source_a:
                    count += 1
            return count

        for idx in self.cross_rescue_a:
            if _cross_count(idx) == 0 and len(b_ids) > 0:
                nbrs = b_ids[np.argsort(dmat[idx, b_ids])[: self.cross_agent_rescue_k]]
                for j in nbrs:
                    jj = int(j)
                    edges.add((idx, jj) if idx < jj else (jj, idx))

        for idx_local in self.cross_rescue_b_local:
            idx = self.num_source_a + idx_local
            if _cross_count(idx) == 0 and len(a_ids) > 0:
                nbrs = a_ids[np.argsort(dmat[idx, a_ids])[: self.cross_agent_rescue_k]]
                for j in nbrs:
                    jj = int(j)
                    edges.add((jj, idx) if jj < idx else (idx, jj))
        return edges

    def compute_inter_spring_data(
        self,
        source_vertices: np.ndarray,
        inter_edges: set[tuple[int, int]],
    ) -> list[tuple[int, int, float, np.ndarray]]:
        spring_data: list[tuple[int, int, float, np.ndarray]] = []
        for i, j in inter_edges:
            d_ij = float(np.linalg.norm(source_vertices[i] - source_vertices[j]))
            omega_ij = np.exp(-5.0 * d_ij)
            r_hat_ij = (source_vertices[i] - source_vertices[j]).astype(float)
            spring_data.append((i, j, omega_ij, r_hat_ij))
        return spring_data

    def _delaunay_edges(self, vertices: np.ndarray) -> set[tuple[int, int]]:
        try:
            simplices = Delaunay(vertices).simplices
        except Exception:
            n = vertices.shape[0]
            if n < 4:
                raise
            d = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            simplices = np.array([[i] + list(np.argsort(d[i])[:3]) for i in range(n)])
        return self._tetrahedra_to_edges(simplices)

    @staticmethod
    def _tetrahedra_to_edges(simplices: np.ndarray) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for tet in simplices:
            a, b, c, d = (int(x) for x in tet)
            for p, q in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                edges.add((p, q) if p < q else (q, p))
        return edges

    @staticmethod
    def _prune_long_edges(
        vertices: np.ndarray,
        edges: set[tuple[int, int]],
        max_len: float,
    ) -> set[tuple[int, int]]:
        return {
            (i, j)
            for i, j in edges
            if np.linalg.norm(vertices[i] - vertices[j]) <= max_len
        }

    @staticmethod
    def _edges_to_adj_list(
        edges: set[tuple[int, int]],
        n_vertices: int,
    ) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(n_vertices)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj
