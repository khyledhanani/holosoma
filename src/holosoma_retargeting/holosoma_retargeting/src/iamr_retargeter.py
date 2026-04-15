"""Interaction-Aware Motion Retargeting (IAMR).

Implements the IAMR module from:
    "Rhythm: Learning Interactive Whole-Body Control for Dual Humanoids"
    arXiv:2603.02856

IAMR resolves the kinematic conflict between individual motion fidelity and
interaction geometry preservation via a *dual reference manifold* formulation:

    Individual Manifold (ℳ_ind):  s^(k) = h_robot^(k) / h_raw^(k)  (per-agent scale)
    Unified Manifold   (ℳ_uni):  s_uni = mean(s^(A), s^(B))        (shared scale)

The SQP objective at each frame is:

    J(q_t) = w_self · J_self + w_inter · J_inter + w_reg · J_reg + λ_rot · J_rot

where:
    J_self  — Laplacian coordinates over **intra-agent** edges (E_self),
              targets computed from ℳ_ind.
    J_inter — Variable-stiffness spring potential over **inter-agent** edges
              (E_inter), targets from ℳ_uni.
              ω_ij(d_ij) = ω_max · exp(−γ · d_ij)
    J_reg   — Temporal-smoothness regularisation:  ||q_t − q_{t-1}||²
    J_rot   — Joint-angle tracking (rotation consistency):
              geodesic approximated via L2 distance to nominal configuration.

Hard constraints (same as DualInteractionMeshRetargeter):
    • Joint limits
    • Ground non-penetration
    • A–B collision avoidance (with optional whitelist slack)
    • Foot-sticking (XY)
    • Trust-region (SOC)

Outputs per retarget_motion call:
    qpos_A, qpos_B   — retargeted joint configurations (T, dof+7)
    interaction_graph — (T, N_a, N_b) bool — active E_inter connectivity
    contact_graph     — (T, N_a_links, N_b_links) bool — A–B physical contact

All saved to dest_res_path .npz when provided.

Paper hyper-parameters (Table in §IAMR):
    w_self = 2.0,  w_inter = 10.0,  w_reg = 0.1,  λ_rot = 0.1
    ω_max = 1.0  (per-edge max stiffness; w_inter handles global scaling)
    γ     = 5.0  (spring decay rate; ~0.6 at 0.1 m, ~0.08 at 0.5 m)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import cvxpy as cp  # type: ignore[import-not-found]
import mujoco  # type: ignore[import-not-found]
import numpy as np
from scipy import sparse as sp  # type: ignore[import-untyped]
from scipy.spatial import Delaunay  # type: ignore[import-untyped]
from tqdm import tqdm

# Add src to path for direct execution
_src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_src_path))

from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)

# ---------------------------------------------------------------------------
# SMPL-X body joint ordering (pose_body indices 0-20)
# Matches the standard AMASS/Inter-X ordering.
# ---------------------------------------------------------------------------

_SMPLX_BODY_JOINT_NAMES: tuple[str, ...] = (
    "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck",
    "L_Collar", "R_Collar", "Head", "L_Shoulder", "R_Shoulder",
    "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
)

# Maps SMPL-X joint name → list of robot hinge joint name keywords.
# A robot joint whose base name (prefix-stripped) starts with a keyword is
# matched to that SMPL-X joint for the J_rot rotation-tracking term.
_SMPLX_TO_JOINT_KEYWORDS: dict[str, list[str]] = {
    "L_Hip":       ["left_hip_pitch", "left_hip_roll", "left_hip_yaw"],
    "R_Hip":       ["right_hip_pitch", "right_hip_roll", "right_hip_yaw"],
    "Spine1":      ["waist"],   # lumbar waist joints → closest spine analogue
    "L_Knee":      ["left_knee"],
    "R_Knee":      ["right_knee"],
    "L_Ankle":     ["left_ankle_pitch", "left_ankle_roll"],
    "R_Ankle":     ["right_ankle_pitch", "right_ankle_roll"],
    "L_Shoulder":  ["left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw"],
    "R_Shoulder":  ["right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw"],
    "L_Elbow":     ["left_elbow"],
    "R_Elbow":     ["right_elbow"],
    "L_Wrist":     ["left_wrist"],
    "R_Wrist":     ["right_wrist"],
}


# ---------------------------------------------------------------------------
# Small data containers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _BodyPointTarget:
    """A point target attached to a MuJoCo body in local-body coordinates."""

    body_name: str
    point_offset: np.ndarray


@dataclass(frozen=True)
class _ContactLinearization:
    phi: float
    j_dx: np.ndarray
    geom1_name: str
    geom2_name: str
    owner1: str
    owner2: str
    whitelisted: bool


@dataclass(frozen=True)
class _FootSupportPoint:
    """Support sample attached to a foot contact body."""

    body_name: str
    side: str
    x_local: float
    clearance: float


# ---------------------------------------------------------------------------
# IAMR Retargeter
# ---------------------------------------------------------------------------

class IAMRRetargeter:
    """Interaction-Aware Motion Retargeting for dual humanoid robots.

    Differs from DualInteractionMeshRetargeter in three key ways:
    1. Dual reference manifolds (individual + unified scaling).
    2. Topological partitioning of the interaction graph into E_self / E_inter.
    3. Asymmetric SQP objective:
       - J_self  uses Laplacian coordinates with **intra-agent** edges and
                 targets from the individually-scaled manifold.
       - J_inter uses a variable-stiffness spring potential with **inter-agent**
                 edges and targets from the unified-scale manifold.
    """

    # Paper hyper-parameters (arXiv:2603.02856)
    W_SELF: float = 2.0
    W_INTER: float = 10.0
    W_REG: float = 0.1
    LAMBDA_ROT: float = 0.1
    OMEGA_MAX: float = 1.0
    GAMMA_DECAY: float = 5.0

    def __init__(
        self,
        task_constants_a: ModuleType,
        dual_scene_xml_path: str,
        *,
        task_constants_b: ModuleType | None = None,
        robot_a_prefix: str = "A_",
        robot_b_prefix: str = "B_",
        q_a_init_idx: int = -7,
        q_b_init_idx: int = -7,
        # Paper hyper-parameters (override defaults above if desired)
        w_self: float = 2.0,
        w_inter: float = 10.0,
        w_reg: float = 0.1,
        lambda_rot: float = 0.1,
        omega_max: float = 1.0,
        gamma_decay: float = 5.0,
        w_pelvis_abs: float = 1.0,
        # Source-foot grounding: hard constraint — if source foot z < threshold,
        # robot foot must touch ground (elastic slack penalised at contact_slack_weight).
        foot_ground_threshold: float = 0.05,  # metres; source foot z below this → grounded
        foot_ground_tolerance: float = 0.005,  # metres; allowable stance-foot clearance
        # Ground-node grounding (like single-robot version)
        n_ground_pts: int = 9,      # ground nodes per agent added to the Laplacian mesh
        ground_range: float = 0.3,  # half-side of ground grid in metres
        # Constraint settings
        activate_foot_sticking: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        n_iter_first: int = 40,
        n_iter_other: int = 10,
        collision_detection_threshold: float = 0.08,
        penetration_tolerance: float = 1e-3,
        foot_sticking_tolerance: float = 1e-3,
        whitelist_margin: float = -0.015,
        ab_top_k_pairs: int = 24,
        contact_slack_weight: float = 300.0,
        max_source_edge_len: float = 0.45,
        cross_agent_rescue_k: int = 3,
        ab_pair_whitelist: set[tuple[str, str]] | None = None,
        debug: bool = False,
    ):
        self.task_constants_a = task_constants_a
        self.task_constants_b = (
            task_constants_b if task_constants_b is not None else task_constants_a
        )
        self.debug = debug

        self.robot_a_prefix = robot_a_prefix
        self.robot_b_prefix = robot_b_prefix
        self.q_a_init_idx = q_a_init_idx
        self.q_b_init_idx = q_b_init_idx

        # IAMR objective weights (§3.2 of arXiv:2603.02856)
        self.w_self = float(w_self)
        self.w_inter = float(w_inter)
        self.w_reg = float(w_reg)
        self.lambda_rot = float(lambda_rot)
        self.omega_max = float(omega_max)
        self.gamma_decay = float(gamma_decay)
        self.w_pelvis_abs = float(w_pelvis_abs)
        self.foot_ground_threshold = float(foot_ground_threshold)
        self.foot_ground_tolerance = float(foot_ground_tolerance)
        _side = max(1, int(round(n_ground_pts ** 0.5)))
        self.n_ground = _side * _side   # actual ground nodes per agent
        self.ground_range = float(ground_range)

        # Constraint parameters
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_joint_limits = activate_joint_limits
        self.step_size = float(step_size)
        self.n_iter_first = int(n_iter_first)
        self.n_iter_other = int(n_iter_other)
        self.collision_detection_threshold = float(collision_detection_threshold)
        self.penetration_tolerance = float(penetration_tolerance)
        self.foot_sticking_tolerance = float(foot_sticking_tolerance)
        self.whitelist_margin = float(whitelist_margin)
        self.ab_top_k_pairs = int(ab_top_k_pairs)
        self.contact_slack_weight = float(contact_slack_weight)
        self.max_source_edge_len = float(max_source_edge_len)
        self.cross_agent_rescue_k = int(cross_agent_rescue_k)
        self.ab_pair_whitelist = ab_pair_whitelist or set()

        # Load dual-scene MuJoCo model
        self.robot_model = mujoco.MjModel.from_xml_path(dual_scene_xml_path)
        self.robot_data = mujoco.MjData(self.robot_model)

        # Validate source joints
        self.demo_joints_a = list(self.task_constants_a.DEMO_JOINTS)
        self.demo_joints_b = list(self.task_constants_b.DEMO_JOINTS)
        if len(self.demo_joints_a) < 22:
            raise ValueError(
                f"Expected at least 22 demo joints for A, got {len(self.demo_joints_a)}."
            )
        if len(self.demo_joints_b) < 22:
            raise ValueError(
                f"Expected at least 22 demo joints for B, got {len(self.demo_joints_b)}."
            )
        self.source_joint_names_a = self.demo_joints_a[:22]
        self.source_joint_names_b = self.demo_joints_b[:22]
        self.num_source_a = len(self.source_joint_names_a)
        self.num_source_b = len(self.source_joint_names_b)
        self.num_source_total = self.num_source_a + self.num_source_b

        self.base_joint_to_link_a = dict(self.task_constants_a.JOINTS_MAPPING)
        self.base_joint_to_link_b = dict(self.task_constants_b.JOINTS_MAPPING)
        self.base_foot_links_a = list(self.task_constants_a.FOOT_STICKING_LINKS)
        self.base_foot_links_b = list(self.task_constants_b.FOOT_STICKING_LINKS)
        self.robot_dof_expected_a = int(self.task_constants_a.ROBOT_DOF)
        self.robot_dof_expected_b = int(self.task_constants_b.ROBOT_DOF)

        self._setup_joint_layouts()
        self._setup_mapped_links()
        self._setup_foot_support_points()
        self._setup_bounds_and_costs()
        self._setup_source_graph_indices()

    # -----------------------------------------------------------------------
    # Setup helpers (same architecture as DualInteractionMeshRetargeter)
    # -----------------------------------------------------------------------

    def _setup_joint_layouts(self) -> None:
        free_joint_ids = [
            j
            for j in range(self.robot_model.njnt)
            if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joint_ids) < 2:
            raise ValueError(
                "IAMRRetargeter expects a dual scene with at least two free joints."
            )

        free_joint_ids_sorted = sorted(
            free_joint_ids, key=lambda j: int(self.robot_model.jnt_qposadr[j])
        )

        joint_id_a: int | None = None
        joint_id_b: int | None = None
        for j in free_joint_ids_sorted:
            body_id = int(self.robot_model.jnt_bodyid[j])
            body_name = (
                mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
                or ""
            )
            if joint_id_a is None and body_name.startswith(self.robot_a_prefix):
                joint_id_a = int(j)
            if joint_id_b is None and body_name.startswith(self.robot_b_prefix):
                joint_id_b = int(j)

        if joint_id_a is None or joint_id_b is None:
            joint_id_a = int(free_joint_ids_sorted[0])
            joint_id_b = int(free_joint_ids_sorted[1])

        self.free_joint_id_a = joint_id_a
        self.free_joint_id_b = joint_id_b

        self.qadr_a = int(self.robot_model.jnt_qposadr[self.free_joint_id_a])
        self.qadr_b = int(self.robot_model.jnt_qposadr[self.free_joint_id_b])

        qadr_all_free = sorted(
            int(self.robot_model.jnt_qposadr[j]) for j in free_joint_ids
        )

        def _next_qadr(qadr: int) -> int:
            return next((q for q in qadr_all_free if q > qadr), int(self.robot_model.nq))

        qend_a = _next_qadr(self.qadr_a)
        qend_b = _next_qadr(self.qadr_b)
        if qend_a <= self.qadr_a or qend_b <= self.qadr_b:
            raise ValueError("Invalid free-joint qpos layout in dual scene model.")

        self.q_block_a = np.arange(self.qadr_a, qend_a, dtype=int)
        self.q_block_b = np.arange(self.qadr_b, qend_b, dtype=int)

        if len(self.q_block_a) < 7 or len(self.q_block_b) < 7:
            raise ValueError("Each robot block must contain at least freejoint qpos (7).")

        self.robot_dof_a = len(self.q_block_a) - 7
        self.robot_dof_b = len(self.q_block_b) - 7

        if self.robot_dof_a != self.robot_dof_expected_a:
            raise ValueError(
                f"Robot A DOF mismatch: constants={self.robot_dof_expected_a}, "
                f"scene={self.robot_dof_a}."
            )
        if self.robot_dof_b != self.robot_dof_expected_b:
            raise ValueError(
                f"Robot B DOF mismatch: constants={self.robot_dof_expected_b}, "
                f"scene={self.robot_dof_b}."
            )

        self.base_quat_slice_a = slice(self.qadr_a + 3, self.qadr_a + 7)
        self.base_quat_slice_b = slice(self.qadr_b + 3, self.qadr_b + 7)

        start_local_a = 7 + self.q_a_init_idx
        start_local_b = 7 + self.q_b_init_idx
        if (
            start_local_a < 0
            or start_local_b < 0
            or start_local_a >= len(self.q_block_a)
            or start_local_b >= len(self.q_block_b)
        ):
            raise ValueError("q_*_init_idx must be >= -7.")
        self.start_local_a = int(start_local_a)
        self.start_local_b = int(start_local_b)

        self.q_a_indices = self.q_block_a[start_local_a:]
        self.q_b_indices = self.q_block_b[start_local_b:]
        self.nq_a = len(self.q_a_indices)
        self.nq_b = len(self.q_b_indices)
        self.nq_ab = self.nq_a + self.nq_b

        # Precompute natural standing root-z so each robot starts with feet on ground.
        self._natural_standing_z_a = self._compute_natural_standing_z(
            self.robot_a_prefix, self.qadr_a
        )
        self._natural_standing_z_b = self._compute_natural_standing_z(
            self.robot_b_prefix, self.qadr_b
        )

    def _compute_natural_standing_z(self, prefix: str, qadr: int) -> float:
        """Return root-z that places the agent's lowest geometry at z=0.

        Uses the model default pose (qpos0) so it is prefix-independent and
        requires no external measurement.
        """
        q_default = self.robot_model.qpos0.copy()
        self.robot_data.qpos[:] = q_default
        mujoco.mj_forward(self.robot_model, self.robot_data)

        root_z = float(self.robot_data.qpos[qadr + 2])

        min_geom_z = np.inf
        for g in range(self.robot_model.ngeom):
            gname = (
                mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            )
            if not gname.startswith(prefix):
                continue
            # Skip floor/ground plane geoms embedded in the robot model
            if "ground" in gname.lower() or "floor" in gname.lower():
                continue
            min_geom_z = min(min_geom_z, float(self.robot_data.geom_xpos[g, 2]))

        if np.isinf(min_geom_z):
            return root_z  # fallback: no geoms found for this agent

        return float(root_z - min_geom_z)

    def _setup_mapped_links(self) -> None:
        self.laplacian_match_links_a = {
            joint_name: self._prefix_body_point_target(link_target, self.robot_a_prefix)
            for joint_name, link_target in self.base_joint_to_link_a.items()
        }
        self.laplacian_match_links_b = {
            joint_name: self._prefix_body_point_target(link_target, self.robot_b_prefix)
            for joint_name, link_target in self.base_joint_to_link_b.items()
        }

        self.foot_links_a = {
            f"{self.robot_a_prefix}{ln}": f"{self.robot_a_prefix}{ln}"
            for ln in self.base_foot_links_a
        }
        self.foot_links_b = {
            f"{self.robot_b_prefix}{ln}": f"{self.robot_b_prefix}{ln}"
            for ln in self.base_foot_links_b
        }

    def _setup_foot_support_points(self) -> None:
        self.foot_support_points_a = self._build_foot_support_points(self.foot_links_a)
        self.foot_support_points_b = self._build_foot_support_points(self.foot_links_b)

    def _build_foot_support_points(
        self,
        foot_links: dict[str, str],
    ) -> list[_FootSupportPoint]:
        support_points: list[_FootSupportPoint] = []
        for body_name in foot_links:
            side = self._infer_foot_side(body_name)
            if side is None:
                continue
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id == -1:
                continue

            clearance = 0.0
            for geom_id in range(self.robot_model.ngeom):
                if int(self.robot_model.geom_bodyid[geom_id]) != body_id:
                    continue
                if int(self.robot_model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    clearance = max(clearance, float(self.robot_model.geom_size[geom_id, 0]))

            support_points.append(
                _FootSupportPoint(
                    body_name=body_name,
                    side=side,
                    x_local=float(self.robot_model.body_pos[body_id, 0]),
                    clearance=clearance,
                )
            )

        return sorted(support_points, key=lambda pt: (pt.side, pt.x_local, pt.body_name))

    @staticmethod
    def _infer_foot_side(body_name: str) -> str | None:
        lower = body_name.lower()
        if "left" in lower:
            return "left"
        if "right" in lower:
            return "right"
        return None

    @staticmethod
    def _prefix_body_point_target(
        target: str | tuple[str, np.ndarray] | _BodyPointTarget,
        prefix: str,
    ) -> str | _BodyPointTarget:
        if isinstance(target, _BodyPointTarget):
            return _BodyPointTarget(
                body_name=f"{prefix}{target.body_name}",
                point_offset=np.asarray(target.point_offset, dtype=float).reshape(3),
            )
        if isinstance(target, tuple):
            body_name, point_offset = target
            return _BodyPointTarget(
                body_name=f"{prefix}{body_name}",
                point_offset=np.asarray(point_offset, dtype=float).reshape(3),
            )
        return f"{prefix}{target}"

    @staticmethod
    def _resolve_body_point_target(
        target: str | tuple[str, np.ndarray] | _BodyPointTarget,
    ) -> tuple[str, np.ndarray]:
        if isinstance(target, _BodyPointTarget):
            return target.body_name, np.asarray(target.point_offset, dtype=float).reshape(3)
        if isinstance(target, tuple):
            body_name, point_offset = target
            return body_name, np.asarray(point_offset, dtype=float).reshape(3)
        return target, np.zeros(3, dtype=float)

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
        self.q_b_lb = lower[self.q_b_indices].copy()
        self.q_b_ub = upper[self.q_b_indices].copy()

        for name, lb_dict, ub_dict, cost_dict, nq_x, lb_x, ub_x, cost_x in [
            (
                "A",
                getattr(self.task_constants_a, "MANUAL_LB", {}),
                getattr(self.task_constants_a, "MANUAL_UB", {}),
                getattr(self.task_constants_a, "MANUAL_COST", {}),
                self.nq_a, self.q_a_lb, self.q_a_ub, None,
            ),
            (
                "B",
                getattr(self.task_constants_b, "MANUAL_LB", {}),
                getattr(self.task_constants_b, "MANUAL_UB", {}),
                getattr(self.task_constants_b, "MANUAL_COST", {}),
                self.nq_b, self.q_b_lb, self.q_b_ub, None,
            ),
        ]:
            _ = name  # suppress linter
            for k, v in lb_dict.items():
                idx = int(k)
                if 0 <= idx < nq_x:
                    lb_x[idx] = float(v)
            for k, v in ub_dict.items():
                idx = int(k)
                if 0 <= idx < nq_x:
                    ub_x[idx] = float(v)

        self.Q_diag_a = np.zeros(self.nq_a, dtype=float)
        self.Q_diag_b = np.zeros(self.nq_b, dtype=float)
        for k, v in getattr(self.task_constants_a, "MANUAL_COST", {}).items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.Q_diag_a[idx] = float(v)
        for k, v in getattr(self.task_constants_b, "MANUAL_COST", {}).items():
            idx = int(k)
            if 0 <= idx < self.nq_b:
                self.Q_diag_b[idx] = float(v)

    def _setup_source_graph_indices(self) -> None:
        self.source_name_to_idx_a = {n: i for i, n in enumerate(self.source_joint_names_a)}
        self.source_name_to_idx_b = {n: i for i, n in enumerate(self.source_joint_names_b)}

        # Critical joints that must always have cross-agent rescue edges in E_inter
        cross_critical = {
            "L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
        }
        # cross_rescue_a: local A indices (0..N_a-1)
        self.cross_rescue_a = [
            self.source_name_to_idx_a[n]
            for n in cross_critical
            if n in self.source_name_to_idx_a
        ]
        # cross_rescue_b_local: local B indices (0..N_b-1) — global offset applied at call time
        self.cross_rescue_b_local = [
            self.source_name_to_idx_b[n]
            for n in cross_critical
            if n in self.source_name_to_idx_b
        ]

        # Foot joint local indices for ground grid centering
        foot_names = {"L_Foot", "R_Foot", "L_Ankle", "R_Ankle"}
        self.foot_local_idx_a = [
            self.source_name_to_idx_a[n]
            for n in foot_names
            if n in self.source_name_to_idx_a
        ] or list(range(min(2, self.num_source_a)))
        self.foot_local_idx_b = [
            self.source_name_to_idx_b[n]
            for n in foot_names
            if n in self.source_name_to_idx_b
        ] or list(range(min(2, self.num_source_b)))

        self.source_foot_phase_indices_a = self._build_source_foot_phase_indices(
            self.source_name_to_idx_a
        )
        self.source_foot_phase_indices_b = self._build_source_foot_phase_indices(
            self.source_name_to_idx_b
        )

        # Per-vertex weights for J_self Laplacian (critical joints get 2.5×)
        critical_w = {
            "Pelvis", "Spine", "Spine1", "Spine2", "Spine3", "Chest",
            "L_Wrist", "R_Wrist", "L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder",
        }
        self.laplacian_weights = np.ones(self.num_source_total, dtype=float)
        for i, name in enumerate(self.source_joint_names_a):
            if name in critical_w:
                self.laplacian_weights[i] = 2.5
        for i, name in enumerate(self.source_joint_names_b):
            if name in critical_w:
                self.laplacian_weights[self.num_source_a + i] = 2.5

        # Build SMPL-X → robot joint rotation map for J_rot.
        # Must be called after q_block / qadr / nq fields are set.
        self._smplx_rot_map_a = self._build_smplx_rot_map("A")
        self._smplx_rot_map_b = self._build_smplx_rot_map("B")
        self._smplx_rot_indices_a = np.array(
            [x[1] for x in self._smplx_rot_map_a], dtype=int
        )
        self._smplx_rot_indices_b = np.array(
            [x[1] for x in self._smplx_rot_map_b], dtype=int
        )

    # -----------------------------------------------------------------------
    # SMPL-X → robot joint rotation map  (for J_rot, §3.2)
    # -----------------------------------------------------------------------

    def _build_smplx_rot_map(
        self, agent: str
    ) -> list[tuple[int, int, np.ndarray]]:
        """Build (smplx_joint_idx, local_qpos_idx, axis) entries for J_rot.

        Scans all hinge joints of the given agent in the dual scene, strips the
        agent prefix, then matches against ``_SMPLX_TO_JOINT_KEYWORDS`` to find
        the corresponding SMPL-X body joint index.  The local_qpos_idx is the
        position of the joint within the optimisation variable dqa / dqb.

        Returns a list of (smplx_idx, local_qpos_idx, axis_3d) tuples.
        """
        prefix = self.robot_a_prefix if agent == "A" else self.robot_b_prefix
        qadr_start = self.qadr_a if agent == "A" else self.qadr_b
        start_local = self.start_local_a if agent == "A" else self.start_local_b
        nq = self.nq_a if agent == "A" else self.nq_b

        smplx_name_to_idx: dict[str, int] = {
            n: i for i, n in enumerate(_SMPLX_BODY_JOINT_NAMES)
        }

        mapping: list[tuple[int, int, np.ndarray]] = []
        seen_local: set[int] = set()

        for j in range(self.robot_model.njnt):
            if int(self.robot_model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                continue
            jname = (
                mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            )
            if not jname.startswith(prefix):
                continue
            base_jname = jname[len(prefix):]
            qadr = int(self.robot_model.jnt_qposadr[j])
            axis = self.robot_model.jnt_axis[j].copy()

            for smplx_name, keywords in _SMPLX_TO_JOINT_KEYWORDS.items():
                if smplx_name not in smplx_name_to_idx:
                    continue
                for kw in keywords:
                    if base_jname.lower().startswith(kw):
                        smplx_idx = smplx_name_to_idx[smplx_name]
                        pos_in_block = qadr - qadr_start
                        local_qpos_idx = pos_in_block - start_local
                        if 0 <= local_qpos_idx < nq and local_qpos_idx not in seen_local:
                            seen_local.add(local_qpos_idx)
                            mapping.append((smplx_idx, local_qpos_idx, axis))
                        break

        if self.debug:
            print(
                f"[IAMR] SMPL-X rot map agent {agent}: "
                f"{len(mapping)} joint(s) matched"
            )
        return mapping

    @staticmethod
    def _build_source_foot_phase_indices(
        source_name_to_idx: dict[str, int],
    ) -> dict[str, dict[str, int | None]]:
        return {
            "left": {
                "toe": source_name_to_idx.get("L_Foot"),
                "heel": source_name_to_idx.get("L_Ankle"),
            },
            "right": {
                "toe": source_name_to_idx.get("R_Foot"),
                "heel": source_name_to_idx.get("R_Ankle"),
            },
        }

    def _classify_source_foot_phases(
        self,
        raw_joints_t: np.ndarray,
        scale: float,
        flags: dict[str, bool],
        *,
        agent: str,
    ) -> dict[str, str]:
        indices = (
            self.source_foot_phase_indices_a
            if agent == "A"
            else self.source_foot_phase_indices_b
        )
        left_flag, right_flag = self._extract_lr_flags(flags)
        phases = {"left": "swing", "right": "swing"}

        for side, active in (("left", left_flag), ("right", right_flag)):
            if not active:
                continue
            toe_idx = indices[side]["toe"]
            heel_idx = indices[side]["heel"]
            toe_z = np.inf if toe_idx is None else float(scale * raw_joints_t[toe_idx, 2])
            heel_z = np.inf if heel_idx is None else float(scale * raw_joints_t[heel_idx, 2])
            toe_grounded = toe_z < self.foot_ground_threshold
            heel_grounded = heel_z < self.foot_ground_threshold

            if toe_grounded and heel_grounded:
                phases[side] = "flat"
            elif toe_grounded:
                phases[side] = "toe"
            elif heel_grounded:
                phases[side] = "heel"

        return phases

    def _compute_smplx_rot_targets(
        self,
        smplx_poses: np.ndarray,
        agent: str,
    ) -> np.ndarray:
        """Project SMPL-X axis-angle body poses onto robot joint axes.

        For each matched robot hinge joint j with SMPL-X body joint k:
            θ̂_j(t) = â_j · pose_body(t, k)

        where â_j is the joint axis in its local parent frame and
        pose_body(t, k) is the 3-D axis-angle rotation of SMPL-X joint k
        at frame t (from the ``pose_body`` array, NOT root_orient).

        Args:
            smplx_poses: (T, 21, 3) float — SMPL-X ``pose_body`` array.
            agent:       "A" or "B".

        Returns:
            (T, nq) array of per-frame per-joint target angles.
        """
        T = smplx_poses.shape[0]
        nq = self.nq_a if agent == "A" else self.nq_b
        rot_map = self._smplx_rot_map_a if agent == "A" else self._smplx_rot_map_b

        theta_hat = np.zeros((T, nq), dtype=float)
        for smplx_idx, local_qpos_idx, axis in rot_map:
            # pose_body shape: (T, 21, 3) — axis-angle per frame per joint
            theta_hat[:, local_qpos_idx] = smplx_poses[:, smplx_idx, :] @ axis
        return theta_hat

    # -----------------------------------------------------------------------
    # Dual reference manifold construction  (§3.1 of arXiv:2603.02856)
    # -----------------------------------------------------------------------

    @staticmethod
    def compute_scale_factors(
        height_a: float,
        height_b: float,
        robot_height_a: float,
        robot_height_b: float,
    ) -> tuple[float, float, float]:
        """Compute s^(A), s^(B), and s_uni as per the paper.

        s^(k) = h_robot^(k) / h_raw^(k)
        s_uni = (s^(A) + s^(B)) / 2
        """
        s_a = robot_height_a / height_a
        s_b = robot_height_b / height_b
        s_uni = (s_a + s_b) / 2.0
        return float(s_a), float(s_b), float(s_uni)

    @staticmethod
    def build_dual_manifolds(
        raw_a: np.ndarray,
        raw_b: np.ndarray,
        s_a: float,
        s_b: float,
        s_uni: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build P_ind and P_uni for a single frame.

        Args:
            raw_a: (N_a, 3) height-normalised but unscaled joints for A.
            raw_b: (N_b, 3) height-normalised but unscaled joints for B.
            s_a:   individual scale factor for A.
            s_b:   individual scale factor for B.
            s_uni: unified scale factor.

        Returns:
            P_ind: (N_a + N_b, 3) individually-scaled stacked joints.
            P_uni: (N_a + N_b, 3) uniformly-scaled stacked joints.
        """
        P_ind = np.vstack([s_a * raw_a, s_b * raw_b])
        P_uni = np.vstack([s_uni * raw_a, s_uni * raw_b])
        return P_ind, P_uni

    # -----------------------------------------------------------------------
    # Ground-node helpers (grounding via Laplacian mesh, like single-robot version)
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_ground_grid(center_xy: np.ndarray, ground_range: float, n_pts: int) -> np.ndarray:
        """Return a flat grid of (n_pts, 3) points at z=0 centred on center_xy."""
        side = max(1, int(round(n_pts ** 0.5)))
        xs = np.linspace(center_xy[0] - ground_range, center_xy[0] + ground_range, side)
        ys = np.linspace(center_xy[1] - ground_range, center_xy[1] + ground_range, side)
        X, Y = np.meshgrid(xs, ys)
        return np.stack([X.ravel(), Y.ravel(), np.zeros(side * side)], axis=1).astype(float)

    def _extend_manifolds_with_ground(
        self,
        P_ind: np.ndarray,
        P_uni: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Append per-agent ground grids to P_ind and P_uni.

        Vertex layout after extension:
            [A_joints (N_a), A_ground (G), B_joints (N_b), B_ground (G)]

        Returns:
            P_ind_ext, P_uni_ext,
            n_verts_a — extended A vertex count (N_a + G), used as partition boundary.
        """
        n_a = self.num_source_a
        G = self.n_ground

        foot_xy_a = np.mean(P_ind[self.foot_local_idx_a, :2], axis=0)
        foot_xy_b = np.mean(P_ind[n_a + np.array(self.foot_local_idx_b), :2], axis=0)

        ground_a = self._make_ground_grid(foot_xy_a, self.ground_range, G)
        ground_b = self._make_ground_grid(foot_xy_b, self.ground_range, G)

        # ground points are already in world scale (z=0); insert between A and B blocks
        P_ind_ext = np.vstack([P_ind[:n_a], ground_a, P_ind[n_a:], ground_b])
        P_uni_ext = np.vstack([P_uni[:n_a], ground_a, P_uni[n_a:], ground_b])
        return P_ind_ext, P_uni_ext, n_a + G

    # -----------------------------------------------------------------------
    # Source graph construction + topological partitioning  (§3.2)
    # -----------------------------------------------------------------------

    def build_source_graph_iamr(
        self,
        P_ind: np.ndarray,
        P_uni: np.ndarray,
        n_verts_a: int | None = None,
    ) -> tuple[list[list[int]], list[list[int]], set[tuple[int, int]]]:
        """Build the interaction mesh graph and partition edges into E_self / E_inter.

        Delaunay tetrahedralisation is performed on P_uni (preserves interaction
        geometry), then edges are partitioned by agent membership.

        Args:
            P_ind: (N_total, 3) individually-scaled vertices (may include ground nodes).
            P_uni: (N_total, 3) unified-scale vertices  (used for Delaunay).
            n_verts_a: extended A vertex count (N_a + G) used as partition boundary.
                       Defaults to self.num_source_a when ground nodes are absent.

        Returns:
            self_adj_list:  E_self adjacency list (intra-agent only).
            full_adj_list:  all-edge adjacency list (for diagnostics).
            inter_edges:    set of (i, j) cross-agent edge pairs.
        """
        _n_verts_a = n_verts_a if n_verts_a is not None else self.num_source_a
        all_edges = self._delaunay_edges(P_uni)
        all_edges = self._prune_long_edges(P_uni, all_edges, self.max_source_edge_len)
        all_edges = self._add_cross_agent_rescue_edges(P_uni, all_edges, _n_verts_a)

        E_self, E_inter = self._partition_edges(all_edges, _n_verts_a)

        n = P_uni.shape[0]
        self_adj_list = self._edges_to_adj_list(E_self, n)
        full_adj_list = self._edges_to_adj_list(all_edges, n)

        return self_adj_list, full_adj_list, E_inter

    def _partition_edges(
        self, edges: set[tuple[int, int]], n_verts_a: int | None = None,
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Split edges into E_self (intra-agent) and E_inter (cross-agent)."""
        n_a = n_verts_a if n_verts_a is not None else self.num_source_a
        E_self: set[tuple[int, int]] = set()
        E_inter: set[tuple[int, int]] = set()
        for i, j in edges:
            if (i < n_a) == (j < n_a):
                E_self.add((i, j))
            else:
                E_inter.add((i, j))
        return E_self, E_inter

    def _delaunay_edges(self, vertices: np.ndarray) -> set[tuple[int, int]]:
        """Compute Delaunay tetrahedra and extract unique edges."""
        try:
            simplices = Delaunay(vertices).simplices
        except Exception:
            n = vertices.shape[0]
            if n < 4:
                raise
            d = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)
            np.fill_diagonal(d, np.inf)
            simplices = np.array(
                [[i] + list(np.argsort(d[i])[:3]) for i in range(n)]
            )
        return self._tetrahedra_to_edges(simplices)

    @staticmethod
    def _tetrahedra_to_edges(simplices: np.ndarray) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for tet in simplices:
            a, b, c, d = (int(x) for x in tet)
            for p, q in [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]:
                edges.add((p, q) if p < q else (q, p))
        return edges

    @staticmethod
    def _prune_long_edges(
        vertices: np.ndarray, edges: set[tuple[int, int]], max_len: float
    ) -> set[tuple[int, int]]:
        return {
            (i, j)
            for i, j in edges
            if np.linalg.norm(vertices[i] - vertices[j]) <= max_len
        }

    def _add_cross_agent_rescue_edges(
        self,
        vertices: np.ndarray,
        edges: set[tuple[int, int]],
        n_verts_a: int | None = None,
    ) -> set[tuple[int, int]]:
        """Ensure critical joints always have at least one cross-agent edge."""
        n_a = n_verts_a if n_verts_a is not None else self.num_source_a
        n_b = self.num_source_b
        # Only original joint vertices participate in rescue edges (not ground nodes)
        a_ids = np.arange(0, self.num_source_a, dtype=int)
        b_ids = np.arange(n_a, n_a + n_b, dtype=int)
        # Global B indices for cross_rescue_b_local
        cross_rescue_b_global = [n_a + li for li in self.cross_rescue_b_local]

        dmat = np.linalg.norm(vertices[:, None] - vertices[None], axis=-1)

        def _cross_count(idx: int) -> int:
            count = 0
            for ei, ej in edges:
                if ei == idx and ej >= n_a:
                    count += 1
                elif ej == idx and ei >= n_a:
                    count += 1
            return count

        for idx in self.cross_rescue_a:
            if _cross_count(idx) == 0:
                nbrs = b_ids[np.argsort(dmat[idx, b_ids])[: self.cross_agent_rescue_k]]
                for j in nbrs:
                    edges.add((idx, int(j)) if idx < int(j) else (int(j), idx))

        for idx in cross_rescue_b_global:
            if _cross_count(idx) == 0:
                nbrs = a_ids[np.argsort(dmat[idx, a_ids])[: self.cross_agent_rescue_k]]
                for j in nbrs:
                    edges.add((int(j), idx) if int(j) < idx else (idx, int(j)))

        return edges

    @staticmethod
    def _edges_to_adj_list(
        edges: set[tuple[int, int]], n_vertices: int
    ) -> list[list[int]]:
        adj: list[list[int]] = [[] for _ in range(n_vertices)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    # -----------------------------------------------------------------------
    # Inter-agent spring data  (J_inter, §3.2)
    # -----------------------------------------------------------------------

    def compute_inter_spring_data(
        self,
        P_uni: np.ndarray,
        inter_edges: set[tuple[int, int]],
        n_verts_a: int | None = None,
    ) -> list[tuple[int, int, float, np.ndarray]]:
        """Compute per-edge spring weights and unified-manifold targets.

        Returns list of (i, j, omega_ij, r_hat_ij) for each inter-agent edge:
            omega_ij = omega_max * exp(-gamma * d_ij)
            r_hat_ij = P_uni[i] - P_uni[j]   (target relative position)

        Ground nodes (A: N_a..n_verts_a-1, B: n_verts_a+N_b..) are excluded
        from inter-agent springs — they only appear in intra-agent Laplacian edges.
        """
        n_a_orig = self.num_source_a
        n_a_ext = n_verts_a if n_verts_a is not None else n_a_orig
        n_b_orig = self.num_source_b

        spring_data: list[tuple[int, int, float, np.ndarray]] = []
        for i, j in inter_edges:
            # Skip edges where either vertex is a ground node
            i_is_ground = n_a_orig <= i < n_a_ext  # A ground
            j_is_ground = j >= n_a_ext + n_b_orig   # B ground
            if i_is_ground or j_is_ground:
                continue
            d_ij = float(np.linalg.norm(P_uni[i] - P_uni[j]))
            omega_ij = self.omega_max * np.exp(-self.gamma_decay * d_ij)
            r_hat_ij = (P_uni[i] - P_uni[j]).astype(float)
            spring_data.append((i, j, omega_ij, r_hat_ij))
        return spring_data

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def retarget_motion(
        self,
        human_joints_a_raw: np.ndarray,
        human_joints_b_raw: np.ndarray,
        height_a: float,
        height_b: float,
        *,
        foot_sticking_sequences_a: list[dict[str, bool]] | None = None,
        foot_sticking_sequences_b: list[dict[str, bool]] | None = None,
        smplx_poses_a: np.ndarray | None = None,
        smplx_poses_b: np.ndarray | None = None,
        q_init_a: np.ndarray | None = None,
        q_init_b: np.ndarray | None = None,
        dest_res_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Retarget dual-human motion via IAMR.

        Args:
            human_joints_a_raw: (T, J, 3) height-normalised but unscaled joints for A.
            human_joints_b_raw: (T, J, 3) height-normalised but unscaled joints for B.
            height_a:  body height of human A in metres (h_raw^(A)).
            height_b:  body height of human B in metres (h_raw^(B)).
            foot_sticking_sequences_a: per-frame contact flags for A.
            foot_sticking_sequences_b: per-frame contact flags for B.
            smplx_poses_a: (T, 21, 3) SMPL-X ``pose_body`` axis-angle for A.
                           Used as θ̂_k^src in J_rot (paper §3.2).  If None,
                           J_rot is disabled for agent A.
            smplx_poses_b: same as smplx_poses_a for agent B.
            q_init_a:   initial local robot state for A [xyz, wxyz, joints].
            q_init_b:   initial local robot state for B.
            dest_res_path: optional path to save outputs as .npz.

        Returns:
            qpos_a:            (T, 7 + dof_a) retargeted joint configurations for A.
            qpos_b:            (T, 7 + dof_b) retargeted joint configurations for B.
            interaction_graph: (T, N_a, N_b) bool — active E_inter per frame.
            contact_graph:     (T, N_a_links, N_b_links) bool — A–B physical contact.
        """
        if human_joints_a_raw.shape[0] != human_joints_b_raw.shape[0]:
            raise ValueError(
                f"Frame count mismatch: A={human_joints_a_raw.shape[0]}, "
                f"B={human_joints_b_raw.shape[0]}"
            )
        if human_joints_a_raw.ndim != 3 or human_joints_a_raw.shape[2] != 3:
            raise ValueError(f"Expected (T, J, 3), got {human_joints_a_raw.shape}")

        num_frames = human_joints_a_raw.shape[0]

        # ---- Dual manifold scale factors ----
        robot_height_a = float(getattr(self.task_constants_a, "ROBOT_HEIGHT", 1.78))
        robot_height_b = float(getattr(self.task_constants_b, "ROBOT_HEIGHT", 1.78))
        s_a, s_b, s_uni = self.compute_scale_factors(
            height_a, height_b, robot_height_a, robot_height_b
        )

        if self.debug:
            print(
                f"[IAMR] Scale factors — s_A={s_a:.4f}, s_B={s_b:.4f}, "
                f"s_uni={s_uni:.4f}"
            )

        # ---- Normalise foot sticking sequences ----
        foot_a = self._normalise_foot_seq(foot_sticking_sequences_a, num_frames)
        foot_b = self._normalise_foot_seq(foot_sticking_sequences_b, num_frames)

        # ---- Compute J_rot targets from SMPL-X source rotations (§3.2) ----
        # θ̂_k^src = pose_body axis-angle projected onto each robot joint axis.
        q_nom_a: np.ndarray | None = None
        q_nom_b: np.ndarray | None = None
        if smplx_poses_a is not None:
            poses_a = np.asarray(smplx_poses_a, dtype=float)
            if poses_a.shape[0] != num_frames or poses_a.ndim != 3 or poses_a.shape[1] < 21:
                raise ValueError(
                    f"smplx_poses_a must be ({num_frames}, 21, 3), "
                    f"got {poses_a.shape}"
                )
            q_nom_a = self._compute_smplx_rot_targets(poses_a[:, :21], "A")
        if smplx_poses_b is not None:
            poses_b = np.asarray(smplx_poses_b, dtype=float)
            if poses_b.shape[0] != num_frames or poses_b.ndim != 3 or poses_b.shape[1] < 21:
                raise ValueError(
                    f"smplx_poses_b must be ({num_frames}, 21, 3), "
                    f"got {poses_b.shape}"
                )
            q_nom_b = self._compute_smplx_rot_targets(poses_b[:, :21], "B")

        # ---- Initialise qpos ----
        # Default base init: place each robot's free-joint XYZ at the scaled
        # Pelvis position (source joint 0) of the first frame so the robot
        # starts above ground.  Overridden by explicit q_init_a/b.
        q = np.copy(self.robot_data.qpos)
        if q_init_a is None:
            pelvis_a = (s_a * human_joints_a_raw[0, 0, :]).astype(float)
            q_a0 = np.zeros(7 + self.robot_dof_a, dtype=float)
            q_a0[:2] = pelvis_a[:2]                    # XY from scaled human pelvis
            q_a0[2] = self._natural_standing_z_a       # Z: robot's natural standing height
            q_a0[3] = 1.0  # quaternion w (identity / upright)
            self._write_local_robot_state(q, q_a0, agent="A")
        else:
            self._write_local_robot_state(q, q_init_a, agent="A")
        if q_init_b is None:
            pelvis_b = (s_b * human_joints_b_raw[0, 0, :]).astype(float)
            q_b0 = np.zeros(7 + self.robot_dof_b, dtype=float)
            q_b0[:2] = pelvis_b[:2]                    # XY from scaled human pelvis
            q_b0[2] = self._natural_standing_z_b       # Z: robot's natural standing height
            q_b0[3] = 1.0
            self._write_local_robot_state(q, q_b0, agent="B")
        else:
            self._write_local_robot_state(q, q_init_b, agent="B")

        q_prev_frame = q.copy()

        qpos_a_list: list[np.ndarray] = []
        qpos_b_list: list[np.ndarray] = []
        igraph_list: list[np.ndarray] = []  # interaction_graph per frame
        cgraph_list: list[np.ndarray] = []  # contact_graph per frame

        with tqdm(range(num_frames), desc="IAMR retargeting") as pbar:
            for t in pbar:
                raw_a_t = human_joints_a_raw[t, : self.num_source_a]
                raw_b_t = human_joints_b_raw[t, : self.num_source_b]

                P_ind, P_uni = self.build_dual_manifolds(raw_a_t, raw_b_t, s_a, s_b, s_uni)

                # Source B foot z values in scaled world frame (before ground extension).
                # Used to detect when source person's foot is on the ground.
                source_b_foot_z = np.array([
                    s_b * raw_b_t[li, 2]
                    for li in self.foot_local_idx_b
                ])
                source_foot_phase_a = self._classify_source_foot_phases(
                    raw_a_t, s_a, foot_a[t], agent="A"
                )
                source_foot_phase_b = self._classify_source_foot_phases(
                    raw_b_t, s_b, foot_b[t], agent="B"
                )

                # Extend with per-agent ground grids (layout: [A_jnts, A_gnd, B_jnts, B_gnd])
                P_ind, P_uni, n_verts_a = self._extend_manifolds_with_ground(P_ind, P_uni)

                self_adj, _, inter_edges = self.build_source_graph_iamr(
                    P_ind, P_uni, n_verts_a=n_verts_a
                )
                spring_data = self.compute_inter_spring_data(
                    P_uni, inter_edges, n_verts_a=n_verts_a
                )

                n_iter = self.n_iter_first if t == 0 else self.n_iter_other
                q, cost = self.iterate(
                    q_n=q,
                    q_t_last=q_prev_frame,
                    P_ind=P_ind,
                    P_uni=P_uni,
                    self_adj_list=self_adj,
                    spring_data=spring_data,
                    foot_sticking_a=foot_a[t],
                    foot_sticking_b=foot_b[t],
                    q_nominal_a=(q_nom_a[t] if q_nom_a is not None else None),
                    q_nominal_b=(q_nom_b[t] if q_nom_b is not None else None),
                    n_iter=n_iter,
                    init_t=(t == 0),
                    n_verts_a=n_verts_a,
                    source_b_foot_z=source_b_foot_z,
                    source_foot_phase_a=source_foot_phase_a,
                    source_foot_phase_b=source_foot_phase_b,
                )

                qpos_a_list.append(self._extract_local_robot_state(q, agent="A"))
                qpos_b_list.append(self._extract_local_robot_state(q, agent="B"))
                igraph_list.append(self._extract_interaction_graph(inter_edges, n_verts_a))
                cgraph_list.append(self._extract_contact_graph(q))
                q_prev_frame = q.copy()
                pbar.set_postfix(cost=f"{cost:.4f}")

        qpos_a = np.asarray(qpos_a_list, dtype=np.float32)
        qpos_b = np.asarray(qpos_b_list, dtype=np.float32)
        interaction_graph = np.asarray(igraph_list, dtype=bool)
        contact_graph = np.asarray(cgraph_list, dtype=bool)

        if dest_res_path is not None:
            np.savez(
                dest_res_path,
                qpos_A=qpos_a,
                qpos_B=qpos_b,
                interaction_graph=interaction_graph,
                contact_graph=contact_graph,
                human_joints_A=human_joints_a_raw,
                human_joints_B=human_joints_b_raw,
                scale_A=s_a,
                scale_B=s_b,
                scale_uni=s_uni,
            )

        return qpos_a, qpos_b, interaction_graph, contact_graph

    # -----------------------------------------------------------------------
    # SQP loop
    # -----------------------------------------------------------------------

    def iterate(
        self,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        P_ind: np.ndarray,
        P_uni: np.ndarray,
        self_adj_list: list[list[int]],
        spring_data: list[tuple[int, int, float, np.ndarray]],
        foot_sticking_a: dict[str, bool],
        foot_sticking_b: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        q_nominal_b: np.ndarray | None,
        *,
        n_iter: int,
        init_t: bool,
        n_verts_a: int | None = None,
        source_b_foot_z: np.ndarray | None = None,
        source_foot_phase_a: dict[str, str] | None = None,
        source_foot_phase_b: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Run multi-iteration SQP for a single frame."""
        last_cost = np.inf
        q_curr = q_n.copy()
        for _ in range(n_iter):
            q_curr, cost = self.solve_single_iteration(
                q_curr=q_curr,
                q_t_last=q_t_last,
                P_ind=P_ind,
                P_uni=P_uni,
                self_adj_list=self_adj_list,
                spring_data=spring_data,
                foot_sticking_a=foot_sticking_a,
                foot_sticking_b=foot_sticking_b,
                q_nominal_a=q_nominal_a,
                q_nominal_b=q_nominal_b,
                init_t=init_t,
                n_verts_a=n_verts_a,
                source_b_foot_z=source_b_foot_z,
                source_foot_phase_a=source_foot_phase_a,
                source_foot_phase_b=source_foot_phase_b,
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
        P_ind: np.ndarray,
        P_uni: np.ndarray,
        self_adj_list: list[list[int]],
        spring_data: list[tuple[int, int, float, np.ndarray]],
        foot_sticking_a: dict[str, bool],
        foot_sticking_b: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        q_nominal_b: np.ndarray | None,
        init_t: bool = False,
        verbose: bool = False,
        n_verts_a: int | None = None,
        source_b_foot_z: np.ndarray | None = None,
        source_foot_phase_a: dict[str, str] | None = None,
        source_foot_phase_b: dict[str, str] | None = None,
    ) -> tuple[np.ndarray, float]:
        """Single QP subproblem following the IAMR objective.

        Objective:
            w_self  · J_self  (Laplacian, intra-agent edges, P_ind targets)
          + w_inter · J_inter (spring potential, inter-agent edges, P_uni targets)
          + w_reg   · J_reg   (temporal smoothness)
          + λ_rot   · J_rot   (joint-angle tracking towards nominal)
        """
        # --- Kinematics: robot link positions & Jacobians at q_curr --------
        _n_verts_a = n_verts_a if n_verts_a is not None else self.num_source_a
        J_target, target_vertices = self._build_target_jacobian_and_vertices(
            q_curr, P_ind, n_verts_a=_n_verts_a
        )
        q_a_n = q_curr[self.q_a_indices]
        q_b_n = q_curr[self.q_b_indices]

        # --- J_self: Laplacian over intra-agent edges (E_self, P_ind) ------
        L_self = calculate_laplacian_matrix(target_vertices, self_adj_list)
        if not sp.issparse(L_self):
            L_self = sp.csr_matrix(L_self)
        L_self = L_self.tocsr()

        kron_self = sp.kron(L_self, sp.eye(3, format="csr"), format="csr")
        J_lap = kron_self @ J_target                        # (3N, nq_ab)
        lap0 = (L_self @ target_vertices).reshape(-1)       # current Laplacian
        target_lap_vec = calculate_laplacian_coordinates(
            P_ind, self_adj_list
        ).reshape(-1)                                        # target from ℳ_ind

        # CVXPY decision variable: dx = [dqa; dqb]
        dx = cp.Variable(self.nq_ab, name="dx")
        dqa = dx[: self.nq_a]
        dqb = dx[self.nq_a :]

        # Slack for Laplacian equality (avoids rank-deficiency issues)
        n_verts_total = P_ind.shape[0]  # includes ground nodes when present
        lap_slack = cp.Variable(3 * n_verts_total, name="lap_slack")

        constraints: list[cp.Constraint] = []
        constraints.append(cp.Constant(J_lap) @ dx - lap_slack == -lap0)

        # --- J_inter: variable-stiffness spring (E_inter, P_uni) -----------
        # Pre-build batched spring matrix:  M @ dx ≈ b  →  obj += w_inter * ||M@dx - b||²
        n_spring = len(spring_data)
        if n_spring > 0:
            M_spring = np.zeros((n_spring * 3, self.nq_ab), dtype=float)
            b_spring = np.zeros(n_spring * 3, dtype=float)
            for k, (vi, vj, omega_ij, r_hat_ij) in enumerate(spring_data):
                # Relative Jacobian: J_rel = J_i - J_j  (both expressed in dx space)
                J_rel = (
                    J_target[3 * vi : 3 * (vi + 1), :]
                    - J_target[3 * vj : 3 * (vj + 1), :]
                )
                r_curr = target_vertices[vi] - target_vertices[vj]
                sqrt_w = float(np.sqrt(omega_ij))
                M_spring[3 * k : 3 * (k + 1), :] = sqrt_w * J_rel
                b_spring[3 * k : 3 * (k + 1)] = sqrt_w * (r_hat_ij - r_curr)
        else:
            M_spring = np.zeros((0, self.nq_ab), dtype=float)
            b_spring = np.zeros(0, dtype=float)

        # --- Foot-sticking constraints --------------------------------------
        if self.activate_foot_sticking:
            constraints.extend(
                self._build_foot_constraints(q_curr, q_t_last, dqa, foot_sticking_a, agent="A")
            )
            constraints.extend(
                self._build_foot_constraints(q_curr, q_t_last, dqb, foot_sticking_b, agent="B")
            )
            stance_constraints_a, stance_slacks_a = self._build_stance_grounding_constraints(
                q_curr, dqa, source_foot_phase_a, agent="A"
            )
            stance_constraints_b, stance_slacks_b = self._build_stance_grounding_constraints(
                q_curr, dqb, source_foot_phase_b, agent="B"
            )
            constraints.extend(stance_constraints_a)
            constraints.extend(stance_constraints_b)
        else:
            stance_slacks_a = []
            stance_slacks_b = []

        # --- Contact constraints (ground + A–B) ----------------------------
        contact_rows = self._linearize_contact_rows(q_curr)
        constraints.extend(self._build_ground_constraints(contact_rows, dx))
        ab_constraints, slack_ab = self._build_ab_constraints(contact_rows, dx)
        constraints.extend(ab_constraints)

        # --- Joint limits --------------------------------------------------
        if self.activate_joint_limits:
            constraints += [dqa >= (self.q_a_lb - q_a_n), dqa <= (self.q_a_ub - q_a_n)]
            constraints += [dqb >= (self.q_b_lb - q_b_n), dqb <= (self.q_b_ub - q_b_n)]

        # --- Trust-region (SOC) --------------------------------------------
        constraints.append(cp.SOC(self.step_size, dx))

        # -------------------------------------------------------------------
        # Build objective
        # -------------------------------------------------------------------
        obj_terms: list[cp.Expression] = []

        # J_self — weighted Laplacian residual (§3.2, Eq. J_self)
        # Extend laplacian_weights with 1.0 for any ground nodes
        # Layout: [A_joints, A_ground, B_joints, B_ground]
        G = self.n_ground
        n_a_orig = self.num_source_a
        n_b_orig = self.num_source_b
        if n_verts_total > n_a_orig + n_b_orig:
            lap_w = np.concatenate([
                self.laplacian_weights[:n_a_orig],
                np.ones(G),
                self.laplacian_weights[n_a_orig:],
                np.ones(G),
            ])
        else:
            lap_w = self.laplacian_weights
        sqrt_w_lap = np.sqrt(np.repeat(lap_w, 3))
        obj_terms.append(
            self.w_self
            * cp.sum_squares(cp.multiply(sqrt_w_lap, lap_slack - target_lap_vec))
        )

        # J_inter — spring potential (§3.2, Eq. J_inter)
        if n_spring > 0:
            obj_terms.append(
                self.w_inter
                * cp.sum_squares(cp.Constant(M_spring) @ dx - b_spring)
            )

        # J_reg — temporal smoothness (§3.2, Eq. J_reg)
        dx_smooth = np.concatenate([
            q_t_last[self.q_a_indices] - q_a_n,
            q_t_last[self.q_b_indices] - q_b_n,
        ])
        obj_terms.append(self.w_reg * cp.sum_squares(dx - dx_smooth))

        # J_rot — source rotation tracking (§3.2, Eq. J_rot)
        # θ̂_k^src projected onto each robot joint axis via _compute_smplx_rot_targets.
        # Approximates geodesic distance on SO(3) with L2 on joint angles.
        if q_nominal_a is not None and self._smplx_rot_indices_a.size > 0:
            idx = self._smplx_rot_indices_a
            z_a = dqa[idx] - (q_nominal_a[idx] - q_a_n[idx])
            obj_terms.append(self.lambda_rot * cp.sum_squares(z_a))
        if q_nominal_b is not None and self._smplx_rot_indices_b.size > 0:
            idx = self._smplx_rot_indices_b
            z_b = dqb[idx] - (q_nominal_b[idx] - q_b_n[idx])
            obj_terms.append(self.lambda_rot * cp.sum_squares(z_b))

        # J_pelvis_abs — soft absolute pelvis-z anchor (prevents vertical drift).
        # Penalises deviation of each robot's Pelvis z from the scaled-human Pelvis z.
        # Necessary because the Laplacian is translation-invariant: without this term
        # the robot can float freely in z when foot-sticking is inactive.
        if self.w_pelvis_abs > 0 and target_vertices.shape[0] > 0:
            # Agent A pelvis (source joint 0)
            j_pz_a = J_target[2, :]  # z-row of pelvis Jacobian in dx space
            target_pz_a = float(P_ind[0, 2])
            curr_pz_a = float(target_vertices[0, 2])
            obj_terms.append(
                self.w_pelvis_abs * (cp.Constant(j_pz_a) @ dx - (target_pz_a - curr_pz_a)) ** 2
            )
            # Agent B pelvis — first B joint, which is at index n_verts_a in extended layout
            b0 = _n_verts_a
            j_pz_b = J_target[3 * b0 + 2, :]
            target_pz_b = float(P_ind[b0, 2])
            curr_pz_b = float(target_vertices[b0, 2])
            obj_terms.append(
                self.w_pelvis_abs * (cp.Constant(j_pz_b) @ dx - (target_pz_b - curr_pz_b)) ** 2
            )

        # Hard foot-z grounding constraint (elastic slack, same pattern as A-B contact whitelist).
        # When the source person's foot is on the ground (z < foot_ground_threshold),
        # the robot foot vertex must satisfy: foot_z_lin <= source_foot_z + slack, slack >= 0.
        # Slack is penalised at contact_slack_weight so the constraint behaves as near-hard —
        # the solver only violates it when physically impossible within the step.
        if source_b_foot_z is not None:
            foot_gnd_slacks: list[cp.Variable] = []
            for k, li in enumerate(self.foot_local_idx_b):
                if source_b_foot_z[k] >= self.foot_ground_threshold:
                    continue  # source foot not grounded; no constraint this frame
                b_vtx_idx = _n_verts_a + li
                if b_vtx_idx >= target_vertices.shape[0]:
                    continue
                j_fz = J_target[3 * b_vtx_idx + 2, :]
                if np.all(j_fz == 0):
                    continue  # unmapped joint; Jacobian is zero
                eps_foot = float(source_b_foot_z[k])
                foot_z_lin = float(target_vertices[b_vtx_idx, 2]) + cp.Constant(j_fz) @ dx
                s_k = cp.Variable(nonneg=True, name=f"foot_gnd_{k}")
                constraints.append(foot_z_lin <= eps_foot + s_k)
                foot_gnd_slacks.append(s_k)
            if foot_gnd_slacks:
                obj_terms.append(
                    self.contact_slack_weight
                    * cp.sum_squares(cp.hstack(foot_gnd_slacks))
                )

        if stance_slacks_a or stance_slacks_b:
            obj_terms.append(
                self.contact_slack_weight
                * cp.sum_squares(cp.hstack(stance_slacks_a + stance_slacks_b))
            )

        # Q_diag cost — penalise large absolute pose deviations
        if np.any(self.Q_diag_a > 0):
            obj_terms.append(
                cp.sum_squares(cp.multiply(np.sqrt(self.Q_diag_a), dqa + q_a_n))
            )
        if np.any(self.Q_diag_b > 0):
            obj_terms.append(
                cp.sum_squares(cp.multiply(np.sqrt(self.Q_diag_b), dqb + q_b_n))
            )

        # A–B contact slack penalty
        if slack_ab is not None and slack_ab.size > 0:
            obj_terms.append(self.contact_slack_weight * cp.sum_squares(slack_ab))

        # -------------------------------------------------------------------
        # Solve (OSQP as per paper, Clarabel as fallback)
        # -------------------------------------------------------------------
        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        self._solve_with_fallback(problem, verbose=verbose)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and init_t:
            # Drop trust-region SOC and retry (first frame only)
            constraints_no_soc = [
                c for c in constraints
                if not isinstance(c, cp.constraints.second_order.SOC)
            ]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints_no_soc)
            self._solve_with_fallback(problem, verbose=verbose)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"IAMR CVXPY solve failed: {problem.status}")

        dx_star = np.asarray(dx.value, dtype=float)
        q_next = q_curr.copy()
        q_next[self.q_a_indices] = q_a_n + dx_star[: self.nq_a]
        q_next[self.q_b_indices] = q_b_n + dx_star[self.nq_a :]
        # Re-normalise base quaternions
        q_next[self.base_quat_slice_a] /= (
            np.linalg.norm(q_next[self.base_quat_slice_a]) + 1e-12
        )
        q_next[self.base_quat_slice_b] /= (
            np.linalg.norm(q_next[self.base_quat_slice_b]) + 1e-12
        )
        return q_next, float(problem.value)

    @staticmethod
    def _solve_with_fallback(problem: cp.Problem, verbose: bool = False) -> None:
        """Try OSQP (paper solver) then fall back to Clarabel."""
        try:
            problem.solve(solver=cp.OSQP, verbose=verbose, eps_abs=1e-5, eps_rel=1e-5)
            if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                return
        except Exception:
            pass
        problem.solve(solver=cp.CLARABEL, verbose=verbose)

    # -----------------------------------------------------------------------
    # Output graph extraction
    # -----------------------------------------------------------------------

    def _extract_interaction_graph(
        self, inter_edges: set[tuple[int, int]], n_verts_a: int | None = None,
    ) -> np.ndarray:
        """Return (N_a, N_b) bool array of active inter-agent edges.

        Only original joint-to-joint edges are recorded; ground node edges are ignored.
        B joints start at index n_verts_a in the extended vertex layout.
        """
        ig = np.zeros((self.num_source_a, self.num_source_b), dtype=bool)
        n_a_orig = self.num_source_a
        n_a_ext = n_verts_a if n_verts_a is not None else n_a_orig
        n_b_orig = self.num_source_b
        for i, j in inter_edges:
            # Only original A joints (0..N_a-1) ↔ original B joints (n_a_ext..n_a_ext+N_b-1)
            if i < n_a_orig and n_a_ext <= j < n_a_ext + n_b_orig:
                ig[i, j - n_a_ext] = True
            elif j < n_a_orig and n_a_ext <= i < n_a_ext + n_b_orig:
                ig[j, i - n_a_ext] = True
        return ig

    def _extract_contact_graph(self, q: np.ndarray) -> np.ndarray:
        """Return (N_a_links, N_b_links) bool array of A–B physical contacts.

        Uses the same MuJoCo collision detection as the constraint builder but
        here we only record contact state (no Jacobians).
        """
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        n_a = len(self.laplacian_match_links_a)
        n_b = len(self.laplacian_match_links_b)
        cg = np.zeros((n_a, n_b), dtype=bool)

        # Map body name → column index in each agent's link dict
        a_body_names = list(self.laplacian_match_links_a.keys())
        b_body_names = list(self.laplacian_match_links_b.keys())
        a_body_idx = {n: i for i, n in enumerate(a_body_names)}
        b_body_idx = {n: i for i, n in enumerate(b_body_names)}

        # Collect geom-body mappings for quick lookup
        geom_body = {
            mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "": int(
                self.robot_model.geom_bodyid[g]
            )
            for g in range(self.robot_model.ngeom)
        }
        body_name_map = {
            int(self.robot_data.xpos[bid].tobytes().__hash__()): mujoco.mj_id2name(  # noqa: SIM118
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, bid
            )
            for bid in range(self.robot_model.nbody)
        }
        # Simpler: build body_id → body_name
        bid_to_name: dict[int, str] = {}
        for bid in range(self.robot_model.nbody):
            bname = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            bid_to_name[bid] = bname

        # Use collision detection at contact threshold
        candidates = self._prefilter_pairs_with_mj_collision(
            self.collision_detection_threshold
        )
        geom_names = [
            mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            for g in range(self.robot_model.ngeom)
        ]
        fromto = np.zeros(6, dtype=float)
        for g1, g2 in candidates:
            name1 = geom_names[g1]
            name2 = geom_names[g2]
            owner1 = self._owner_from_geom(g1, name1)
            owner2 = self._owner_from_geom(g2, name2)
            if not self._is_ab_pair(owner1, owner2):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(
                self.robot_model, self.robot_data,
                int(g1), int(g2),
                float(self.collision_detection_threshold), fromto,
            )
            if dist > self.collision_detection_threshold:
                continue

            # Determine which body each geom belongs to
            bid1 = int(self.robot_model.geom_bodyid[g1])
            bid2 = int(self.robot_model.geom_bodyid[g2])
            bname1 = bid_to_name.get(bid1, "")
            bname2 = bid_to_name.get(bid2, "")

            # Strip agent prefix to get base body name, match to source joint
            base1 = self._strip_agent_prefix(bname1)
            base2 = self._strip_agent_prefix(bname2)

            # Figure out which is A and which is B
            if owner1 == "A" and owner2 == "B":
                a_body, b_body = base1, base2
            else:
                a_body, b_body = base2, base1

            ai = a_body_idx.get(a_body)
            bi = b_body_idx.get(b_body)
            if ai is not None and bi is not None:
                cg[ai, bi] = True

        return cg

    # -----------------------------------------------------------------------
    # Foot constraints
    # -----------------------------------------------------------------------

    def _build_foot_constraints(
        self,
        q_curr: np.ndarray,
        q_t_last: np.ndarray,
        dqi: cp.Expression,
        flags: dict[str, bool],
        *,
        agent: str,
    ) -> list[cp.Constraint]:
        if (agent == "A" and self.q_a_init_idx >= 12) or (
            agent == "B" and self.q_b_init_idx >= 12
        ):
            return []

        links = self.foot_links_a if agent == "A" else self.foot_links_b
        j_now, p_now = self._calc_manipulator_jacobians(q_curr, links)
        _, p_prev = self._calc_manipulator_jacobians(q_t_last, links)

        left_flag, right_flag = self._extract_lr_flags(flags)
        idx = self.q_a_indices if agent == "A" else self.q_b_indices
        constraints: list[cp.Constraint] = []

        for link_name, j_full in j_now.items():
            lnk = link_name.lower()
            apply = ("left" in lnk and left_flag) or ("right" in lnk and right_flag)
            if not apply:
                continue
            p_lb = p_prev[link_name] - p_now[link_name] - self.foot_sticking_tolerance
            p_ub = p_prev[link_name] - p_now[link_name] + self.foot_sticking_tolerance
            jxy = j_full[:2, idx]
            constraints += [jxy @ dqi >= p_lb[:2], jxy @ dqi <= p_ub[:2]]
        return constraints

    def _build_stance_grounding_constraints(
        self,
        q_curr: np.ndarray,
        dqi: cp.Expression,
        source_foot_phase: dict[str, str] | None,
        *,
        agent: str,
    ) -> tuple[list[cp.Constraint], list[cp.Variable]]:
        if source_foot_phase is None:
            return [], []

        support_points = (
            self.foot_support_points_a if agent == "A" else self.foot_support_points_b
        )
        if not support_points:
            return [], []

        support_targets = {pt.body_name: pt.body_name for pt in support_points}
        j_now, p_now = self._calc_manipulator_jacobians(q_curr, support_targets)
        idx = self.q_a_indices if agent == "A" else self.q_b_indices

        constraints: list[cp.Constraint] = []
        slacks: list[cp.Variable] = []
        for point in support_points:
            phase = source_foot_phase.get(point.side, "swing")
            if not self._support_point_active_for_phase(point, phase):
                continue
            if point.body_name not in j_now:
                continue
            jz = j_now[point.body_name][2, idx]
            if np.allclose(jz, 0.0):
                continue
            target_z = point.clearance + self.foot_ground_tolerance
            foot_z_lin = float(p_now[point.body_name][2]) + jz @ dqi
            slack = cp.Variable(nonneg=True, name=f"{agent}_{point.side}_stance_slack")
            constraints.append(foot_z_lin <= target_z + slack)
            slacks.append(slack)

        return constraints, slacks

    @staticmethod
    def _support_point_active_for_phase(
        point: _FootSupportPoint,
        phase: str,
    ) -> bool:
        if phase == "flat":
            return True
        if phase == "toe":
            return point.x_local >= -1e-6
        if phase == "heel":
            return point.x_local <= 1e-6
        return False

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

    # -----------------------------------------------------------------------
    # Contact constraints (ground + A–B)
    # -----------------------------------------------------------------------

    def _build_ground_constraints(
        self, rows: list[_ContactLinearization], dx: cp.Expression
    ) -> list[cp.Constraint]:
        return [
            row.j_dx @ dx >= (-row.phi - self.penetration_tolerance)
            for row in rows
            if self._is_ground_pair(row.owner1, row.owner2)
        ]

    def _build_ab_constraints(
        self,
        rows: list[_ContactLinearization],
        dx: cp.Expression,
    ) -> tuple[list[cp.Constraint], cp.Variable | None]:
        ab_rows = sorted(
            [r for r in rows if self._is_ab_pair(r.owner1, r.owner2)],
            key=lambda r: (r.whitelisted, r.phi),
        )[: max(0, self.ab_top_k_pairs)]

        if not ab_rows:
            return [], None

        n_slack = sum(1 for r in ab_rows if r.whitelisted)
        slack_vars = (
            cp.Variable(n_slack, nonneg=True, name="ab_slack") if n_slack > 0 else None
        )

        constraints: list[cp.Constraint] = []
        s_idx = 0
        for row in ab_rows:
            if row.whitelisted:
                assert slack_vars is not None
                s = slack_vars[s_idx]
                s_idx += 1
                constraints.append(
                    row.j_dx @ dx >= (self.whitelist_margin - row.phi - s)
                )
            else:
                constraints.append(
                    row.j_dx @ dx >= (-row.phi - self.penetration_tolerance)
                )
        return constraints, slack_vars

    def _linearize_contact_rows(self, q: np.ndarray) -> list[_ContactLinearization]:
        self.robot_data.qpos[:] = q
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
            if not (self._is_ground_pair(owner1, owner2) or self._is_ab_pair(owner1, owner2)):
                continue

            fromto[:] = 0.0
            dist = mujoco.mj_geomDistance(
                self.robot_model, self.robot_data,
                int(g1), int(g2),
                float(self.collision_detection_threshold), fromto,
            )
            if dist > self.collision_detection_threshold:
                continue

            j_full = self._compute_jacobian_for_contact_relative(
                self.robot_model.geom(g1),
                self.robot_model.geom(g2),
                name1, name2, fromto, dist,
            )
            j_dx = np.concatenate(
                [j_full[self.q_a_indices], j_full[self.q_b_indices]], axis=0
            )
            rows.append(
                _ContactLinearization(
                    phi=float(dist),
                    j_dx=j_dx,
                    geom1_name=name1,
                    geom2_name=name2,
                    owner1=owner1,
                    owner2=owner2,
                    whitelisted=self._is_whitelisted_pair(name1, name2),
                )
            )
        return rows

    def _prefilter_pairs_with_mj_collision(
        self, threshold: float
    ) -> set[tuple[int, int]]:
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
        if self.robot_a_prefix and name.startswith(self.robot_a_prefix):
            return "A"
        if self.robot_b_prefix and name.startswith(self.robot_b_prefix):
            return "B"
        return "O"

    def _owner_from_geom(self, geom_id: int, geom_name: str) -> str:
        owner = self._owner_from_prefixed_name(geom_name)
        if owner != "O":
            return owner

        body_id = int(self.robot_model.geom_bodyid[int(geom_id)])
        body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        owner = self._owner_from_prefixed_name(body_name)
        if owner != "O":
            return owner

        if int(self.robot_model.geom_type[int(geom_id)]) == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(self.robot_model.geom_dataid[int(geom_id)])
            if mesh_id >= 0:
                mesh_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
                owner = self._owner_from_prefixed_name(mesh_name)
                if owner != "O":
                    return owner

        return "O"

    @staticmethod
    def _is_ground_pair(o1: str, o2: str) -> bool:
        return (o1 == "G" and o2 in {"A", "B"}) or (o2 == "G" and o1 in {"A", "B"})

    @staticmethod
    def _is_ab_pair(o1: str, o2: str) -> bool:
        return (o1 == "A" and o2 == "B") or (o1 == "B" and o2 == "A")

    def _is_whitelisted_pair(self, geom1: str, geom2: str) -> bool:
        if not self.ab_pair_whitelist:
            return False
        pair_exact = tuple(sorted((geom1, geom2)))
        if pair_exact in self.ab_pair_whitelist:
            return True
        base1 = self._strip_agent_prefix(geom1)
        base2 = self._strip_agent_prefix(geom2)
        return tuple(sorted((base1, base2))) in self.ab_pair_whitelist

    def _strip_agent_prefix(self, name: str) -> str:
        if self.robot_a_prefix and name.startswith(self.robot_a_prefix):
            return name[len(self.robot_a_prefix):]
        if self.robot_b_prefix and name.startswith(self.robot_b_prefix):
            return name[len(self.robot_b_prefix):]
        return name

    # -----------------------------------------------------------------------
    # Kinematics / Jacobians
    # -----------------------------------------------------------------------

    def _build_target_jacobian_and_vertices(
        self,
        q: np.ndarray,
        source_vertices: np.ndarray,
        n_verts_a: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Jacobians and FK positions for all source joints.

        Ground nodes (if present) keep their z=0 positions and get zero Jacobians,
        which is correct: they are fixed reference points, not robot links.

        Args:
            n_verts_a: extended A vertex count (N_a + G); B joints start here.
                       Defaults to self.num_source_a.

        Returns:
            J_target:       (3 * N_total, nq_ab)  — full stacked Jacobian in dx space.
            target_vertices:(N_total, 3) — FK robot link positions (fallback to
                            source_vertices for unmapped joints and ground nodes).
        """
        ja_dict, pa_dict = self._calc_manipulator_jacobians(q, self.laplacian_match_links_a)
        jb_dict, pb_dict = self._calc_manipulator_jacobians(q, self.laplacian_match_links_b)

        n_v = source_vertices.shape[0]  # total including ground nodes
        b_offset = n_verts_a if n_verts_a is not None else self.num_source_a

        target_vertices = source_vertices.copy()
        J_target = np.zeros((3 * n_v, self.nq_ab), dtype=float)

        for i, name in enumerate(self.source_joint_names_a):
            row = slice(3 * i, 3 * (i + 1))
            if name in pa_dict:
                target_vertices[i] = pa_dict[name]
                J_target[row, :] = self._project_j_to_dx(ja_dict[name])

        # B joints start at b_offset (after A joints + A ground nodes)
        for i, name in enumerate(self.source_joint_names_b):
            idx = b_offset + i
            row = slice(3 * idx, 3 * (idx + 1))
            if name in pb_dict:
                target_vertices[idx] = pb_dict[name]
                J_target[row, :] = self._project_j_to_dx(jb_dict[name])

        return J_target, target_vertices

    def _project_j_to_dx(self, j_full: np.ndarray) -> np.ndarray:
        """Project full-nq Jacobian into the [dqa; dqb] decision variable space."""
        return np.hstack([j_full[:, self.q_a_indices], j_full[:, self.q_b_indices]])

    def _compute_jacobian_for_contact_relative(
        self, geom1, geom2, geom1_name, geom2_name, fromto, dist
    ) -> np.ndarray:
        pos1, pos2 = fromto[:3], fromto[3:]
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
        links: dict[str, str | tuple[str, np.ndarray] | _BodyPointTarget],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        J_dict: dict[str, np.ndarray] = {}
        p_dict: dict[str, np.ndarray] = {}
        for key, target in links.items():
            body_name, point_offset = self._resolve_body_point_target(target)
            body_id = mujoco.mj_name2id(
                self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id == -1:
                continue
            J = self._calc_contact_jacobian_from_point(body_id, point_offset)
            R_WB = self.robot_data.xmat[body_id].reshape(3, 3)
            pos = self.robot_data.xpos[body_id] + R_WB @ point_offset
            J_dict[key] = np.asarray(J, dtype=float, copy=True)
            p_dict[key] = np.asarray(pos, dtype=float, copy=True)
        return J_dict, p_dict

    def _calc_contact_jacobian_from_point(
        self, body_idx: int, p_body: np.ndarray, input_world: bool = False
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
        """Return T (nv × nq) such that v = T(q) @ qdot."""
        nq = self.robot_model.nq
        nv = self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        def _e_world(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
            return np.array([
                [-qx,  qw,  qz, -qy],
                [-qy, -qz,  qw,  qx],
                [-qz,  qy, -qx,  qw],
            ])

        for j in range(self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            qadr = int(self.robot_model.jnt_qposadr[j])
            dadr = int(self.robot_model.jnt_dofadr[j])
            if jt == mujoco.mjtJoint.mjJNT_FREE:
                T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)
                qw, qx, qy, qz = self.robot_data.qpos[qadr + 3 : qadr + 7]
                T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = 2.0 * _e_world(
                    qw, qx, qy, qz
                )
            elif jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                T[dadr, qadr] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint not supported in IAMRRetargeter.")
        return T

    # -----------------------------------------------------------------------
    # Local robot state I/O
    # -----------------------------------------------------------------------

    def _write_local_robot_state(
        self, q_full: np.ndarray, q_local: np.ndarray, *, agent: str
    ) -> None:
        block = self.q_block_a if agent == "A" else self.q_block_b
        dof = self.robot_dof_a if agent == "A" else self.robot_dof_b
        min_width = 7 + dof
        if q_local.shape[0] < min_width:
            raise ValueError(
                f"Agent {agent}: expected local state length >= {min_width}, "
                f"got {q_local.shape[0]}"
            )
        q_full[block[:min_width]] = q_local[:min_width]

    def _extract_local_robot_state(
        self, q_full: np.ndarray, *, agent: str
    ) -> np.ndarray:
        block = self.q_block_a if agent == "A" else self.q_block_b
        dof = self.robot_dof_a if agent == "A" else self.robot_dof_b
        return np.asarray(q_full[block[: 7 + dof]], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalise_foot_seq(
        seq: list[dict[str, bool]] | None, n_frames: int
    ) -> list[dict[str, bool]]:
        if seq is None:
            return [{} for _ in range(n_frames)]
        if len(seq) != n_frames:
            raise ValueError(
                f"Foot-sticking sequence length {len(seq)} != n_frames {n_frames}"
            )
        return seq

    def _normalise_nominal_seq(
        self,
        seq: np.ndarray | None,
        n_frames: int,
        agent: str,
    ) -> np.ndarray | None:
        if seq is None:
            return None
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != n_frames:
            raise ValueError(
                f"Nominal sequence for agent {agent} must be (T={n_frames}, D), "
                f"got {arr.shape}"
            )
        nq_opt = self.nq_a if agent == "A" else self.nq_b
        dof = self.robot_dof_a if agent == "A" else self.robot_dof_b
        start = self.start_local_a if agent == "A" else self.start_local_b
        if arr.shape[1] == nq_opt:
            return arr
        if arr.shape[1] >= 7 + dof:
            return arr[:, start : start + nq_opt]
        raise ValueError(
            f"Nominal sequence width for agent {agent} must be {nq_opt} (opt slice) "
            f"or >= {7 + dof} (full local state), got {arr.shape[1]}"
        )

    @staticmethod
    def _project_nominal_indices(
        indices: np.ndarray, start_local: int, nq_local: int
    ) -> np.ndarray:
        if indices.size == 0:
            return np.array([], dtype=int)
        mask = (indices >= start_local) & (indices < start_local + nq_local)
        return (indices[mask] - start_local).astype(int)
