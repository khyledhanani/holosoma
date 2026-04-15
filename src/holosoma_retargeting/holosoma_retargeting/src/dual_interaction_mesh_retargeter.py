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
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from utils import (  # type: ignore[import-not-found,no-redef]  # noqa: E402
    calculate_laplacian_coordinates,
    calculate_laplacian_matrix,
)


@dataclass(frozen=True)
class ContactLinearization:
    phi: float
    j_dx: np.ndarray
    geom1_name: str
    geom2_name: str
    owner1: str
    owner2: str
    whitelisted: bool


@dataclass(frozen=True)
class BodyPointTarget:
    """A point target attached to a MuJoCo body in local-body coordinates."""

    body_name: str
    point_offset: np.ndarray


class DualInteractionMeshRetargeter:
    """Dual-robot SQP retargeter core for synchronized A/B retargeting.

    This class follows the architecture of InteractionMeshRetargeter but:
    - removes object-tail assumptions,
    - uses a single dual decision variable dx = [dqa, dqb],
    - adds A-B collision constraints with optional whitelist slack.
    """

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
        w_nominal_tracking_init: float = 0.0,
        nominal_tracking_tau: float = 1e6,
        max_source_edge_len: float = 0.45,
        cross_agent_rescue_k: int = 3,
        ab_pair_whitelist: set[tuple[str, str]] | None = None,
        debug: bool = False,
    ):
        self.task_constants_a = task_constants_a
        self.task_constants_b = task_constants_b if task_constants_b is not None else task_constants_a
        self.debug = debug

        self.robot_a_prefix = robot_a_prefix
        self.robot_b_prefix = robot_b_prefix

        self.q_a_init_idx = q_a_init_idx
        self.q_b_init_idx = q_b_init_idx
        self.activate_foot_sticking = activate_foot_sticking
        self.activate_joint_limits = activate_joint_limits

        self.step_size = step_size
        self.n_iter_first = n_iter_first
        self.n_iter_other = n_iter_other
        self.collision_detection_threshold = collision_detection_threshold
        self.penetration_tolerance = penetration_tolerance
        self.foot_sticking_tolerance = foot_sticking_tolerance

        self.whitelist_margin = whitelist_margin
        self.ab_top_k_pairs = ab_top_k_pairs
        self.contact_slack_weight = contact_slack_weight
        self.w_nominal_tracking_init = float(w_nominal_tracking_init)
        self.nominal_tracking_tau = float(nominal_tracking_tau)

        self.max_source_edge_len = max_source_edge_len
        self.cross_agent_rescue_k = cross_agent_rescue_k
        self.ab_pair_whitelist = ab_pair_whitelist or set()

        self.robot_model = mujoco.MjModel.from_xml_path(dual_scene_xml_path)
        self.robot_data = mujoco.MjData(self.robot_model)

        self.demo_joints_a = list(self.task_constants_a.DEMO_JOINTS)
        self.demo_joints_b = list(self.task_constants_b.DEMO_JOINTS)
        self.base_joint_to_link_a = dict(self.task_constants_a.JOINTS_MAPPING)
        self.base_joint_to_link_b = dict(self.task_constants_b.JOINTS_MAPPING)

        # Use all source joints that have an explicit target mapping.
        self.source_joint_names_a = [n for n in self.demo_joints_a if n in self.base_joint_to_link_a]
        self.source_joint_names_b = [n for n in self.demo_joints_b if n in self.base_joint_to_link_b]
        if not self.source_joint_names_a:
            raise ValueError("No mapped source joints found for A. Check task_constants_a.JOINTS_MAPPING.")
        if not self.source_joint_names_b:
            raise ValueError("No mapped source joints found for B. Check task_constants_b.JOINTS_MAPPING.")
        self.num_source_a = len(self.source_joint_names_a)
        self.num_source_b = len(self.source_joint_names_b)
        self.num_source_total = self.num_source_a + self.num_source_b

        self.base_foot_links_a = list(self.task_constants_a.FOOT_STICKING_LINKS)
        self.base_foot_links_b = list(self.task_constants_b.FOOT_STICKING_LINKS)
        self.robot_dof_expected_a = int(self.task_constants_a.ROBOT_DOF)
        self.robot_dof_expected_b = int(self.task_constants_b.ROBOT_DOF)

        self._setup_joint_layouts()
        self._setup_mapped_links()
        self._setup_bounds_and_costs()
        self._setup_source_graph_indices()

    # --------------------------------------------------------------------------
    # Setup helpers
    # --------------------------------------------------------------------------
    def _setup_joint_layouts(self) -> None:
        free_joint_ids = [
            j for j in range(self.robot_model.njnt) if self.robot_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]
        if len(free_joint_ids) < 2:
            raise ValueError(
                "DualInteractionMeshRetargeter expects a dual scene with at least two free joints (robot A and robot B)."
            )

        free_joint_ids_sorted = sorted(free_joint_ids, key=lambda j: int(self.robot_model.jnt_qposadr[j]))

        joint_id_a: int | None = None
        joint_id_b: int | None = None
        for j in free_joint_ids_sorted:
            body_id = int(self.robot_model.jnt_bodyid[j])
            body_name = mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
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

        qadr_all_free = sorted(int(self.robot_model.jnt_qposadr[j]) for j in free_joint_ids)

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
                f"Robot A DOF mismatch between constants ({self.robot_dof_expected_a}) and scene ({self.robot_dof_a})."
            )
        if self.robot_dof_b != self.robot_dof_expected_b:
            raise ValueError(
                f"Robot B DOF mismatch between constants ({self.robot_dof_expected_b}) and scene ({self.robot_dof_b})."
            )

        self.base_quat_slice_a = slice(self.qadr_a + 3, self.qadr_a + 7)
        self.base_quat_slice_b = slice(self.qadr_b + 3, self.qadr_b + 7)

        start_local_a = 7 + self.q_a_init_idx
        start_local_b = 7 + self.q_b_init_idx
        if start_local_a < 0 or start_local_b < 0 or start_local_a >= len(self.q_block_a) or start_local_b >= len(self.q_block_b):
            raise ValueError("q_*_init_idx must be >= -7.")
        self.start_local_a = int(start_local_a)
        self.start_local_b = int(start_local_b)

        self.q_a_indices = self.q_block_a[start_local_a:]
        self.q_b_indices = self.q_block_b[start_local_b:]

        self.nq_a = len(self.q_a_indices)
        self.nq_b = len(self.q_b_indices)
        self.nq_ab = self.nq_a + self.nq_b

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
            f"{self.robot_a_prefix}{link_name}": f"{self.robot_a_prefix}{link_name}" for link_name in self.base_foot_links_a
        }
        self.foot_links_b = {
            f"{self.robot_b_prefix}{link_name}": f"{self.robot_b_prefix}{link_name}" for link_name in self.base_foot_links_b
        }

    @staticmethod
    def _prefix_body_point_target(
        target: str | tuple[str, np.ndarray] | BodyPointTarget,
        prefix: str,
    ) -> str | BodyPointTarget:
        if isinstance(target, BodyPointTarget):
            return BodyPointTarget(
                body_name=f"{prefix}{target.body_name}",
                point_offset=np.asarray(target.point_offset, dtype=float).reshape(3),
            )
        if isinstance(target, tuple):
            body_name, point_offset = target
            return BodyPointTarget(
                body_name=f"{prefix}{body_name}",
                point_offset=np.asarray(point_offset, dtype=float).reshape(3),
            )
        return f"{prefix}{target}"

    @staticmethod
    def _resolve_body_point_target(
        target: str | tuple[str, np.ndarray] | BodyPointTarget,
    ) -> tuple[str, np.ndarray]:
        if isinstance(target, BodyPointTarget):
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

        manual_lb_a = getattr(self.task_constants_a, "MANUAL_LB", {})
        manual_ub_a = getattr(self.task_constants_a, "MANUAL_UB", {})
        manual_cost_a = getattr(self.task_constants_a, "MANUAL_COST", {})
        manual_lb_b = getattr(self.task_constants_b, "MANUAL_LB", {})
        manual_ub_b = getattr(self.task_constants_b, "MANUAL_UB", {})
        manual_cost_b = getattr(self.task_constants_b, "MANUAL_COST", {})

        for k, v in manual_lb_a.items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.q_a_lb[idx] = float(v)
        for k, v in manual_lb_b.items():
            idx = int(k)
            if 0 <= idx < self.nq_b:
                self.q_b_lb[idx] = float(v)
        for k, v in manual_ub_a.items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.q_a_ub[idx] = float(v)
        for k, v in manual_ub_b.items():
            idx = int(k)
            if 0 <= idx < self.nq_b:
                self.q_b_ub[idx] = float(v)

        self.Q_diag_a = np.zeros(self.nq_a, dtype=float)
        self.Q_diag_b = np.zeros(self.nq_b, dtype=float)
        for k, v in manual_cost_a.items():
            idx = int(k)
            if 0 <= idx < self.nq_a:
                self.Q_diag_a[idx] = float(v)
        for k, v in manual_cost_b.items():
            idx = int(k)
            if 0 <= idx < self.nq_b:
                self.Q_diag_b[idx] = float(v)

    def _setup_source_graph_indices(self) -> None:
        self.source_name_to_idx_a = {n: i for i, n in enumerate(self.source_joint_names_a)}
        self.source_name_to_idx_b = {n: i for i, n in enumerate(self.source_joint_names_b)}

        cross_critical = {
            "L_Wrist",
            "R_Wrist",
            "L_Elbow",
            "R_Elbow",
            "L_Shoulder",
            "R_Shoulder",
        }
        self.cross_rescue_a = [self.source_name_to_idx_a[n] for n in cross_critical if n in self.source_name_to_idx_a]
        self.cross_rescue_b = [
            self.num_source_a + self.source_name_to_idx_b[n] for n in cross_critical if n in self.source_name_to_idx_b
        ]

        critical_weights = {
            "Pelvis",
            "Spine",
            "Spine1",
            "Spine2",
            "Spine3",
            "Chest",
            "L_Wrist",
            "R_Wrist",
            "L_Elbow",
            "R_Elbow",
            "L_Shoulder",
            "R_Shoulder",
        }
        self.laplacian_weights = np.ones(self.num_source_total, dtype=float)
        for i, name in enumerate(self.source_joint_names_a):
            if name in critical_weights:
                self.laplacian_weights[i] = 2.5
        offset_b = self.num_source_a
        for i, name in enumerate(self.source_joint_names_b):
            if name in critical_weights:
                self.laplacian_weights[offset_b + i] = 2.5

        nominal_indices_a = np.asarray(getattr(self.task_constants_a, "NOMINAL_TRACKING_INDICES", []), dtype=int)
        nominal_indices_b = np.asarray(getattr(self.task_constants_b, "NOMINAL_TRACKING_INDICES", []), dtype=int)
        self.track_nominal_indices_a = self._project_nominal_indices(nominal_indices_a, self.start_local_a, self.nq_a)
        self.track_nominal_indices_b = self._project_nominal_indices(nominal_indices_b, self.start_local_b, self.nq_b)

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------
    def retarget_motion(
        self,
        human_joint_motions_a: np.ndarray,
        human_joint_motions_b: np.ndarray,
        foot_sticking_sequences_a: list[dict[str, bool]] | None = None,
        foot_sticking_sequences_b: list[dict[str, bool]] | None = None,
        q_nominal_motions_a: np.ndarray | None = None,
        q_nominal_motions_b: np.ndarray | None = None,
        q_init_a: np.ndarray | None = None,
        q_init_b: np.ndarray | None = None,
        dest_res_path: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retarget a dual sequence with synchronized per-frame SQP solves.

        Args:
            human_joint_motions_a: (T, Ja, 3) source joints for person A.
            human_joint_motions_b: (T, Jb, 3) source joints for person B.
            foot_sticking_sequences_a: length-T flags for A.
            foot_sticking_sequences_b: length-T flags for B.
            q_nominal_motions_a: optional per-frame nominal states for A.
            q_nominal_motions_b: optional per-frame nominal states for B.
            q_init_a: optional local robot state for A in MuJoCo order [xyz, wxyz, joints].
            q_init_b: optional local robot state for B in MuJoCo order [xyz, wxyz, joints].
            dest_res_path: optional output npz path.
        """
        if human_joint_motions_a.ndim != 3 or human_joint_motions_a.shape[2] != 3:
            raise ValueError(f"Expected (T, J, 3), got {human_joint_motions_a.shape}")
        if human_joint_motions_b.ndim != 3 or human_joint_motions_b.shape[2] != 3:
            raise ValueError(f"Expected (T, J, 3), got {human_joint_motions_b.shape}")
        if human_joint_motions_a.shape[0] != human_joint_motions_b.shape[0]:
            raise ValueError(
                f"A/B frame count mismatch: {human_joint_motions_a.shape[0]} vs {human_joint_motions_b.shape[0]}"
            )
        if human_joint_motions_a.shape[1] < self.num_source_a:
            raise ValueError(
                f"Expected at least {self.num_source_a} source joints for A, got {human_joint_motions_a.shape[1]}"
            )
        if human_joint_motions_b.shape[1] < self.num_source_b:
            raise ValueError(
                f"Expected at least {self.num_source_b} source joints for B, got {human_joint_motions_b.shape[1]}"
            )

        num_frames = int(human_joint_motions_a.shape[0])

        foot_sticking_sequences_a = self._normalize_foot_sequence(foot_sticking_sequences_a, num_frames)
        foot_sticking_sequences_b = self._normalize_foot_sequence(foot_sticking_sequences_b, num_frames)
        q_nominal_motions_a = self._normalize_nominal_sequence(q_nominal_motions_a, num_frames, agent="A")
        q_nominal_motions_b = self._normalize_nominal_sequence(q_nominal_motions_b, num_frames, agent="B")

        q = np.copy(self.robot_data.qpos)
        if q_init_a is not None:
            self._write_local_robot_state(q, q_init_a, agent="A")
        if q_init_b is not None:
            self._write_local_robot_state(q, q_init_b, agent="B")

        q_prev_frame = q.copy()
        q_history: list[np.ndarray] = []
        qpos_a_list: list[np.ndarray] = []
        qpos_b_list: list[np.ndarray] = []

        with tqdm(range(num_frames)) as pbar:
            for t in pbar:
                source_vertices = np.vstack(
                    [
                        human_joint_motions_a[t, : self.num_source_a, :],
                        human_joint_motions_b[t, : self.num_source_b, :],
                    ]
                )
                _, adj_list, target_laplacian = self._build_source_graph(source_vertices)
                n_iter = self.n_iter_first if t == 0 else self.n_iter_other
                w_nominal_tracking = self._nominal_weight_for_frame(t)

                q, cost = self.iterate(
                    q_n=q,
                    q_t_last=q_prev_frame,
                    source_vertices=source_vertices,
                    adj_list=adj_list,
                    target_laplacian=target_laplacian,
                    foot_sticking_a=foot_sticking_sequences_a[t],
                    foot_sticking_b=foot_sticking_sequences_b[t],
                    q_nominal_a=(q_nominal_motions_a[t] if q_nominal_motions_a is not None else None),
                    q_nominal_b=(q_nominal_motions_b[t] if q_nominal_motions_b is not None else None),
                    w_nominal_tracking=w_nominal_tracking,
                    n_iter=n_iter,
                    init_t=(t == 0),
                )

                q_history.append(q.copy())
                qpos_a_list.append(self._extract_local_robot_state(q, agent="A"))
                qpos_b_list.append(self._extract_local_robot_state(q, agent="B"))
                q_prev_frame = q.copy()
                pbar.set_postfix(cost=float(cost))

        qpos_a = np.asarray(qpos_a_list, dtype=np.float32)
        qpos_b = np.asarray(qpos_b_list, dtype=np.float32)
        if dest_res_path is not None:
            np.savez(
                dest_res_path,
                qpos=np.asarray(q_history, dtype=np.float32),
                qpos_A=qpos_a,
                qpos_B=qpos_b,
                human_joints_A=human_joint_motions_a,
                human_joints_B=human_joint_motions_b,
            )
        return qpos_a, qpos_b

    # --------------------------------------------------------------------------
    # SQP loop
    # --------------------------------------------------------------------------
    def iterate(
        self,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        source_vertices: np.ndarray,
        adj_list: list[list[int]],
        target_laplacian: np.ndarray,
        foot_sticking_a: dict[str, bool],
        foot_sticking_b: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        q_nominal_b: np.ndarray | None,
        w_nominal_tracking: float,
        *,
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
                adj_list=adj_list,
                target_laplacian=target_laplacian,
                foot_sticking_a=foot_sticking_a,
                foot_sticking_b=foot_sticking_b,
                q_nominal_a=q_nominal_a,
                q_nominal_b=q_nominal_b,
                w_nominal_tracking=w_nominal_tracking,
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
        adj_list: list[list[int]],
        target_laplacian: np.ndarray,
        foot_sticking_a: dict[str, bool],
        foot_sticking_b: dict[str, bool],
        q_nominal_a: np.ndarray | None,
        q_nominal_b: np.ndarray | None,
        w_nominal_tracking: float,
        init_t: bool,
        verbose: bool = False,
    ) -> tuple[np.ndarray, float]:
        J_target, target_vertices = self._build_target_jacobian_and_vertices(q_curr, source_vertices)

        L = calculate_laplacian_matrix(target_vertices, adj_list)
        if not sp.issparse(L):
            L = sp.csr_matrix(L)
        L = L.tocsr()

        kron = sp.kron(L, sp.eye(3, format="csr"), format="csr")
        J_lap = kron @ J_target
        lap0 = (L @ target_vertices).reshape(-1)
        target_lap_vec = target_laplacian.reshape(-1)

        dx = cp.Variable(self.nq_ab, name="dx")
        dqa = dx[: self.nq_a]
        dqb = dx[self.nq_a :]
        lap_var = cp.Variable(3 * self.num_source_total, name="laplacian")

        constraints: list[cp.Constraint] = []
        constraints.append(cp.Constant(J_lap) @ dx - lap_var == -lap0)

        # Foot sticking constraints per robot (XY only).
        if self.activate_foot_sticking:
            constraints.extend(self._build_foot_constraints(q_curr, q_t_last, dqa, foot_sticking_a, agent="A"))
            constraints.extend(self._build_foot_constraints(q_curr, q_t_last, dqb, foot_sticking_b, agent="B"))

        # Contact constraints: A-ground, B-ground, and A-B (with optional whitelist slack).
        contact_rows = self._linearize_contact_rows(q_curr)
        constraints.extend(self._build_ground_constraints(contact_rows, dx))
        ab_constraints, slack_vars = self._build_ab_constraints(contact_rows, dx)
        constraints.extend(ab_constraints)

        # Joint limits.
        if self.activate_joint_limits:
            q_a_n = q_curr[self.q_a_indices]
            q_b_n = q_curr[self.q_b_indices]
            constraints.extend([dqa >= (self.q_a_lb - q_a_n), dqa <= (self.q_a_ub - q_a_n)])
            constraints.extend([dqb >= (self.q_b_lb - q_b_n), dqb <= (self.q_b_ub - q_b_n)])

        constraints.append(cp.SOC(self.step_size, dx))

        # Objective.
        sqrt_w = np.sqrt(np.repeat(self.laplacian_weights, 3))
        obj_terms: list[cp.Expression] = [
            cp.sum_squares(cp.multiply(sqrt_w, lap_var - target_lap_vec)),
        ]

        q_a_n = q_curr[self.q_a_indices]
        q_b_n = q_curr[self.q_b_indices]
        if (w_nominal_tracking > 0.0) and (q_nominal_a is not None) and (self.track_nominal_indices_a.size > 0):
            idx = self.track_nominal_indices_a
            za = dqa[idx] - (q_nominal_a[idx] - q_a_n[idx])
            obj_terms.append(w_nominal_tracking * cp.sum_squares(za))
        if (w_nominal_tracking > 0.0) and (q_nominal_b is not None) and (self.track_nominal_indices_b.size > 0):
            idx = self.track_nominal_indices_b
            zb = dqb[idx] - (q_nominal_b[idx] - q_b_n[idx])
            obj_terms.append(w_nominal_tracking * cp.sum_squares(zb))

        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(self.Q_diag_a), dqa + q_a_n)))
        obj_terms.append(cp.sum_squares(cp.multiply(np.sqrt(self.Q_diag_b), dqb + q_b_n)))

        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n
        dqb_smooth = q_t_last[self.q_b_indices] - q_b_n
        obj_terms.append(0.2 * cp.sum_squares(dqa - dqa_smooth))
        obj_terms.append(0.2 * cp.sum_squares(dqb - dqb_smooth))

        if slack_vars is not None and slack_vars.size > 0:
            obj_terms.append(self.contact_slack_weight * cp.sum_squares(slack_vars))

        problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints)
        problem.solve(solver=cp.CLARABEL, verbose=verbose)
        if (problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)) and init_t:
            constraints_wo_soc = [c for c in constraints if not isinstance(c, cp.constraints.second_order.SOC)]
            problem = cp.Problem(cp.Minimize(cp.sum(obj_terms)), constraints_wo_soc)
            problem.solve(solver=cp.CLARABEL, verbose=verbose)

        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            raise RuntimeError(f"Dual CVXPY solve failed: {problem.status}")

        dx_star = np.asarray(dx.value, dtype=float)
        q_next = q_curr.copy()
        q_next[self.q_a_indices] = q_curr[self.q_a_indices] + dx_star[: self.nq_a]
        q_next[self.q_b_indices] = q_curr[self.q_b_indices] + dx_star[self.nq_a :]
        q_next[self.base_quat_slice_a] /= np.linalg.norm(q_next[self.base_quat_slice_a]) + 1e-12
        q_next[self.base_quat_slice_b] /= np.linalg.norm(q_next[self.base_quat_slice_b]) + 1e-12
        return q_next, float(problem.value)

    # --------------------------------------------------------------------------
    # Source graph + Laplacian target
    # --------------------------------------------------------------------------
    def _build_source_graph(self, source_vertices: np.ndarray) -> tuple[np.ndarray, list[list[int]], np.ndarray]:
        tetrahedra = self._delaunay_tetrahedra(source_vertices)
        edges = self._tetrahedra_to_edges(tetrahedra)
        edges = self._prune_long_edges(source_vertices, edges, self.max_source_edge_len)
        edges = self._add_cross_agent_rescue_edges(source_vertices, edges)
        adj_list = self._edges_to_adj_list(edges, source_vertices.shape[0])
        target_laplacian = calculate_laplacian_coordinates(source_vertices, adj_list)
        return tetrahedra, adj_list, target_laplacian

    def _delaunay_tetrahedra(self, vertices: np.ndarray) -> np.ndarray:
        try:
            return Delaunay(vertices).simplices
        except Exception:
            # Degenerate fallback: nearest-neighbor pseudo-tetrahedra.
            n = vertices.shape[0]
            if n < 4:
                raise
            d = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)
            np.fill_diagonal(d, np.inf)
            simplices: list[list[int]] = []
            for i in range(n):
                nbrs = np.argsort(d[i])[:3]
                simplices.append([i, int(nbrs[0]), int(nbrs[1]), int(nbrs[2])])
            return np.asarray(simplices, dtype=int)

    @staticmethod
    def _tetrahedra_to_edges(tetrahedra: np.ndarray) -> set[tuple[int, int]]:
        edges: set[tuple[int, int]] = set()
        for tet in tetrahedra:
            a, b, c, d = [int(x) for x in tet]
            pairs = [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]
            for i, j in pairs:
                edges.add((i, j) if i < j else (j, i))
        return edges

    @staticmethod
    def _prune_long_edges(vertices: np.ndarray, edges: set[tuple[int, int]], max_len: float) -> set[tuple[int, int]]:
        kept: set[tuple[int, int]] = set()
        for i, j in edges:
            if np.linalg.norm(vertices[i] - vertices[j]) <= max_len:
                kept.add((i, j))
        return kept

    def _add_cross_agent_rescue_edges(self, vertices: np.ndarray, edges: set[tuple[int, int]]) -> set[tuple[int, int]]:
        n_a = self.num_source_a
        n_b = self.num_source_b
        a_ids = np.arange(0, n_a, dtype=int)
        b_ids = np.arange(n_a, n_a + n_b, dtype=int)
        dmat = np.linalg.norm(vertices[:, None, :] - vertices[None, :, :], axis=-1)

        def _cross_count(idx: int) -> int:
            count = 0
            for i, j in edges:
                if i == idx and j >= n_a:
                    count += 1
                elif j == idx and i >= n_a:
                    count += 1
            return count

        for idx in self.cross_rescue_a:
            if _cross_count(idx) > 0:
                continue
            nbrs = b_ids[np.argsort(dmat[idx, b_ids])[: self.cross_agent_rescue_k]]
            for j in nbrs:
                i0, j0 = (idx, int(j)) if idx < int(j) else (int(j), idx)
                edges.add((i0, j0))

        for idx in self.cross_rescue_b:
            if _cross_count(idx) > 0:
                continue
            nbrs = a_ids[np.argsort(dmat[idx, a_ids])[: self.cross_agent_rescue_k]]
            for j in nbrs:
                i0, j0 = (int(j), idx) if int(j) < idx else (idx, int(j))
                edges.add((i0, j0))
        return edges

    @staticmethod
    def _edges_to_adj_list(edges: set[tuple[int, int]], n_vertices: int) -> list[list[int]]:
        adj = [[] for _ in range(n_vertices)]
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        return adj

    # --------------------------------------------------------------------------
    # Target assembly
    # --------------------------------------------------------------------------
    def _build_target_jacobian_and_vertices(
        self,
        q: np.ndarray,
        source_vertices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ja_dict, pa_dict = self._calc_manipulator_jacobians(q, self.laplacian_match_links_a)
        jb_dict, pb_dict = self._calc_manipulator_jacobians(q, self.laplacian_match_links_b)

        n_v = self.num_source_total
        target_vertices = source_vertices.copy()
        J_target = np.zeros((3 * n_v, self.nq_ab), dtype=float)

        for i, joint_name in enumerate(self.source_joint_names_a):
            row = slice(3 * i, 3 * (i + 1))
            if joint_name in pa_dict:
                target_vertices[i] = pa_dict[joint_name]
                J_target[row, :] = self._project_j_full_to_dx(ja_dict[joint_name])

        offset = self.num_source_a
        for i, joint_name in enumerate(self.source_joint_names_b):
            idx = offset + i
            row = slice(3 * idx, 3 * (idx + 1))
            if joint_name in pb_dict:
                target_vertices[idx] = pb_dict[joint_name]
                J_target[row, :] = self._project_j_full_to_dx(jb_dict[joint_name])

        return J_target, target_vertices

    def _project_j_full_to_dx(self, j_full: np.ndarray) -> np.ndarray:
        return np.hstack([j_full[:, self.q_a_indices], j_full[:, self.q_b_indices]])

    # --------------------------------------------------------------------------
    # Foot constraints
    # --------------------------------------------------------------------------
    def _build_foot_constraints(
        self,
        q_curr: np.ndarray,
        q_t_last: np.ndarray,
        dqi: cp.Expression,
        flags: dict[str, bool],
        *,
        agent: str,
    ) -> list[cp.Constraint]:
        if (agent == "A" and self.q_a_init_idx >= 12) or (agent == "B" and self.q_b_init_idx >= 12):
            return []

        links = self.foot_links_a if agent == "A" else self.foot_links_b
        j_now, p_now = self._calc_manipulator_jacobians(q_curr, links)
        _, p_prev = self._calc_manipulator_jacobians(q_t_last, links)

        left_flag, right_flag = self._extract_lr_flags(flags)
        constraints: list[cp.Constraint] = []

        idx = self.q_a_indices if agent == "A" else self.q_b_indices
        for link_name, j_full in j_now.items():
            lnk = link_name.lower()
            apply_left = ("left" in lnk) and left_flag
            apply_right = ("right" in lnk) and right_flag
            if not (apply_left or apply_right):
                continue

            p_lb = p_prev[link_name] - p_now[link_name] - self.foot_sticking_tolerance
            p_ub = p_prev[link_name] - p_now[link_name] + self.foot_sticking_tolerance
            jxy = j_full[:2, idx]
            constraints.extend([jxy @ dqi >= p_lb[:2], jxy @ dqi <= p_ub[:2]])
        return constraints

    @staticmethod
    def _extract_lr_flags(flags: dict[str, bool]) -> tuple[bool, bool]:
        left = False
        right = False
        for k, v in flags.items():
            lk = k.lower()
            if lk.startswith("l") or "left" in lk:
                left = bool(v)
            if lk.startswith("r") or "right" in lk:
                right = bool(v)
        return left, right

    @staticmethod
    def _normalize_foot_sequence(
        seq: list[dict[str, bool]] | None, n_frames: int
    ) -> list[dict[str, bool]]:
        if seq is None:
            return [{} for _ in range(n_frames)]
        if len(seq) != n_frames:
            raise ValueError(f"Foot sticking sequence length {len(seq)} != n_frames {n_frames}")
        return seq

    def _normalize_nominal_sequence(
        self,
        seq: np.ndarray | None,
        n_frames: int,
        *,
        agent: str,
    ) -> np.ndarray | None:
        if seq is None:
            return None
        arr = np.asarray(seq, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != n_frames:
            raise ValueError(f"Nominal sequence for agent {agent} must have shape (T, D) with T={n_frames}, got {arr.shape}")

        if agent == "A":
            expected_opt = self.nq_a
            expected_local = 7 + self.robot_dof_a
            start_local = self.start_local_a
        else:
            expected_opt = self.nq_b
            expected_local = 7 + self.robot_dof_b
            start_local = self.start_local_b

        if arr.shape[1] == expected_opt:
            return np.asarray(arr, dtype=float)
        if arr.shape[1] >= expected_local:
            end = start_local + expected_opt
            return np.asarray(arr[:, start_local:end], dtype=float)

        raise ValueError(
            f"Nominal sequence width for agent {agent} must be {expected_opt} (opt slice) "
            f"or >= {expected_local} (full local state), got {arr.shape[1]}"
        )

    def _nominal_weight_for_frame(self, frame_idx: int) -> float:
        if self.w_nominal_tracking_init <= 0.0:
            return 0.0
        tau = max(self.nominal_tracking_tau, 1e-9)
        return float(self.w_nominal_tracking_init * np.exp(-float(frame_idx) / tau))

    @staticmethod
    def _project_nominal_indices(indices: np.ndarray, start_local: int, nq_local: int) -> np.ndarray:
        if indices.size == 0:
            return np.array([], dtype=int)
        mask = (indices >= start_local) & (indices < (start_local + nq_local))
        return (indices[mask] - start_local).astype(int)

    # --------------------------------------------------------------------------
    # Contact constraints
    # --------------------------------------------------------------------------
    def _build_ground_constraints(self, rows: list[ContactLinearization], dx: cp.Expression) -> list[cp.Constraint]:
        constraints: list[cp.Constraint] = []
        for row in rows:
            if not self._is_ground_pair(row.owner1, row.owner2):
                continue
            rhs = -row.phi - self.penetration_tolerance
            constraints.append(row.j_dx @ dx >= rhs)
        return constraints

    def _build_ab_constraints(
        self,
        rows: list[ContactLinearization],
        dx: cp.Expression,
    ) -> tuple[list[cp.Constraint], cp.Variable | None]:
        ab_rows = [r for r in rows if self._is_ab_pair(r.owner1, r.owner2)]
        if not ab_rows:
            return [], None

        # Prioritize non-whitelisted most violating pairs first, then whitelisted by violation.
        ab_rows = sorted(ab_rows, key=lambda r: (r.whitelisted, r.phi))
        kept = ab_rows[: max(0, int(self.ab_top_k_pairs))]

        n_slack = sum(1 for r in kept if r.whitelisted)
        slack_vars = cp.Variable(n_slack, nonneg=True, name="ab_contact_slack") if n_slack > 0 else None

        constraints: list[cp.Constraint] = []
        s_idx = 0
        for row in kept:
            if row.whitelisted:
                assert slack_vars is not None
                s = slack_vars[s_idx]
                s_idx += 1
                rhs = self.whitelist_margin - row.phi - s
                constraints.append(row.j_dx @ dx >= rhs)
            else:
                rhs = -row.phi - self.penetration_tolerance
                constraints.append(row.j_dx @ dx >= rhs)
        return constraints, slack_vars

    def _linearize_contact_rows(self, q: np.ndarray) -> list[ContactLinearization]:
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        candidates = self._prefilter_pairs_with_mj_collision(self.collision_detection_threshold)
        geom_names = [mujoco.mj_id2name(self.robot_model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(self.robot_model.ngeom)]

        rows: list[ContactLinearization] = []
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
            j_dx = np.concatenate([j_full[self.q_a_indices], j_full[self.q_b_indices]], axis=0)
            rows.append(
                ContactLinearization(
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
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if g1 < 0 or g2 < 0:
                continue
            if g1 == g2:
                continue
            candidates.add((min(g1, g2), max(g1, g2)))

        m.geom_margin[:] = self._saved_margins
        return candidates

    def _owner_from_prefixed_name(self, name: str) -> str:
        ln = name.lower()
        if "ground" in ln or "floor" in ln:
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
    def _is_ground_pair(owner1: str, owner2: str) -> bool:
        return (owner1 == "G" and owner2 in {"A", "B"}) or (owner2 == "G" and owner1 in {"A", "B"})

    @staticmethod
    def _is_ab_pair(owner1: str, owner2: str) -> bool:
        return (owner1 == "A" and owner2 == "B") or (owner1 == "B" and owner2 == "A")

    def _is_whitelisted_pair(self, geom1: str, geom2: str) -> bool:
        if not self.ab_pair_whitelist:
            return False

        pair_exact = tuple(sorted((geom1, geom2)))
        if pair_exact in self.ab_pair_whitelist:
            return True

        base1 = self._strip_agent_prefix(geom1)
        base2 = self._strip_agent_prefix(geom2)
        pair_base = tuple(sorted((base1, base2)))
        return pair_base in self.ab_pair_whitelist

    def _strip_agent_prefix(self, name: str) -> str:
        if self.robot_a_prefix and name.startswith(self.robot_a_prefix):
            return name[len(self.robot_a_prefix) :]
        if self.robot_b_prefix and name.startswith(self.robot_b_prefix):
            return name[len(self.robot_b_prefix) :]
        return name

    # --------------------------------------------------------------------------
    # Kinematics / Jacobians (reused architecture from single retargeter)
    # --------------------------------------------------------------------------
    def _compute_jacobian_for_contact_relative(self, geom1, geom2, geom1_name, geom2_name, fromto, dist):
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
            n_hat = np.array([0.0, 0.0, 0.0])

        j_body_a = self._calc_contact_jacobian_from_point(geom1.bodyid, pos1, input_world=True)
        j_body_b = self._calc_contact_jacobian_from_point(geom2.bodyid, pos2, input_world=True)
        return n_hat @ (j_body_a - j_body_b)

    def _build_transform_qdot_to_qvel_fast(self, use_world_omega: bool = True) -> np.ndarray:
        nq = self.robot_model.nq
        nv = self.robot_model.nv
        T = np.zeros((nv, nq), dtype=float)

        def e_world(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, qz, -qy],
                    [-qy, -qz, qw, qx],
                    [-qz, qy, -qx, qw],
                ]
            )

        def e_body(qw, qx, qy, qz):
            return np.array(
                [
                    [-qx, qw, -qz, qy],
                    [-qy, qz, qw, -qx],
                    [-qz, -qy, qx, qw],
                ]
            )

        e_fn = e_world if use_world_omega else e_body

        for j in range(self.robot_model.njnt):
            jt = self.robot_model.jnt_type[j]
            qadr = int(self.robot_model.jnt_qposadr[j])
            dadr = int(self.robot_model.jnt_dofadr[j])

            if jt == mujoco.mjtJoint.mjJNT_FREE:
                T[dadr : dadr + 3, qadr : qadr + 3] = np.eye(3)
                qw, qx, qy, qz = self.robot_data.qpos[qadr + 3 : qadr + 7]
                T[dadr + 3 : dadr + 6, qadr + 3 : qadr + 7] = 2.0 * e_fn(qw, qx, qy, qz)
            elif jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
                T[dadr, qadr] = 1.0
            elif jt == mujoco.mjtJoint.mjJNT_BALL:
                raise NotImplementedError("BALL joint mapping not implemented.")
        return T

    def _calc_contact_jacobian_from_point(self, body_idx: int, p_body: np.ndarray, input_world: bool = False) -> np.ndarray:
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
        T = self._build_transform_qdot_to_qvel_fast()
        return Jp @ T

    def _calc_manipulator_jacobians(
        self,
        q: np.ndarray,
        links: dict[str, str | tuple[str, np.ndarray] | BodyPointTarget],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        self.robot_data.qpos[:] = q
        mujoco.mj_forward(self.robot_model, self.robot_data)

        J_dict: dict[str, np.ndarray] = {}
        p_dict: dict[str, np.ndarray] = {}

        for key, target in links.items():
            body_name, point_offset = self._resolve_body_point_target(target)
            body_id = mujoco.mj_name2id(self.robot_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id == -1:
                # Ignore missing links to keep class robust to imperfect dual XML naming.
                continue
            J = self._calc_contact_jacobian_from_point(body_id, point_offset)
            rot_world_body = self.robot_data.xmat[body_id].reshape(3, 3)
            pos = self.robot_data.xpos[body_id] + rot_world_body @ point_offset
            J_dict[key] = np.asarray(J, dtype=float, copy=True)
            p_dict[key] = np.asarray(pos, dtype=float, copy=True)
        return J_dict, p_dict

    # --------------------------------------------------------------------------
    # Local robot state IO
    # --------------------------------------------------------------------------
    def _write_local_robot_state(self, q_full: np.ndarray, q_local: np.ndarray, *, agent: str) -> None:
        if agent == "A":
            block = self.q_block_a
            min_width = 7 + self.robot_dof_a
        else:
            block = self.q_block_b
            min_width = 7 + self.robot_dof_b
        if q_local.shape[0] < min_width:
            raise ValueError(f"Expected local state length >= {min_width}, got {q_local.shape[0]}")
        q_full[block[:min_width]] = q_local[:min_width]

    def _extract_local_robot_state(self, q_full: np.ndarray, *, agent: str) -> np.ndarray:
        if agent == "A":
            block = self.q_block_a
            width = 7 + self.robot_dof_a
        else:
            block = self.q_block_b
            width = 7 + self.robot_dof_b
        return np.asarray(q_full[block[:width]], dtype=np.float32)
