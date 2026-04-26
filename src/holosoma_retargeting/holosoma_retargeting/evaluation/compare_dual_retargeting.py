"""Batch-compare IAMR vs coupled dual retargeting on Inter-X sequences.

This script runs both dual retargeters, evaluates per-agent robot-only metrics,
and writes detailed CSV plus aggregate JSON summaries.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import MotionDataConfig
from holosoma_retargeting.config_types.retargeter import RetargeterConfig
from holosoma_retargeting.config_types.robot import RobotConfig
from holosoma_retargeting.config_types.task import TaskConfig
from holosoma_retargeting.path_utils import resolve_portable_path
from holosoma_retargeting.evaluation.eval_retargeting import (
    RetargetingEvaluator,
    create_task_constants,
)
from holosoma_retargeting.examples.dual_robot_retarget import (
    DualRetargetConfig,
    _resolve_asset_paths,
    main as run_dual_baseline,
)
from holosoma_retargeting.examples.iamr_dual_retarget import (
    IAMRRetargetConfig,
    main as run_iamr,
)
from holosoma_retargeting.src.utils import extract_foot_sticking_sequence_velocity


TOE_BODY_NAMES = {
    "g1": ("left_ankle_roll_sphere_5_link", "right_ankle_roll_sphere_5_link"),
    "t1": ("left_foot_sphere_5_link", "right_foot_sphere_5_link"),
}


@dataclass
class CompareDualRetargetingConfig:
    data_dir: Path = Path("DATA/interx_dual_inputs")
    raw_interx_dir: Path = Path("Inter-X/visualize/smplx_viewer_tool/data/motions")
    output_root: Path = Path("DATA/interx_compare_iamr_vs_baseline")

    sequence_ids: list[str] | None = None
    sequence_ids_file: Path | None = None
    num_sequences: int = 50

    robot: str = "g1"
    robot_a: str | None = None
    robot_b: str | None = None
    data_format: str = "smplx"
    max_frames: int | None = None
    interaction_distance_threshold: float = 0.25

    overwrite: bool = False
    skip_existing: bool = True
    evaluate_only: bool = False
    run_baseline: bool = True
    run_iamr: bool = True

    robot_config: RobotConfig = field(default_factory=lambda: RobotConfig(robot_type="g1"))
    motion_data_config: MotionDataConfig = field(
        default_factory=lambda: MotionDataConfig(data_format="smplx", robot_type="g1")
    )
    task_config: TaskConfig = field(default_factory=TaskConfig)
    retargeter: RetargeterConfig = field(default_factory=RetargeterConfig)


def _resolve_robot_and_motion_config(
    cfg: CompareDualRetargetingConfig,
) -> tuple[RobotConfig, RobotConfig, MotionDataConfig, MotionDataConfig]:
    robot_type_a = cfg.robot_a or cfg.robot
    robot_type_b = cfg.robot_b or cfg.robot

    robot_config_a = cfg.robot_config
    if robot_config_a.robot_type != robot_type_a:
        robot_config_a = replace(robot_config_a, robot_type=robot_type_a)

    robot_config_b = cfg.robot_config
    if robot_config_b.robot_type != robot_type_b:
        robot_config_b = replace(robot_config_b, robot_type=robot_type_b)

    motion_config_a = cfg.motion_data_config
    if motion_config_a.robot_type != robot_type_a or motion_config_a.data_format != cfg.data_format:
        motion_config_a = replace(motion_config_a, robot_type=robot_type_a, data_format=cfg.data_format)

    motion_config_b = cfg.motion_data_config
    if motion_config_b.robot_type != robot_type_b or motion_config_b.data_format != cfg.data_format:
        motion_config_b = replace(motion_config_b, robot_type=robot_type_b, data_format=cfg.data_format)

    return robot_config_a, robot_config_b, motion_config_a, motion_config_b


def _select_sequence_ids(cfg: CompareDualRetargetingConfig) -> list[str]:
    if cfg.sequence_ids:
        return cfg.sequence_ids

    if cfg.sequence_ids_file is not None:
        return [line.strip() for line in cfg.sequence_ids_file.read_text().splitlines() if line.strip()]

    files = sorted(cfg.data_dir.glob("*.npz"))
    return [path.stem for path in files[: cfg.num_sequences]]


def _build_evaluator(
    robot_config: RobotConfig,
    motion_config: MotionDataConfig,
) -> RetargetingEvaluator:
    asset_root = Path(__file__).resolve().parents[1]
    constants = create_task_constants(
        robot_config=robot_config,
        motion_data_config=motion_config,
        object_name="ground",
    )
    _resolve_asset_paths(constants, asset_root)

    return RetargetingEvaluator(
        robot_model_path=str(constants.ROBOT_URDF_FILE),
        object_model_path=None,
        object_name="ground",
        demo_joints=list(motion_config.resolved_demo_joints),
        joints_mapping=dict(motion_config.resolved_joints_mapping),
        visualize=False,
        constants=constants,
    )


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _safe_max(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(values))


def _detect_foot_sliding_generic(
    evaluator: RetargetingEvaluator,
    q_trajectory: np.ndarray,
    contact_sequences: list[dict[str, bool]],
    toe_body_names: tuple[str, str],
) -> tuple[float, np.ndarray]:
    left_toe_positions = []
    right_toe_positions = []
    for q in q_trajectory:
        toe_positions = evaluator._get_robot_link_positions(q, list(toe_body_names))
        left_toe_positions.append(toe_positions[0])
        right_toe_positions.append(toe_positions[1])

    left_toe_positions = np.asarray(left_toe_positions)
    right_toe_positions = np.asarray(right_toe_positions)

    left_toe_xy_vel = np.linalg.norm(np.diff(left_toe_positions[:, :2], axis=0), axis=1)
    right_toe_xy_vel = np.linalg.norm(np.diff(right_toe_positions[:, :2], axis=0), axis=1)
    left_toe_xy_vel = np.concatenate([[0.0], left_toe_xy_vel])
    right_toe_xy_vel = np.concatenate([[0.0], right_toe_xy_vel])

    left_contact = np.asarray([frame["L_Toe"] for frame in contact_sequences], dtype=bool)
    right_contact = np.asarray([frame["R_Toe"] for frame in contact_sequences], dtype=bool)

    left_sliding = left_contact & (left_toe_xy_vel > evaluator.sliding_threshold)
    right_sliding = right_contact & (right_toe_xy_vel > evaluator.sliding_threshold)

    num_contact_frames = int(np.sum(left_contact | right_contact))
    max_velocities = np.max(
        np.vstack(
            [
                left_toe_xy_vel * left_sliding,
                right_toe_xy_vel * right_sliding,
            ]
        ),
        axis=0,
    )
    max_velocities = max_velocities[max_velocities > 0]
    if num_contact_frames == 0:
        return 0.0, max_velocities
    return len(max_velocities) / num_contact_frames, max_velocities


def _load_method_payload(method: str, raw_data: Any, agent: str) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(raw_data[f"qpos_{agent}"], dtype=np.float32)
    human_joints = np.asarray(raw_data[f"human_joints_{agent}"], dtype=np.float32)

    if method == "iamr":
        scale_key = f"scale_{agent}"
        if scale_key in raw_data.files:
            human_joints = human_joints * float(raw_data[scale_key])

    n = min(len(qpos), len(human_joints))
    return qpos[:n], human_joints[:n]


def _compute_link_tracking_payload(
    evaluator: RetargetingEvaluator,
    qpos: np.ndarray,
    human_joints: np.ndarray,
) -> dict[str, Any]:
    joint_names = list(evaluator.joints_mapping.keys())
    link_names = [evaluator.joints_mapping[joint_name] for joint_name in joint_names]
    joint_indices = [evaluator.demo_joints.index(joint_name) for joint_name in joint_names]

    source_points = human_joints[:, joint_indices, :]
    robot_points = np.asarray(
        [evaluator._get_robot_link_positions(q, link_names) for q in qpos],
        dtype=np.float32,
    )

    return {
        "qpos": qpos,
        "human_joints": human_joints,
        "joint_names": joint_names,
        "link_names": link_names,
        "source_points": source_points,
        "robot_points": robot_points,
    }


def _compute_tracking_metrics(source_points: np.ndarray, robot_points: np.ndarray) -> dict[str, float]:
    position_errors = np.linalg.norm(robot_points - source_points, axis=-1)
    frame_mean_errors = position_errors.mean(axis=1)
    return {
        "tracking_error_mean": float(np.mean(position_errors)),
        "tracking_error_frame_mean": float(np.mean(frame_mean_errors)),
        "tracking_error_p95": float(np.percentile(position_errors, 95)),
        "tracking_error_max": float(np.max(position_errors)),
    }


def _evaluate_agent(
    *,
    method: str,
    sequence_id: str,
    agent: str,
    payload: dict[str, Any],
    evaluator: RetargetingEvaluator,
    motion_config: MotionDataConfig,
    robot_type: str,
    runtime_sec: float | None,
) -> dict[str, Any]:
    qpos = payload["qpos"]
    human_joints = payload["human_joints"]

    penetration_duration, penetration_max_depths = evaluator.evaluate_penetration(qpos)
    contact_sequences = extract_foot_sticking_sequence_velocity(
        human_joints,
        evaluator.demo_joints,
        motion_config.toe_names,
    )
    sliding_duration, max_toe_sliding_velocities = _detect_foot_sliding_generic(
        evaluator,
        qpos,
        contact_sequences[: len(qpos)],
        TOE_BODY_NAMES[robot_type],
    )

    penetration_max_depths = np.asarray(penetration_max_depths, dtype=np.float32)
    max_toe_sliding_velocities = np.asarray(max_toe_sliding_velocities, dtype=np.float32)
    tracking_metrics = _compute_tracking_metrics(payload["source_points"], payload["robot_points"])

    return {
        "sequence_id": sequence_id,
        "method": method,
        "agent": agent,
        "robot_type": robot_type,
        "num_frames": int(len(qpos)),
        "runtime_sec": runtime_sec,
        "penetration_duration": float(penetration_duration),
        "penetration_max_depth_mean": _safe_mean(penetration_max_depths),
        "penetration_max_depth_max": _safe_max(penetration_max_depths),
        "sliding_duration": float(sliding_duration),
        "max_toe_sliding_velocity_mean": _safe_mean(max_toe_sliding_velocities),
        "max_toe_sliding_velocity_max": _safe_max(max_toe_sliding_velocities),
        **tracking_metrics,
    }


def _compute_interaction_metrics(
    payload_a: dict[str, Any],
    payload_b: dict[str, Any],
    distance_threshold: float,
) -> dict[str, float]:
    source_a = payload_a["source_points"]
    source_b = payload_b["source_points"]
    robot_a = payload_a["robot_points"]
    robot_b = payload_b["robot_points"]

    source_dist = np.linalg.norm(source_a[:, :, None, :] - source_b[:, None, :, :], axis=-1)
    robot_dist = np.linalg.norm(robot_a[:, :, None, :] - robot_b[:, None, :, :], axis=-1)

    source_active = source_dist <= distance_threshold
    robot_active = robot_dist <= distance_threshold

    tp = int(np.sum(source_active & robot_active))
    fp = int(np.sum(~source_active & robot_active))
    fn = int(np.sum(source_active & ~robot_active))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0.0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)
    jaccard = tp / max(tp + fp + fn, 1)

    active_distance_errors = np.abs(robot_dist - source_dist)[source_active]
    active_distance_sq_errors = np.square(robot_dist - source_dist)[source_active]

    return {
        "source_interaction_density": float(np.mean(source_active)),
        "robot_interaction_density": float(np.mean(robot_active)),
        "interaction_precision": float(precision),
        "interaction_recall": float(recall),
        "interaction_f1": float(f1),
        "interaction_jaccard": float(jaccard),
        "interaction_distance_mae": _safe_mean(active_distance_errors),
        "interaction_distance_rmse": float(np.sqrt(_safe_mean(active_distance_sq_errors))),
    }


def _run_baseline_sequence(cfg: CompareDualRetargetingConfig, sequence_id: str, output_dir: Path) -> float | None:
    raw_path = output_dir / f"{sequence_id}_dual_coupled_raw.npz"
    if raw_path.exists() and cfg.skip_existing and not cfg.overwrite:
        return None

    if cfg.evaluate_only or not cfg.run_baseline:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing baseline raw output: {raw_path}")
        return None

    start = time.perf_counter()
    run_dual_baseline(
        DualRetargetConfig(
            data_dir=cfg.data_dir,
            sequence_id=sequence_id,
            output_dir=output_dir,
            robot=cfg.robot,
            robot_a=cfg.robot_a,
            robot_b=cfg.robot_b,
            data_format=cfg.data_format,
            max_frames=cfg.max_frames,
            run_retarget=True,
            coupled_dual=True,
            coupled_warm_start_nominal=True,
            robot_config=cfg.robot_config,
            motion_data_config=cfg.motion_data_config,
            task_config=cfg.task_config,
            retargeter=cfg.retargeter,
        )
    )
    return time.perf_counter() - start


def _run_iamr_sequence(cfg: CompareDualRetargetingConfig, sequence_id: str, output_dir: Path) -> float | None:
    raw_path = output_dir / f"{sequence_id}_iamr_raw.npz"
    if raw_path.exists() and cfg.skip_existing and not cfg.overwrite:
        return None

    if cfg.evaluate_only or not cfg.run_iamr:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing IAMR raw output: {raw_path}")
        return None

    start = time.perf_counter()
    run_iamr(
        IAMRRetargetConfig(
            data_dir=cfg.data_dir,
            sequence_id=sequence_id,
            raw_interx_dir=cfg.raw_interx_dir,
            output_dir=output_dir,
            robot=cfg.robot,
            robot_a=cfg.robot_a,
            robot_b=cfg.robot_b,
            data_format=cfg.data_format,
            max_frames=cfg.max_frames,
            robot_config=cfg.robot_config,
            motion_data_config=cfg.motion_data_config,
            task_config=cfg.task_config,
            retargeter=cfg.retargeter,
        )
    )
    return time.perf_counter() - start


def _aggregate_method_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "penetration_duration",
        "penetration_max_depth_mean",
        "penetration_max_depth_max",
        "sliding_duration",
        "max_toe_sliding_velocity_mean",
        "max_toe_sliding_velocity_max",
        "tracking_error_mean",
        "tracking_error_frame_mean",
        "tracking_error_p95",
        "tracking_error_max",
    ]

    summary: dict[str, Any] = {"rows": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{metric}_std"] = float(np.std(values)) if values else 0.0

    runtimes = [float(row["runtime_sec"]) for row in rows if row["runtime_sec"] is not None]
    summary["runtime_sec_mean"] = float(np.mean(runtimes)) if runtimes else 0.0
    summary["runtime_sec_std"] = float(np.std(runtimes)) if runtimes else 0.0
    return summary


def _aggregate_sequence_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "penetration_duration_mean_ab",
        "sliding_duration_mean_ab",
        "tracking_error_mean_ab",
        "tracking_error_p95_mean_ab",
        "interaction_precision",
        "interaction_recall",
        "interaction_f1",
        "interaction_jaccard",
        "interaction_distance_mae",
        "interaction_distance_rmse",
        "source_interaction_density",
        "robot_interaction_density",
    ]

    summary: dict[str, Any] = {"rows": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        summary[f"{metric}_mean"] = float(np.mean(values)) if values else 0.0
        summary[f"{metric}_std"] = float(np.std(values)) if values else 0.0
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"Unsupported type: {type(value)!r}")


def main(cfg: CompareDualRetargetingConfig) -> None:
    cfg.data_dir = resolve_portable_path(cfg.data_dir, prefer_bundle=True, must_exist=True)
    cfg.raw_interx_dir = resolve_portable_path(cfg.raw_interx_dir, prefer_bundle=True, must_exist=True)
    cfg.output_root = resolve_portable_path(cfg.output_root, prefer_bundle=True)
    if cfg.sequence_ids_file is not None:
        cfg.sequence_ids_file = resolve_portable_path(cfg.sequence_ids_file, prefer_bundle=True, must_exist=True)

    if cfg.robot not in TOE_BODY_NAMES:
        raise ValueError(f"Unsupported robot for sliding metrics: {cfg.robot}")

    sequence_ids = _select_sequence_ids(cfg)
    if not sequence_ids:
        raise ValueError("No sequence IDs selected.")

    robot_config_a, robot_config_b, motion_config_a, motion_config_b = _resolve_robot_and_motion_config(cfg)
    if robot_config_a.robot_type not in TOE_BODY_NAMES or robot_config_b.robot_type not in TOE_BODY_NAMES:
        raise ValueError("This comparison script currently supports g1/t1 toe-link mappings only.")

    baseline_dir = cfg.output_root / "baseline"
    iamr_dir = cfg.output_root / "iamr"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    iamr_dir.mkdir(parents=True, exist_ok=True)

    evaluator_a = _build_evaluator(robot_config_a, motion_config_a)
    evaluator_b = _build_evaluator(robot_config_b, motion_config_b)

    detailed_rows: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []

    for idx, sequence_id in enumerate(sequence_ids, start=1):
        print(f"[{idx}/{len(sequence_ids)}] {sequence_id}")

        baseline_runtime = _run_baseline_sequence(cfg, sequence_id, baseline_dir)
        iamr_runtime = _run_iamr_sequence(cfg, sequence_id, iamr_dir)

        baseline_raw_path = baseline_dir / f"{sequence_id}_dual_coupled_raw.npz"
        iamr_raw_path = iamr_dir / f"{sequence_id}_iamr_raw.npz"
        baseline_raw = np.load(baseline_raw_path, allow_pickle=True)
        iamr_raw = np.load(iamr_raw_path, allow_pickle=True)

        for method, raw_data, runtime_sec in (
            ("baseline", baseline_raw, baseline_runtime),
            ("iamr", iamr_raw, iamr_runtime),
        ):
            qpos_a, human_a = _load_method_payload(method, raw_data, "A")
            qpos_b, human_b = _load_method_payload(method, raw_data, "B")
            payload_a = _compute_link_tracking_payload(evaluator_a, qpos_a, human_a)
            payload_b = _compute_link_tracking_payload(evaluator_b, qpos_b, human_b)

            row_a = _evaluate_agent(
                method=method,
                sequence_id=sequence_id,
                agent="A",
                payload=payload_a,
                evaluator=evaluator_a,
                motion_config=motion_config_a,
                robot_type=robot_config_a.robot_type,
                runtime_sec=runtime_sec,
            )
            row_b = _evaluate_agent(
                method=method,
                sequence_id=sequence_id,
                agent="B",
                payload=payload_b,
                evaluator=evaluator_b,
                motion_config=motion_config_b,
                robot_type=robot_config_b.robot_type,
                runtime_sec=runtime_sec,
            )
            detailed_rows.extend([row_a, row_b])
            interaction_metrics = _compute_interaction_metrics(
                payload_a,
                payload_b,
                cfg.interaction_distance_threshold,
            )

            seq_summary = {
                "sequence_id": sequence_id,
                "method": method,
                "runtime_sec": runtime_sec,
                "penetration_duration_mean_ab": float(np.mean([row_a["penetration_duration"], row_b["penetration_duration"]])),
                "sliding_duration_mean_ab": float(np.mean([row_a["sliding_duration"], row_b["sliding_duration"]])),
                "tracking_error_mean_ab": float(np.mean([row_a["tracking_error_mean"], row_b["tracking_error_mean"]])),
                "tracking_error_p95_mean_ab": float(np.mean([row_a["tracking_error_p95"], row_b["tracking_error_p95"]])),
                **interaction_metrics,
            }
            if method == "iamr":
                seq_summary["interaction_density"] = float(np.mean(raw_data["interaction_graph"]))
                seq_summary["contact_density"] = float(np.mean(raw_data["contact_graph"]))
            sequence_summaries.append(seq_summary)

    detail_csv = cfg.output_root / "comparison_details.csv"
    summary_json = cfg.output_root / "comparison_summary.json"
    sequence_json = cfg.output_root / "sequence_summaries.json"

    _write_csv(detail_csv, detailed_rows)

    baseline_rows = [row for row in detailed_rows if row["method"] == "baseline"]
    iamr_rows = [row for row in detailed_rows if row["method"] == "iamr"]
    baseline_sequence_rows = [row for row in sequence_summaries if row["method"] == "baseline"]
    iamr_sequence_rows = [row for row in sequence_summaries if row["method"] == "iamr"]

    aggregate = {
        "config": asdict(cfg),
        "num_sequences": len(sequence_ids),
        "baseline": _aggregate_method_rows(baseline_rows),
        "iamr": _aggregate_method_rows(iamr_rows),
        "baseline_sequence": _aggregate_sequence_rows(baseline_sequence_rows),
        "iamr_sequence": _aggregate_sequence_rows(iamr_sequence_rows),
        "delta_iamr_minus_baseline": {},
        "delta_iamr_minus_baseline_sequence": {},
    }

    for key, value in aggregate["iamr"].items():
        if key in aggregate["baseline"] and isinstance(value, (float, int)):
            aggregate["delta_iamr_minus_baseline"][key] = float(value) - float(aggregate["baseline"][key])
    for key, value in aggregate["iamr_sequence"].items():
        if key in aggregate["baseline_sequence"] and isinstance(value, (float, int)):
            aggregate["delta_iamr_minus_baseline_sequence"][key] = (
                float(value) - float(aggregate["baseline_sequence"][key])
            )

    summary_json.write_text(json.dumps(aggregate, indent=2, default=_json_default))
    sequence_json.write_text(json.dumps(sequence_summaries, indent=2, default=_json_default))

    print(f"Wrote detailed metrics to {detail_csv}")
    print(f"Wrote aggregate summary to {summary_json}")
    print(f"Wrote per-sequence summary to {sequence_json}")


if __name__ == "__main__":
    main(tyro.cli(CompareDualRetargetingConfig))
