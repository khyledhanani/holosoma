#!/usr/bin/env python3
"""Summarize batch dual-human retarget runs from a manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def _owner_from_prefixed_name(name: str) -> str:
    lower = (name or "").lower()
    if "ground" in lower or "floor" in lower:
        return "G"
    if name.startswith("A_"):
        return "A"
    if name.startswith("B_"):
        return "B"
    if name.startswith("H_"):
        return "H"
    return "O"


def _owner_from_geom(model: mujoco.MjModel, geom_id: int) -> str:
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
    owner = _owner_from_prefixed_name(geom_name)
    if owner != "O":
        return owner

    body_id = int(model.geom_bodyid[int(geom_id)])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    owner = _owner_from_prefixed_name(body_name)
    if owner != "O":
        return owner

    if int(model.geom_type[int(geom_id)]) == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[int(geom_id)])
        if mesh_id >= 0:
            mesh_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
            owner = _owner_from_prefixed_name(mesh_name)
            if owner != "O":
                return owner

    return "O"


def _load_manifest(manifest_path: Path) -> dict[str, object]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_manifest_path(run_dir: Path | None, manifest_path: Path | None) -> Path:
    if manifest_path is not None:
        return manifest_path.resolve()
    if run_dir is None:
        raise ValueError("Pass either --manifest or --run-dir.")
    return (run_dir.resolve() / "manifest.json").resolve()


def _resolve_dual_scene(npz_path: Path) -> Path:
    data = np.load(npz_path, allow_pickle=True)
    if "dual_scene_xml" in data:
        return Path(str(data["dual_scene_xml"])).resolve()
    fallback = npz_path.with_name(f"{npz_path.stem}_dual_scene.xml")
    if fallback.exists():
        return fallback.resolve()
    raise FileNotFoundError(f"Could not resolve dual scene XML for {npz_path}")


def _compute_pair_metrics(npz_path: Path, collision_threshold: float) -> dict[str, float | int | str]:
    dual_scene_xml = _resolve_dual_scene(npz_path)
    data = np.load(npz_path, allow_pickle=True)
    model = mujoco.MjModel.from_xml_path(str(dual_scene_xml))
    mj_data = mujoco.MjData(model)

    qpos_a = np.asarray(data["qpos_A"], dtype=float)
    qpos_b = np.asarray(data["qpos_B"], dtype=float)
    n_a = qpos_a.shape[1]
    n_b = qpos_b.shape[1]

    min_dists: list[float] = []
    neg_pairs_per_frame: list[int] = []
    ab_contacts_per_frame: list[int] = []

    saved_margin = model.geom_margin.copy()
    for frame_idx in range(qpos_a.shape[0]):
        q = np.zeros(model.nq, dtype=float)
        q[:n_a] = qpos_a[frame_idx]
        q[n_a : n_a + n_b] = qpos_b[frame_idx]
        mj_data.qpos[:] = q
        mujoco.mj_forward(model, mj_data)

        model.geom_margin[:] = collision_threshold
        mujoco.mj_collision(model, mj_data)

        min_dist = np.nan
        neg_pairs = 0
        ab_contacts = 0
        for contact_idx in range(mj_data.ncon):
            contact = mj_data.contact[contact_idx]
            g1 = int(contact.geom1)
            g2 = int(contact.geom2)
            if { _owner_from_geom(model, g1), _owner_from_geom(model, g2) } != {"A", "B"}:
                continue
            ab_contacts += 1
            dist = float(contact.dist)
            min_dist = dist if np.isnan(min_dist) else min(min_dist, dist)
            if dist < 0.0:
                neg_pairs += 1

        min_dists.append(min_dist)
        neg_pairs_per_frame.append(neg_pairs)
        ab_contacts_per_frame.append(ab_contacts)

    model.geom_margin[:] = saved_margin

    min_d = np.asarray(min_dists, dtype=float)
    neg_pairs_arr = np.asarray(neg_pairs_per_frame, dtype=int)
    ab_contacts_arr = np.asarray(ab_contacts_per_frame, dtype=int)
    valid_mask = ~np.isnan(min_d)
    valid_min = min_d[valid_mask]

    thresholds = [-1e-3, -5e-3, -1e-2, -2e-2]
    metrics: dict[str, float | int | str] = {
        "dual_scene_xml": str(dual_scene_xml),
        "frames": int(qpos_a.shape[0]),
        "frames_with_any_ab_contact": int(np.sum(valid_mask)),
        "frames_with_penetration": int(np.sum(neg_pairs_arr > 0)),
        "max_penetrating_pairs_in_frame": int(np.max(neg_pairs_arr)) if neg_pairs_arr.size else 0,
        "mean_penetrating_pairs_per_frame": float(np.mean(neg_pairs_arr)) if neg_pairs_arr.size else 0.0,
        "mean_ab_contacts_per_frame": float(np.mean(ab_contacts_arr)) if ab_contacts_arr.size else 0.0,
        "min_dist_min": float(np.min(valid_min)) if valid_min.size else float("nan"),
        "min_dist_mean": float(np.mean(valid_min)) if valid_min.size else float("nan"),
    }
    for threshold in thresholds:
        key = f"frames_min_dist_lt_{abs(threshold):.3f}m".replace(".", "p")
        metrics[key] = int(np.sum(min_d < threshold))
    return metrics


def _row_sort_key(row: dict[str, object]) -> tuple[float, float, float]:
    min_dist = float(row.get("min_dist_min", float("inf")))
    frames_lt_5mm = float(row.get("frames_min_dist_lt_0p005m", 0))
    mean_pen = float(row.get("mean_penetrating_pairs_per_frame", 0.0))
    penetration_depth = max(0.0, -min_dist)
    return (frames_lt_5mm, penetration_depth, mean_pen)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: dict[str, object], ranked_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Dual Human Model Grid Summary",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Sequence: `{summary['sequence_id']}`",
        f"- Completed pairs: `{summary['completed_pairs']}`",
        f"- Failed pairs: `{summary['failed_pairs']}`",
        f"- Dry-run pairs: `{summary['dry_run_pairs']}`",
        "",
        "## Best Pairs",
        "",
        "| Pair | Worst min dist (m) | Frames < 5mm | Mean penetrating pairs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ranked_rows[:10]:
        lines.append(
            f"| `{row['pair']}` | {float(row['min_dist_min']):.6f} | "
            f"{int(row['frames_min_dist_lt_0p005m'])} | {float(row['mean_penetrating_pairs_per_frame']):.3f} |"
        )
    lines += [
        "",
        "## Worst Pairs",
        "",
        "| Pair | Worst min dist (m) | Frames < 5mm | Mean penetrating pairs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in ranked_rows[-10:]:
        lines.append(
            f"| `{row['pair']}` | {float(row['min_dist_min']):.6f} | "
            f"{int(row['frames_min_dist_lt_0p005m'])} | {float(row['mean_penetrating_pairs_per_frame']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a dual human model batch run.")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to a batch manifest.json.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Path to a batch run directory containing manifest.json.")
    parser.add_argument("--collision-threshold", type=float, default=0.08)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing summary outputs.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest_path = _resolve_manifest_path(args.run_dir, args.manifest)
    manifest = _load_manifest(manifest_path)
    run_dir = Path(str(manifest.get("run_dir", manifest_path.parent))).resolve()

    summary_dir = run_dir / "summary"
    summary_json = summary_dir / "summary.json"
    summary_csv = summary_dir / "pair_metrics.csv"
    summary_md = summary_dir / "summary.md"

    if not args.overwrite:
        existing = [path for path in (summary_json, summary_csv, summary_md) if path.exists()]
        if existing:
            raise FileExistsError(f"Summary outputs already exist: {', '.join(str(p) for p in existing)}")

    rows: list[dict[str, object]] = []
    failed_pairs = 0
    dry_run_pairs = 0
    skipped_pairs = 0
    for run_record in manifest.get("runs", []):
        row = dict(run_record)
        status = str(row.get("status", "unknown"))
        if status in {"ok", "skipped_existing"} and Path(str(row["final_npz"])).exists():
            metrics = _compute_pair_metrics(Path(str(row["final_npz"])).resolve(), args.collision_threshold)
            row.update(metrics)
        elif status == "failed":
            failed_pairs += 1
        elif status == "dry_run":
            dry_run_pairs += 1
        elif status == "skipped_existing":
            skipped_pairs += 1
        rows.append(row)

    ranked_rows = sorted(
        [
            row
            for row in rows
            if row.get("status") in {"ok", "skipped_existing"} and "min_dist_min" in row
        ],
        key=_row_sort_key,
    )

    summary_payload: dict[str, object] = {
        "run_id": manifest.get("run_id", run_dir.name),
        "run_dir": str(run_dir),
        "sequence_id": manifest.get("sequence_id"),
        "completed_pairs": len(ranked_rows),
        "failed_pairs": failed_pairs,
        "dry_run_pairs": dry_run_pairs,
        "skipped_existing_pairs": skipped_pairs,
        "collision_threshold": float(args.collision_threshold),
        "best_pair": ranked_rows[0]["pair"] if ranked_rows else None,
        "worst_pair": ranked_rows[-1]["pair"] if ranked_rows else None,
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "summary_json": str(summary_json),
    }

    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps({"summary": summary_payload, "rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(summary_csv, rows)
    _write_markdown(summary_md, summary_payload, ranked_rows)

    print(f"RUN_DIR={run_dir}")
    print(f"SUMMARY_JSON={summary_json}")
    print(f"SUMMARY_CSV={summary_csv}")
    print(f"SUMMARY_MD={summary_md}")
    print(f"COMPLETED_PAIRS={summary_payload['completed_pairs']}")
    print(f"FAILED_PAIRS={failed_pairs}")


if __name__ == "__main__":
    main()
