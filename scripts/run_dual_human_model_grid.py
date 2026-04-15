#!/usr/bin/env python3
"""Generate available humanoid variants and run the dual retargeter over their pair grid."""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import mujoco


ROOT = Path(__file__).resolve().parents[1]
CREATE_HUMANOID_SCRIPT = ROOT / "scripts" / "create_humanoid.py"
DUAL_RETARGET_SCRIPT = (
    ROOT
    / "src"
    / "holosoma_retargeting"
    / "holosoma_retargeting"
    / "examples"
    / "dual_robot_retarget.py"
)
DEFAULT_BASELINE_XML = ROOT / "models" / "mujoco_models" / "converted_model_test.xml"
DEFAULT_VARIANTS_DIR = ROOT / "models" / "mujoco_models" / "human_variants"
DEFAULT_OUTPUT_ROOT = ROOT / "DATA" / "human_model_combo_runs"
DEFAULT_ENV_PYTHON = Path("/Users/khyledhanani/.holosoma_deps/miniconda3/envs/hsretargeting/bin/python3.11")


@dataclass(frozen=True)
class HumanModelSpec:
    label: str
    xml_path: Path
    preset: str | None
    source: str


def _spec_to_record(spec: HumanModelSpec) -> dict[str, str | None]:
    return {
        "label": spec.label,
        "xml_path": str(spec.xml_path),
        "preset": spec.preset,
        "source": spec.source,
    }


def _load_preset_names() -> list[str]:
    tree = ast.parse(CREATE_HUMANOID_SCRIPT.read_text(encoding="utf-8"), filename=str(CREATE_HUMANOID_SCRIPT))
    for node in tree.body:
        target_name = None
        value = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "PRESET_BETAS":
                    target_name = target.id
                    value = node.value
                    break
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "PRESET_BETAS":
                target_name = node.target.id
                value = node.value

        if target_name == "PRESET_BETAS" and isinstance(value, ast.Dict):
            keys = []
            for key in value.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.append(key.value)
            if keys:
                return sorted(keys)
    raise RuntimeError(f"Could not parse PRESET_BETAS keys from {CREATE_HUMANOID_SCRIPT}")


def _inventory_specs(variants_dir: Path, include_baseline: bool) -> list[HumanModelSpec]:
    specs: list[HumanModelSpec] = []
    if include_baseline:
        specs.append(
            HumanModelSpec(
                label="original",
                xml_path=DEFAULT_BASELINE_XML,
                preset=None,
                source="existing",
            )
        )
    for preset in _load_preset_names():
        specs.append(
            HumanModelSpec(
                label=preset,
                xml_path=variants_dir / f"{preset}.xml",
                preset=preset,
                source="generated",
            )
        )
    return specs


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _mujoco_validate(xml_path: Path) -> dict[str, int]:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    return {
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "nmesh": int(model.nmesh),
    }


def _boolean_flag(flag: str, value: bool) -> str:
    return flag if value else f"--no-{flag[2:]}"


def _sanitize_label(raw: str) -> str:
    keep = [ch.lower() if ch.isalnum() else "_" for ch in raw.strip()]
    collapsed = "".join(keep)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "run"


def _default_run_id(sequence_id: str, collision_mode: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{_sanitize_label(sequence_id)}_{collision_mode}"


def _resolve_run_dir(
    *,
    output_root: Path,
    sequence_id: str,
    run_label: str | None,
    run_dir: Path | None,
    collision_mode: str,
) -> tuple[str, Path]:
    if run_dir is not None:
        resolved = run_dir.resolve()
        return resolved.name, resolved
    run_id = _sanitize_label(run_label) if run_label else _default_run_id(sequence_id, collision_mode)
    return run_id, (output_root / sequence_id / run_id).resolve()


def _ensure_model(
    spec: HumanModelSpec,
    *,
    python_exec: Path,
    overwrite_generated: bool,
    collision_mode: str,
    gender: str,
    replace_feet: bool,
    big_ankle: bool,
    box_body: bool,
) -> dict[str, object]:
    spec.xml_path.parent.mkdir(parents=True, exist_ok=True)
    generated = False

    if spec.preset is not None and (overwrite_generated or not spec.xml_path.exists()):
        cmd = [
            str(python_exec),
            str(CREATE_HUMANOID_SCRIPT),
            "--preset",
            spec.preset,
            "--gender",
            gender,
            "--collision-mode",
            collision_mode,
            "--out-xml",
            str(spec.xml_path),
            _boolean_flag("--replace-feet", replace_feet),
            _boolean_flag("--big-ankle", big_ankle),
            _boolean_flag("--box-body", box_body),
            "--overwrite",
        ]
        _run(cmd, cwd=ROOT)
        generated = True

    if not spec.xml_path.exists():
        raise FileNotFoundError(f"Missing humanoid XML for {spec.label}: {spec.xml_path}")

    stats = _mujoco_validate(spec.xml_path)
    result: dict[str, object] = {
        "label": spec.label,
        "xml_path": str(spec.xml_path),
        "source": spec.source,
        "preset": spec.preset,
        "generated": generated,
    }
    result.update(stats)
    return result


def _iter_pairs(models: list[HumanModelSpec], unique_pairs: bool) -> list[tuple[HumanModelSpec, HumanModelSpec]]:
    if unique_pairs:
        return list(itertools.combinations_with_replacement(models, 2))
    return list(itertools.product(models, repeat=2))


def _save_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all available humanoid presets and run dual_robot_retarget.py "
            "for every A/B model combination."
        )
    )
    parser.add_argument("--sequence-id", required=True, help="Inter-X sequence id to retarget.")
    parser.add_argument(
        "--python-exec",
        type=Path,
        default=DEFAULT_ENV_PYTHON if DEFAULT_ENV_PYTHON.exists() else Path(sys.executable),
        help="Python executable used to launch create_humanoid.py and dual_robot_retarget.py.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "DATA" / "interx_dual_inputs")
    parser.add_argument("--variants-dir", type=Path, default=DEFAULT_VARIANTS_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Optional label for this batch run. Default: timestamped unique run id.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Explicit run directory to use instead of creating a new timestamped folder.",
    )
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of model labels to run.")
    parser.add_argument(
        "--include-baseline",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the existing converted_model_test.xml as label 'original'.",
    )
    parser.add_argument(
        "--unique-pairs",
        action="store_true",
        help="Only run unordered pairs with replacement instead of the full ordered A/B grid.",
    )
    parser.add_argument(
        "--skip-existing-runs",
        action="store_true",
        help="Skip any pair whose final coupled output npz already exists.",
    )
    parser.add_argument(
        "--overwrite-generated",
        action="store_true",
        help="Regenerate preset XMLs even if they already exist in the variants dir.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate/validate models and write the manifest, but do not launch retarget runs.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed runs and continue processing the remaining pairs.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug cap forwarded to retarget.")
    parser.add_argument("--collision-mode", choices=["proxy", "mesh"], default="proxy")
    parser.add_argument("--gender", choices=["neutral", "male", "female"], default="neutral")
    parser.add_argument(
        "--replace-feet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to create_humanoid.py for generated variants.",
    )
    parser.add_argument(
        "--big-ankle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to create_humanoid.py for generated variants.",
    )
    parser.add_argument(
        "--box-body",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Forwarded to create_humanoid.py for generated variants.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants_dir = args.variants_dir.resolve()
    output_root = args.output_root.resolve()
    run_id, run_dir = _resolve_run_dir(
        output_root=output_root,
        sequence_id=args.sequence_id,
        run_label=args.run_label,
        run_dir=args.run_dir,
        collision_mode=args.collision_mode,
    )
    manifest_path = run_dir / "manifest.json"

    inventory = _inventory_specs(variants_dir, include_baseline=bool(args.include_baseline))
    if args.models:
        requested = set(args.models)
        inventory = [spec for spec in inventory if spec.label in requested]
        missing = sorted(requested - {spec.label for spec in inventory})
        if missing:
            raise ValueError(f"Unknown model labels: {', '.join(missing)}")
    if not inventory:
        raise ValueError("No human models selected.")

    manifest: dict[str, object] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sequence_id": args.sequence_id,
        "python_exec": str(args.python_exec.resolve()),
        "data_dir": str(args.data_dir.resolve()),
        "variants_dir": str(variants_dir),
        "output_root": str(output_root),
        "collision_mode": args.collision_mode,
        "include_baseline": bool(args.include_baseline),
        "unique_pairs": bool(args.unique_pairs),
        "models": [],
        "runs": [],
    }

    resolved_models: list[HumanModelSpec] = []
    for spec in inventory:
        model_record = _ensure_model(
            spec,
            python_exec=args.python_exec.resolve(),
            overwrite_generated=bool(args.overwrite_generated),
            collision_mode=args.collision_mode,
            gender=args.gender,
            replace_feet=bool(args.replace_feet),
            big_ankle=bool(args.big_ankle),
            box_body=bool(args.box_body),
        )
        manifest["models"].append(model_record)
        resolved_models.append(spec)
        _save_manifest(manifest_path, manifest)

    pair_count = 0
    failures = 0
    for pair_idx, (spec_a, spec_b) in enumerate(
        _iter_pairs(resolved_models, unique_pairs=bool(args.unique_pairs)),
        start=1,
    ):
        pair_count += 1
        pair_label = f"{spec_a.label}__{spec_b.label}"
        pair_dir_name = f"{pair_idx:03d}_{pair_label}"
        pair_output_dir = run_dir / "pairs" / pair_dir_name
        final_npz = pair_output_dir / f"{args.sequence_id}.npz"
        run_record: dict[str, object] = {
            "pair_index": pair_idx,
            "pair": pair_label,
            "pair_dir_name": pair_dir_name,
            "model_a": _spec_to_record(spec_a),
            "model_b": _spec_to_record(spec_b),
            "output_dir": str(pair_output_dir),
            "final_npz": str(final_npz),
        }

        if args.skip_existing_runs and final_npz.exists():
            run_record["status"] = "skipped_existing"
            manifest["runs"].append(run_record)
            _save_manifest(manifest_path, manifest)
            continue

        if args.dry_run:
            run_record["status"] = "dry_run"
            manifest["runs"].append(run_record)
            _save_manifest(manifest_path, manifest)
            continue

        cmd = [
            str(args.python_exec.resolve()),
            str(DUAL_RETARGET_SCRIPT),
            "--target-mode",
            "human_to_human",
            "--run-retarget",
            "--coupled-dual",
            "--sequence-id",
            args.sequence_id,
            "--data-dir",
            str(args.data_dir.resolve()),
            "--output-dir",
            str(pair_output_dir),
            "--human-model-xml-a",
            str(spec_a.xml_path),
            "--human-model-xml-b",
            str(spec_b.xml_path),
        ]
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(int(args.max_frames))])

        run_record["cmd"] = cmd
        try:
            _run(cmd, cwd=ROOT)
            if not final_npz.exists():
                raise FileNotFoundError(f"Expected output missing after retarget: {final_npz}")
            run_record["status"] = "ok"
        except Exception as exc:  # noqa: BLE001
            failures += 1
            run_record["status"] = "failed"
            run_record["error"] = str(exc)
            manifest["runs"].append(run_record)
            _save_manifest(manifest_path, manifest)
            if not args.continue_on_error:
                raise
            continue

        manifest["runs"].append(run_record)
        _save_manifest(manifest_path, manifest)

    manifest["summary"] = {
        "models": len(resolved_models),
        "pairs": pair_count,
        "failures": failures,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
    }
    _save_manifest(manifest_path, manifest)

    print(f"RUN_ID={run_id}")
    print(f"RUN_DIR={run_dir}")
    print(f"MODELS={len(resolved_models)}")
    print(f"PAIRS={pair_count}")
    print(f"FAILURES={failures}")
    print(f"MANIFEST={manifest_path}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
