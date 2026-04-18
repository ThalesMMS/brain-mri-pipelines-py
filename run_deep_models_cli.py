import argparse
import os
import sys
from pathlib import Path
from typing import Any


# Ensure brain_mri is in path
sys.path.append(os.getcwd())

from brain_mri.experiments.run_manifest import (
    capture_pip_freeze,
    generate_timestamp,
    git_commit,
    git_is_dirty,
    manifest_file as _manifest_file,
    relative_path as _relative_path,
    relativize_command,
    sha256_if_exists as _sha256_if_exists,
    unique_manifest_path as _unique_manifest_path,
    write_manifest,
)


def _dependencies_snapshot() -> list[str] | None:
    return capture_pip_freeze()


def _write_run_manifest(
    *,
    app: Any,
    args: argparse.Namespace,
    base_dir: Path,
    split_csv: Path,
    backbones: list[str],
    results_by_backbone: dict[str, Any],
    experiments_sha256_before: str | None,
) -> Path:
    timestamp = generate_timestamp()
    manifest_path = _unique_manifest_path(app.output_dir / "manifests" / "deep_models", timestamp)

    outputs: dict[str, Any] = {}
    for backbone in backbones:
        result = results_by_backbone.get(backbone)
        artifact_paths = getattr(result, "artifact_paths", {}) or {}
        outputs[backbone] = {
            name: _manifest_file(Path(path), base_dir)
            for name, path in artifact_paths.items()
        }

    manifest = {
        "cli": "deep_models",
        "timestamp": timestamp,
        "git_commit": git_commit(),
        "git_dirty": git_is_dirty(),
        "command": relativize_command(list(sys.argv), base_dir),
        "args": {
            "seed": int(args.seed),
            "epochs": int(args.epochs),
            "backbones": backbones,
            "multimodal": bool(args.multimodal),
        },
        "env": {
            "DEEP_SCENARIO": os.environ.get("DEEP_SCENARIO"),
            "USE_MULTIMODAL": os.environ.get("USE_MULTIMODAL"),
            "RESNET_SEED": os.environ.get("RESNET_SEED"),
        },
        "inputs": {
            "split_csv": _manifest_file(split_csv, base_dir),
            "training_experiments": {
                "path": _relative_path(app.experiment_history_path, base_dir),
                "sha256": experiments_sha256_before,
            },
        },
        "outputs": {
            **outputs,
            "training_experiments": _manifest_file(app.experiment_history_path, base_dir),
        },
        "dependencies": _dependencies_snapshot(),
    }
    write_manifest(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deep models (EfficientNet/MedicalNet/DenseNet) headlessly on the current split."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.getenv("RESNET_SEED", "42")),
        help="Seed for deterministic training (exported to RESNET_SEED).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Max epochs for classification (passed to hparams['max_epochs']).",
    )
    parser.add_argument(
        "--backbones",
        type=str,
        default="efficientnet,medicalnet,densenet",
        help="Comma-separated list: efficientnet,medicalnet,densenet",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Enable multimodal fusion with clinical features (USE_MULTIMODAL=1).",
    )
    args = parser.parse_args()

    base_dir = Path(os.getcwd())
    print(f"Running deep models headlessly in {base_dir}")

    from brain_mri.experiments.history import ExperimentHistoryMixin
    from brain_mri.ml.ml_training import MLTrainingMixin

    class HeadlessApp(ExperimentHistoryMixin, MLTrainingMixin):
        def __init__(self, base_dir: Path):
            self.base_dir = base_dir
            self.dataset_dir = self.base_dir / "axl"
            self.output_dir = self.base_dir / "output"
            self.output_dir.mkdir(exist_ok=True)
            self.descriptors_csv = self.output_dir / "ventricle_descriptors.csv"
            self.csv_path = self.base_dir / "oasis_longitudinal_demographic.csv"
            self.experiment_history_path = self.output_dir / "training_experiments.json"

    app = HeadlessApp(base_dir)

    split_csv = app.output_dir / "exam_level_dataset_split.csv"
    if not split_csv.exists():
        print(f"[WARN] Split CSV not found at {split_csv}. Creating dataset...")
        app.create_exam_level_dataset()

    os.environ["RESNET_SEED"] = str(int(args.seed))
    os.environ.setdefault("DEEP_SCENARIO", "deep_current_split")
    os.environ["USE_MULTIMODAL"] = "1" if args.multimodal else "0"

    backbones = [b.strip() for b in str(args.backbones).split(",") if b.strip()]
    allowed = {"efficientnet", "medicalnet", "densenet"}
    unknown = [b for b in backbones if b not in allowed]
    if unknown:
        raise SystemExit(f"Unknown backbones: {unknown}. Allowed: {sorted(allowed)}")

    experiments_sha256_before = _sha256_if_exists(app.experiment_history_path)
    results_by_backbone: dict[str, Any] = {}
    for backbone in backbones:
        print(f"\n--- Deep: {backbone} (classification) ---")
        results_by_backbone[backbone] = app._train_pytorch_model(
            mode="classification",
            backbone=backbone,
            hparams={"max_epochs": int(args.epochs), "seed": int(args.seed)},
        )

    manifest_path = _write_run_manifest(
        app=app,
        args=args,
        base_dir=base_dir,
        split_csv=split_csv,
        backbones=backbones,
        results_by_backbone=results_by_backbone,
        experiments_sha256_before=experiments_sha256_before,
    )
    print(f"[OK] Wrote run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
