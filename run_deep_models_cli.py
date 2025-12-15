import argparse
import os
import sys
from pathlib import Path


# Ensure brain_mri is in path
sys.path.append(os.getcwd())


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

    for backbone in backbones:
        print(f"\n--- Deep: {backbone} (classification) ---")
        app._train_pytorch_model(
            mode="classification",
            backbone=backbone,
            hparams={"max_epochs": int(args.epochs), "seed": int(args.seed)},
        )


if __name__ == "__main__":
    main()
