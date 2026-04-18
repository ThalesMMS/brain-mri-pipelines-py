from __future__ import annotations

import os
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import messagebox
except Exception:
    tk = None

    class _HeadlessMessageBox:
        def showinfo(self, *args, **kwargs):
            return None

        def showwarning(self, *args, **kwargs):
            return None

        def showerror(self, *args, **kwargs):
            return None

        def askyesno(self, *args, **kwargs):
            return True

    messagebox = _HeadlessMessageBox()

try:
    from matplotlib.figure import Figure
except Exception:
    Figure = None

from .dataset_builder import (
    DATASET_SPLIT_FILENAME,
    DatasetBuilderConfig,
    create_exam_level_dataset as build_exam_level_dataset,
    list_orientation_paths,
    populate_orientation_paths,
)
from .deep_training import DeepTrainingConfig, train_pytorch_model
from .training_utils import load_split_dataframe


class MLTrainingMixin:
    def _dataset_builder_config(self) -> DatasetBuilderConfig:
        return DatasetBuilderConfig(
            dataset_dir=Path(self.dataset_dir),
            output_dir=Path(self.output_dir),
            csv_path=Path(self.csv_path),
            descriptors_csv=Path(self.descriptors_csv),
        )

    def _resolve_split_csv_path(self) -> Path:
        override = os.getenv("SPLIT_CSV_PATH", "").strip()
        if override:
            return Path(override)
        return Path(self.output_dir) / DATASET_SPLIT_FILENAME

    def _save_experiment_callback(self):
        return getattr(self, "_save_experiment", None)

    def _is_headless(self) -> bool:
        return not hasattr(self, "root") or self.root is None

    def _plot_confusion_figure(self, cm, classes, title="Teste"):
        if Figure is None or not hasattr(self, "plot_confusion_matrix"):
            return None
        fig = Figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        self.plot_confusion_matrix(ax, cm, classes, title)
        fig.tight_layout()
        return fig

    def _save_and_maybe_show_figure(self, figure, output_path: Path, title: str):
        if figure is None:
            return None
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        if not self._is_headless() and hasattr(self, "_show_plot_window"):
            try:
                self._show_plot_window(title, figure)
            except Exception:
                pass
        return output_path

    def _load_training_dataframe(self, required_columns):
        return load_split_dataframe(self._resolve_split_csv_path(), required_columns=required_columns)

    def _list_orientation_paths(self, mri_id, base_dirs):
        return list_orientation_paths(mri_id, base_dirs)

    def _populate_orientation_paths(self, df_subset, base_dirs):
        return populate_orientation_paths(df_subset, base_dirs)

    def create_exam_level_dataset(self):
        try:
            dataset_df, output_path = build_exam_level_dataset(self._dataset_builder_config())
        except ImportError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return None
        except (FileNotFoundError, KeyError) as exc:
            messagebox.showerror("Erro", str(exc))
            return None
        except ValueError as exc:
            messagebox.showwarning("Aviso", str(exc))
            return None

        messagebox.showinfo("Sucesso", f"Dataset criado em {output_path.name}\nTotal: {len(dataset_df)} exames.")
        return dataset_df, output_path

    def train_svm_classifier(self, features=None, scenario=None):
        try:
            from .classical_training import train_svm_classifier as run_train_svm
            df = self._load_training_dataframe(["split", "Final_Group"])
        except FileNotFoundError:
            messagebox.showwarning("Aviso", "Crie o dataset primeiro.")
            return None
        except (ImportError, ValueError) as exc:
            messagebox.showerror("Erro", str(exc))
            return None

        try:
            result = run_train_svm(
                df=df,
                features=features,
                scenario=scenario,
                output_dir=Path(self.output_dir),
                save_experiment_fn=self._save_experiment_callback(),
            )
        except ImportError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return None
        except ValueError as exc:
            messagebox.showwarning("Aviso", str(exc))
            return None

        if result.confusion_matrix is not None:
            try:
                fig = self._plot_confusion_figure(result.confusion_matrix, ["0", "1"])
                self._save_and_maybe_show_figure(fig, Path(self.output_dir) / "confusion_svm.png", "Matriz SVM")
            except Exception:
                pass

        messagebox.showinfo("Resultado SVM", result.message)
        return result

    def train_xgboost_regressor(self, features=None, scenario=None, seed=None):
        try:
            from .classical_training import train_xgboost_regressor as run_train_xgboost
            df = self._load_training_dataframe(["split", "Subject_ID", "age"])
        except FileNotFoundError:
            messagebox.showwarning("Aviso", "Crie o dataset primeiro.")
            return None
        except (ImportError, ValueError) as exc:
            messagebox.showerror("Erro", str(exc))
            return None

        try:
            result = run_train_xgboost(
                df=df,
                features=features,
                scenario=scenario,
                output_dir=Path(self.output_dir),
                seed=seed,
                save_experiment_fn=self._save_experiment_callback(),
            )
        except ImportError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return None
        except ValueError as exc:
            messagebox.showwarning("Aviso", str(exc))
            return None

        messagebox.showinfo("Resultado XGBoost", result.message)
        return result

    def train_efficientnet_classifier(self):
        return self._train_pytorch_model(mode="classification", backbone="efficientnet")

    def train_efficientnet_regressor(self):
        return self._train_pytorch_model(mode="regression", backbone="efficientnet")

    def train_densenet_classifier(self):
        return self._train_pytorch_model(mode="classification", backbone="densenet")

    def train_densenet_regressor(self):
        return self._train_pytorch_model(mode="regression", backbone="densenet")

    def train_medicalnet_classifier(self):
        return self._train_pytorch_model(mode="classification", backbone="medicalnet")

    def train_medicalnet_regressor(self):
        return self._train_pytorch_model(mode="regression", backbone="medicalnet")

    def _train_pytorch_model(self, mode="classification", backbone="medicalnet", hparams=None):
        split_csv_path = self._resolve_split_csv_path()
        if not split_csv_path.exists():
            messagebox.showwarning("Aviso", "Crie o dataset (Criar Dataset) antes de treinar.")
            return None

        config = DeepTrainingConfig(
            split_csv_path=split_csv_path,
            output_dir=Path(self.output_dir),
            mode=mode,
            backbone=backbone,
            hyperparameters=dict(hparams or {}),
            save_experiment_fn=self._save_experiment_callback(),
            headless=self._is_headless(),
            dataset_dir=Path(self.dataset_dir),
            show_plot_window_fn=getattr(self, "_show_plot_window", None),
            plot_confusion_matrix_fn=lambda cm, classes: self._plot_confusion_figure(cm, classes),
        )

        try:
            result = train_pytorch_model(config)
        except ImportError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return None
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Erro", str(exc))
            return None

        messagebox.showinfo(backbone, result.summary_message)
        return result

    def _resolve_backbone_checkpoint(self, backbone: str) -> Path | None:
        candidates = [
            Path(self.output_dir) / f"best_{backbone}_classifier.pth",
            Path(self.output_dir) / f"{backbone}_classification.pth",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _refine_with_rl(self, backbone: str, episodes=8, horizon=4, micro_epochs=1, train_subset=120, val_subset=80):
        split_csv_path = self._resolve_split_csv_path()
        if not split_csv_path.exists():
            messagebox.showwarning("Aviso", "Crie o dataset (Criar Dataset) antes de rodar o RL.")
            return None

        checkpoint_path = self._resolve_backbone_checkpoint(backbone)
        if checkpoint_path is None:
            messagebox.showwarning("Aviso", f"Treine o backbone {backbone} antes de refinar com RL.")
            return None

        try:
            from .rl_refinement import RLRefinementConfig, refine_model_with_rl
        except ImportError as exc:
            messagebox.showerror("Dependência ausente", str(exc))
            return None

        config = RLRefinementConfig(
            checkpoint_path=checkpoint_path,
            split_csv_path=split_csv_path,
            output_dir=Path(self.output_dir),
            episodes=episodes,
            horizon=horizon,
            micro_epochs=micro_epochs,
            train_subset=train_subset,
            val_subset=val_subset,
            dataset_dir=Path(self.dataset_dir),
            backbone=backbone,
            save_experiment_fn=self._save_experiment_callback(),
        )

        try:
            result = refine_model_with_rl(config)
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
            messagebox.showerror("Erro", str(exc))
            return None

        if result.best_hyperparameters:
            try:
                self._train_pytorch_model(
                    mode="classification",
                    backbone=backbone,
                    hparams=result.best_hyperparameters,
                )
            except Exception:
                pass

        messagebox.showinfo(
            f"{backbone} + RL",
            (
                f"Best hparams: {result.best_hyperparameters}\n"
                f"Checkpoint: {result.refined_checkpoint_path.name}\n"
                f"Val balanced acc: {result.metrics.get('refined_val_balanced_accuracy', 0.0):.2%}"
            ),
        )
        return result

    def refine_efficientnet_with_rl(self, episodes=8, horizon=4, micro_epochs=1, train_subset=120, val_subset=80):
        return self._refine_with_rl(
            "efficientnet",
            episodes=episodes,
            horizon=horizon,
            micro_epochs=micro_epochs,
            train_subset=train_subset,
            val_subset=val_subset,
        )

    def refine_densenet_with_rl(self, episodes=8, horizon=4, micro_epochs=1, train_subset=120, val_subset=80):
        return self._refine_with_rl(
            "densenet",
            episodes=episodes,
            horizon=horizon,
            micro_epochs=micro_epochs,
            train_subset=train_subset,
            val_subset=val_subset,
        )
