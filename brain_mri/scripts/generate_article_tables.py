"""Generate LaTeX tables for the paper from reproducible artifacts.

Source of truth:
- output/dataset_stats.json (and its referenced split CSV)
- output/training_experiments.json

This script writes versioned LaTeX snippets under Artigo/generated/ so the paper
can \input{} them and avoid numeric drift.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "output"
ARTICLE_DIR = REPO_ROOT / "Artigo"
GENERATED_DIR = ARTICLE_DIR / "generated"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_float(value: float | None, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{value:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    return f"{100.0 * float(value):.{digits}f}\\%"


def _latex_escape(text: str) -> str:
    # Minimal escaping for LaTeX text contexts.
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _int_or_zero(value: Any) -> int:
    v = _safe_int(value)
    return 0 if v is None else v


@dataclass(frozen=True)
class ConfusionMetrics:
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def sensitivity(self) -> float:
        denom = self.tp + self.fn
        return (self.tp / denom) if denom else 0.0

    def specificity(self) -> float:
        denom = self.tn + self.fp
        return (self.tn / denom) if denom else 0.0

    def balanced_accuracy(self) -> float:
        return 0.5 * (self.sensitivity() + self.specificity())

    def collapse_label(self) -> str:
        sens = self.sensitivity()
        spec = self.specificity()
        if sens == 0.0 and spec == 1.0:
            return "Sim (tudo \\textit{Nondemented})"
        if sens == 1.0 and spec == 0.0:
            return "Sim (tudo \\textit{Demented})"
        if sens == 0.0 or spec == 0.0:
            return "Sim"
        return "Não"


def _confusion_from_entry(entry: dict[str, Any]) -> ConfusionMetrics | None:
    cm = entry.get("test_confusion_matrix")
    if not cm or not isinstance(cm, list) or len(cm) != 2:
        return None
    try:
        tn, fp = cm[0]
        fn, tp = cm[1]
        return ConfusionMetrics(int(tn), int(fp), int(fn), int(tp))
    except Exception:
        return None


def _latest_by(entries: Iterable[dict[str, Any]], *, predicate) -> dict[str, Any] | None:
    filtered = [e for e in entries if predicate(e)]
    if not filtered:
        return None
    # timestamps are stored as strings "YYYY-MM-DD HH:MM:SS"; lexicographic works.
    return sorted(filtered, key=lambda e: str(e.get("timestamp", "")))[-1]


def _sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def _write_or_check(path: Path, content: str, *, check: bool) -> None:
    if not content.endswith("\n"):
        content += "\n"
    if check:
        if not path.exists():
            raise SystemExit(f"[CHECK] Missing generated file: {path}")
        current = path.read_text(encoding="utf-8")
        if current != content:
            raise SystemExit(
                f"[CHECK] Drift detected in {path}. Re-run with --write to regenerate."
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _generate_dataset_split_table(dataset_stats: dict[str, Any]) -> str:
    split_csv_rel = dataset_stats.get("split_csv")
    split_csv = (REPO_ROOT / split_csv_rel) if split_csv_rel else (OUTPUT_DIR / "exam_level_dataset_split.csv")
    if not split_csv.exists():
        raise SystemExit(f"Split CSV not found: {split_csv}")

    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit(f"pandas is required to generate tables: {exc}")

    df = pd.read_csv(split_csv)
    group_col = "Final_Group" if "Final_Group" in df.columns else ("Group" if "Group" in df.columns else None)
    if group_col is None:
        raise SystemExit("Split CSV has no 'Final_Group' or 'Group' column.")
    if "split" not in df.columns:
        raise SystemExit("Split CSV has no 'split' column.")

    # Normalize class labels to the two-class setup used in the paper.
    def norm(v: str) -> str:
        v = str(v)
        if v.lower().startswith("non"):
            return "Nondemented"
        return "Demented"

    df = df.copy()
    df["_class"] = df[group_col].map(norm)

    counts = (
        df.groupby(["split", "_class"]).size().unstack(fill_value=0).reindex(columns=["Nondemented", "Demented"], fill_value=0)
    )
    # Ensure splits order
    split_order = ["train", "validation", "test"]
    counts = counts.reindex(split_order).fillna(0).astype(int)
    counts["Total"] = counts.sum(axis=1)

    total_row = counts.sum(axis=0)
    total_row.name = "Total"
    counts2 = counts.copy()
    counts2.loc["Total"] = total_row

    lines = []
    lines.append("% Auto-generated from output/dataset_stats.json + split CSV")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Distribuição do dataset em conjuntos de treino, validação e teste (gerado automaticamente).}")
    lines.append("\\label{tab:dataset_split}")
    lines.append("\\begin{tabular}{|l|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Conjunto} & \\textbf{Nondemented} & \\textbf{Demented} & \\textbf{Total} \\\\")
    lines.append("\\hline")
    for split in ["train", "validation", "test", "Total"]:
        row = counts2.loc[split]
        name = {"train": "Treino", "validation": "Validação", "test": "Teste", "Total": "\\textbf{Total}"}[split]
        lines.append(f"{name} & {int(row['Nondemented'])} & {int(row['Demented'])} & {int(row['Total'])} \\\\")
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _generate_dataset_macros(dataset_stats: dict[str, Any]) -> str:
    """Generate LaTeX macros for dataset-level counts.

    These macros are meant to be used in prose to avoid hardcoding numbers.
    """
    total_exams = _safe_int(dataset_stats.get("total_exams"))
    total_subjects = _safe_int(dataset_stats.get("total_subjects"))

    exams_by_split = dataset_stats.get("exams_by_split") or {}
    subjects_by_split = dataset_stats.get("subjects_by_split") or {}

    has_axl = _safe_int(dataset_stats.get("has_axl"))
    has_cor = _safe_int(dataset_stats.get("has_cor"))
    has_sag = _safe_int(dataset_stats.get("has_sag"))

    combos = dataset_stats.get("orientation_combo_counts") or {}
    combo_111 = _int_or_zero(combos.get("111", 0))  # axl+cor+sag
    combo_011 = _int_or_zero(combos.get("011", 0))  # cor+sag

    def get_split(d: dict[str, Any], key: str) -> int | None:
        return _safe_int(d.get(key))

    lines: list[str] = []
    lines.append("% Auto-generated from output/dataset_stats.json")
    lines.append("% Dataset totals")
    if total_exams is None or total_subjects is None:
        raise SystemExit("dataset_stats.json missing total_exams/total_subjects")
    lines.append(f"\\providecommand{{\\DatasetTotalExams}}{{{total_exams}}}")
    lines.append(f"\\providecommand{{\\DatasetTotalSubjects}}{{{total_subjects}}}")

    # Splits (exams)
    train_ex = get_split(exams_by_split, "train")
    val_ex = get_split(exams_by_split, "validation")
    test_ex = get_split(exams_by_split, "test")
    if train_ex is None or val_ex is None or test_ex is None:
        raise SystemExit("dataset_stats.json missing exams_by_split {train,validation,test}")
    lines.append("% Split sizes (exams)")
    lines.append(f"\\providecommand{{\\DatasetTrainExams}}{{{train_ex}}}")
    lines.append(f"\\providecommand{{\\DatasetValidationExams}}{{{val_ex}}}")
    lines.append(f"\\providecommand{{\\DatasetTestExams}}{{{test_ex}}}")

    # Splits (subjects)
    train_sub = get_split(subjects_by_split, "train")
    val_sub = get_split(subjects_by_split, "validation")
    test_sub = get_split(subjects_by_split, "test")
    if train_sub is not None and val_sub is not None and test_sub is not None:
        lines.append("% Split sizes (subjects)")
        lines.append(f"\\providecommand{{\\DatasetTrainSubjects}}{{{train_sub}}}")
        lines.append(f"\\providecommand{{\\DatasetValidationSubjects}}{{{val_sub}}}")
        lines.append(f"\\providecommand{{\\DatasetTestSubjects}}{{{test_sub}}}")

    # Orientation coverage
    lines.append("% Orientation coverage (exams)")
    lines.append(f"\\providecommand{{\\DatasetHasAxlExams}}{{{_int_or_zero(has_axl)}}}")
    lines.append(f"\\providecommand{{\\DatasetHasCorExams}}{{{_int_or_zero(has_cor)}}}")
    lines.append(f"\\providecommand{{\\DatasetHasSagExams}}{{{_int_or_zero(has_sag)}}}")

    # Orientation combinations
    # Avoid digits in control sequence names (TeX control sequences are letters-only).
    lines.append("% Orientation combinations")
    lines.append(f"\\providecommand{{\\DatasetComboAxlCorSagExams}}{{{combo_111}}}")
    lines.append(f"\\providecommand{{\\DatasetComboCorSagExams}}{{{combo_011}}}")

    return "\n".join(lines)


def _generate_classification_summary_table(experiments: list[dict[str, Any]]) -> str:
    svm_without = _latest_by(experiments, predicate=lambda e: e.get("model") == "SVM" and e.get("scenario") == "svm_without_mmse_cdr")
    svm_with = _latest_by(experiments, predicate=lambda e: e.get("model") == "SVM" and e.get("scenario") == "svm_with_mmse_cdr")
    svm_old = _latest_by(experiments, predicate=lambda e: e.get("model") == "SVM" and e.get("scenario") is None)

    eff = _latest_by(experiments, predicate=lambda e: e.get("model") == "efficientnet_classification")
    med = _latest_by(experiments, predicate=lambda e: e.get("model") == "medicalnet_classification")
    dense = _latest_by(experiments, predicate=lambda e: e.get("model") == "densenet_classification")

    rows: list[tuple[str, dict[str, Any] | None]] = [
        ("SVM (sem MMSE/CDR)", svm_without),
        ("SVM (com MMSE/CDR)", svm_with),
        ("EfficientNet-B0", eff),
        ("MedicalNet", med),
        ("DenseNet", dense),
    ]

    lines = []
    lines.append("% Auto-generated from output/training_experiments.json")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Modelo} & \\textbf{$n$} & \\textbf{Acc} & \\textbf{Bal. Acc} & \\textbf{Sens} & \\textbf{Spec} & \\textbf{F1} & \\textbf{Colapso} \\\\")
    lines.append("\\hline")

    for label, entry in rows:
        if entry is None:
            continue
        cm = _confusion_from_entry(entry)
        if cm is None:
            continue
        acc = float(entry.get("test_accuracy", 0.0))
        f1 = float(entry.get("test_f1", 0.0))
        sens = cm.sensitivity()
        spec = cm.specificity()
        bacc = cm.balanced_accuracy()
        collapse = cm.collapse_label()
        lines.append(
            f"{label} & {cm.n} & {_fmt_float(acc)} & {_fmt_float(bacc)} & {_fmt_float(sens)} & {_fmt_float(spec)} & {_fmt_float(f1)} & {collapse} \\\\"  # noqa: E501
        )
        lines.append("\\hline")

    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Resumo dos resultados de classificação (teste). $n$ = número de exames no teste. Sens = sensibilidade (\\textit{Demented}); Spec = especificidade (\\textit{Nondemented}).}"
    )
    lines.append("\\label{tab:classification_summary}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _generate_svm_confusion_tables(experiments: list[dict[str, Any]]) -> tuple[str, str]:
    svm_without = _latest_by(experiments, predicate=lambda e: e.get("model") == "SVM" and e.get("scenario") == "svm_without_mmse_cdr")
    svm_with = _latest_by(experiments, predicate=lambda e: e.get("model") == "SVM" and e.get("scenario") == "svm_with_mmse_cdr")
    if svm_without is None or svm_with is None:
        raise SystemExit("Missing SVM scenarios required for confusion tables.")

    def table_for(entry: dict[str, Any], title: str, label: str) -> str:
        cm = _confusion_from_entry(entry)
        if cm is None:
            raise SystemExit(f"Missing confusion matrix for {label}")
        lines = []
        lines.append("% Auto-generated from output/training_experiments.json")
        lines.append("\\begin{table}[h!]")
        lines.append("\\centering")
        lines.append("\\begin{tabular}{|c|c|c|}")
        lines.append("\\hline")
        lines.append(f"\\multicolumn{{3}}{{|c|}}{{\\textbf{{{title}}}}} \\\\")
        lines.append("\\hline")
        lines.append("\\textbf{Real$\\backslash$Pred} & \\textbf{Nondemented} & \\textbf{Demented} \\\\")
        lines.append("\\hline")
        lines.append(f"\\textbf{{Nondemented}} & {cm.tn} & {cm.fp} \\\\")
        lines.append("\\hline")
        lines.append(f"\\textbf{{Demented}} & {cm.fn} & {cm.tp} \\\\")
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        scenario = _latex_escape(str(entry.get("scenario", "")))
        lines.append(
            f"\\caption{{Matriz de confusão no teste (scenario \\texttt{{{scenario}}}).}}"
        )
        lines.append(f"\\label{{{label}}}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    without_tex = table_for(svm_without, "Matriz de Confusão (SVM sem MMSE/CDR, Teste)", "tab:svm_confusion_without")
    with_tex = table_for(svm_with, "Matriz de Confusão (SVM com MMSE/CDR, Teste)", "tab:svm_confusion_with")
    return without_tex, with_tex


def _generate_xgboost_regression_table(experiments: list[dict[str, Any]]) -> str:
    principal = _latest_by(
        experiments,
        predicate=lambda e: e.get("model") == "XGBoost" and e.get("scenario") == "xgb_train_and_test_current_split",
    )
    hist_val = _latest_by(experiments, predicate=lambda e: e.get("model") == "XGBoost" and e.get("scenario") is None)
    hist_eval = _latest_by(experiments, predicate=lambda e: e.get("model") == "XGBoost" and e.get("scenario") == "xgb_test_eval_existing")
    if principal is None:
        raise SystemExit("Missing principal XGBoost run: scenario=xgb_train_and_test_current_split")

    def row(label: str, e: dict[str, Any] | None) -> str | None:
        if e is None:
            return None
        return (
            f"{label} & {_fmt_float(e.get('val_mae'))} & {_fmt_float(e.get('val_rmse'))} & {_fmt_float(e.get('val_r2'))} & "
            f"{_fmt_float(e.get('test_mae'))} & {_fmt_float(e.get('test_rmse'))} & {_fmt_float(e.get('test_r2'))} \\\\"
        )

    lines = []
    lines.append("% Auto-generated from output/training_experiments.json")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\scriptsize")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Experimento} & \\textbf{Val MAE} & \\textbf{Val RMSE} & \\textbf{Val $R^2$} & \\textbf{Test MAE} & \\textbf{Test RMSE} & \\textbf{Test $R^2$} \\\\")
    lines.append("\\hline")
    lines.append(row("XGBoost (split atual)", principal))
    lines.append("\\hline")
    r = row("XGBoost (hist., apenas val)", hist_val)
    if r is not None:
        lines.append(r)
        lines.append("\\hline")
    r = row("XGBoost (hist., modelo exist.)", hist_eval)
    if r is not None:
        lines.append(r)
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Regressão de idade com XGBoost.}"
    )
    lines.append("\\label{tab:xgb_regression_summary}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _generate_pc1_embeddings_table(metrics_csv: Path) -> str:
    df = pd.read_csv(metrics_csv)
    required = {
        "embedding_space",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "train_rows",
        "test_rows",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"PC1 metrics.csv missing columns: {sorted(missing)}")

    def _pick(space: str) -> dict[str, Any] | None:
        sub = df[df["embedding_space"] == space]
        if sub.empty:
            return None
        return sub.iloc[0].to_dict()

    dl = _pick("dl")
    trad = _pick("traditional")

    def _row(label: str, entry: dict[str, Any] | None) -> str:
        if entry is None:
            return f"{label} & -- & -- & -- & -- \\\\"
        return (
            f"{label} & {int(entry.get('train_rows', 0))} & {int(entry.get('test_rows', 0))} & "
            f"{_fmt_pct(entry.get('accuracy'))} & {_fmt_pct(entry.get('balanced_accuracy'))} & {_fmt_pct(entry.get('f1'))} \\\\"
        )

    lines = []
    lines.append("% Auto-generated from output/etapa1/metrics.csv")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append("\\textbf{Espaço} & \\textbf{Treino} & \\textbf{Teste} & \\textbf{Acc} & \\textbf{Bal. Acc} & \\textbf{F1} \\\\")
    lines.append("\\hline")
    lines.append(_row("DL (embeddings)", dl))
    lines.append("\\hline")
    lines.append(_row("Tradicional (descritores)", trad))
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Comparação de um método não-profundo (regressão logística balanceada) treinado sobre embeddings DL vs. sobre descritores tradicionais.}"
    )
    lines.append("\\label{tab:pc1_embeddings}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _generate_pc2_finetune_summary(experiments: list[dict[str, Any]]) -> tuple[str, str]:
    """Generate a PC2-specific summary + a stable hash for audit.

    PC2 is expected to be executed via scripts with an explicit frozen warmup phase
    followed by unfreezing encoders (recorded in training_experiments.json).
    """

    def _is_pc2_finetune_entry(entry: dict[str, Any]) -> bool:
        # PC2 may use an overridden scenario prefix via scripts/run_pc2_finetune.py
        # (e.g., --deep-scenario). We identify the run by its fine-tuning contract
        # rather than by a hard-coded scenario prefix.
        model = str(entry.get("model") or "")
        if not model.endswith("_classification"):
            return False
        if not bool(entry.get("freeze_backbone_initial", False)):
            return False
        # The pipeline records an explicit unfreeze moment for the fine-tuning phase.
        if entry.get("unfreeze_epoch") is None:
            return False
        return True

    pc2 = _latest_by(experiments, predicate=_is_pc2_finetune_entry)
    if pc2 is None:
        raise SystemExit(
            "Missing PC2 run. Expected a fine-tuning classification entry (freeze_backbone_initial=true and unfreeze_epoch set) "
            "in output/training_experiments.json. Run: python3 brain_mri/scripts/run_pc2_finetune.py"
        )

    cm = _confusion_from_entry(pc2)
    acc = float(pc2.get("test_accuracy", 0.0))
    f1 = float(pc2.get("test_f1", 0.0))
    bacc = cm.balanced_accuracy() if cm is not None else None
    collapse = cm.collapse_label() if cm is not None else "-"

    scenario = _latex_escape(str(pc2.get("scenario", "")))
    model_raw = str(pc2.get("model", ""))
    # Mapeia nomes internos para nomes legíveis
    model_display_map = {
        "efficientnet_classification": "EfficientNet-B0",
        "medicalnet_classification": "MedicalNet",
        "densenet_classification": "DenseNet",
    }
    model = model_display_map.get(model_raw, _latex_escape(model_raw))

    pretrained = bool(pc2.get("pretrained", False))
    freeze_init = bool(pc2.get("freeze_backbone_initial", False))
    warmup_epochs = _safe_int(pc2.get("freeze_warmup_epochs"))
    unfreeze_epoch = _safe_int(pc2.get("unfreeze_epoch"))
    trainable_init = _safe_int(pc2.get("trainable_params_initial"))
    trainable_after = _safe_int(pc2.get("trainable_params_after_unfreeze"))

    def _yn(v: bool) -> str:
        return "Sim" if v else "Não"

    def _int_or_dash(v: int | None) -> str:
        return "-" if v is None else str(int(v))

    lines = []
    lines.append("% Auto-generated from output/training_experiments.json (fine-tuning run)")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{|l|c|c|c|c|c|c|}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Modelo} & \\textbf{Pré-treino} & \\textbf{Warmup cong.} & \\textbf{Libera ep.} & \\textbf{Acc} & \\textbf{Bal. Acc} & \\textbf{F1} \\\\")
    lines.append("\\hline")
    lines.append(
        f"{model} & {_yn(pretrained)} & {_int_or_dash(warmup_epochs)} & {_int_or_dash(unfreeze_epoch)} & "
        f"{_fmt_pct(acc)} & {_fmt_pct(bacc)} & {_fmt_pct(f1)} \\\\"
    )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Evidência de transfer learning e ajuste fino explícito (warmup com encoders congelados e posterior liberação) e métricas no teste.}"
    )
    lines.append("\\label{tab:pc2_finetune}")
    lines.append("\\end{table}")
    lines.append("")
    lines.append("% Audit fields (for traceability)")
    lines.append(f"% scenario={scenario}")
    lines.append(f"% freeze_backbone_initial={freeze_init}")
    lines.append(f"% trainable_params_initial={_int_or_dash(trainable_init)}")
    lines.append(f"% trainable_params_after_unfreeze={_int_or_dash(trainable_after)}")
    lines.append(f"% collapse={_latex_escape(str(collapse))}")

    entry_hash = _sha256_bytes(
        json.dumps(pc2, sort_keys=True, ensure_ascii=True).encode("utf-8")
    )
    return "\n".join(lines), entry_hash


def _generate_pc3_table(output_dir: Path) -> tuple[str, dict[str, str]]:
    comp_path = output_dir / "etapa3" / "comparativo.csv"
    hist_path = output_dir / "etapa3" / "rl_history.json"
    if not comp_path.exists() or not hist_path.exists():
        raise SystemExit(
            "Missing PC3 artifacts. Expected output/etapa3/comparativo.csv and output/etapa3/rl_history.json. "
            "Run: python3 brain_mri/scripts/run_pc3_rl_refinement.py"
        )

    df = pd.read_csv(comp_path)
    required = {
        "method",
        "budget_evaluations",
        "selection_metric",
        "selection_value",
        "val_balanced_accuracy",
        "test_balanced_accuracy",
        "test_accuracy",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"PC3 comparativo.csv missing columns: {sorted(missing)}")

    # Stable order for the table.
    method_order = ["baseline_pc2", "traditional_tuning", "rl_ppo_actor_critic"]
    df = df.copy()
    df["_order"] = df["method"].apply(lambda m: method_order.index(m) if m in method_order else 999)
    df = df.sort_values("_order").drop(columns=["_order"])

    def _name(m: str) -> str:
        return {
            "baseline_pc2": "Modelo base",
            "traditional_tuning": "Ajuste tradicional",
            "rl_ppo_actor_critic": "RL (PPO ator-crítico)",
        }.get(m, m)

    lines = []
    lines.append("% Auto-generated from output/etapa3/comparativo.csv")
    lines.append("\\begin{table}[h!]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{|l|c|c|c|c|}")
    lines.append("\\hline")
    lines.append(
        "\\textbf{Método} & \\textbf{Orçamento} & \\textbf{Seleção (val)} & \\textbf{Bal. Acc (teste)} & \\textbf{Acc (teste)} \\\\")
    lines.append("\\hline")
    for _, r in df.iterrows():
        lines.append(
            f"{_name(str(r['method']))} & {int(r['budget_evaluations'])} & "
            f"{_fmt_pct(float(r['selection_value']))} & {_fmt_pct(float(r['test_balanced_accuracy']))} & {_fmt_pct(float(r['test_accuracy']))} \\\\"
        )
        lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append(
        "\\caption{Comparação do modelo base vs. refinamento por RL (PPO ator-crítico) vs. ajuste tradicional. A seleção e a recompensa usam apenas validação; o conjunto de teste é reportado apenas após seleção.}"
    )
    lines.append("\\label{tab:pc3_comparison}")
    lines.append("\\end{table}")

    hashes = {
        "pc3_comparativo_sha256": _sha256_bytes(comp_path.read_bytes()),
        "pc3_rl_history_sha256": _sha256_bytes(hist_path.read_bytes()),
    }
    return "\n".join(lines), hashes


def generate(*, check: bool) -> None:
    dataset_stats_path = OUTPUT_DIR / "dataset_stats.json"
    experiments_path = OUTPUT_DIR / "training_experiments.json"
    if not dataset_stats_path.exists():
        raise SystemExit(f"Missing artifact: {dataset_stats_path}")
    if not experiments_path.exists():
        raise SystemExit(f"Missing artifact: {experiments_path}")

    dataset_stats = _read_json(dataset_stats_path)
    experiments = _read_json(experiments_path)
    if not isinstance(experiments, list):
        raise SystemExit("training_experiments.json must be a list of experiments")

    # Dataset split table
    ds_split_tex = _generate_dataset_split_table(dataset_stats)
    _write_or_check(GENERATED_DIR / "dataset_split_table.tex", ds_split_tex, check=check)

    # Dataset macros for prose
    ds_macros_tex = _generate_dataset_macros(dataset_stats)
    _write_or_check(GENERATED_DIR / "dataset_macros.tex", ds_macros_tex, check=check)

    # Classification summary
    class_sum_tex = _generate_classification_summary_table(experiments)
    _write_or_check(GENERATED_DIR / "classification_summary_table.tex", class_sum_tex, check=check)

    # SVM confusion matrices
    svm_without_tex, svm_with_tex = _generate_svm_confusion_tables(experiments)
    _write_or_check(GENERATED_DIR / "svm_confusion_without.tex", svm_without_tex, check=check)
    _write_or_check(GENERATED_DIR / "svm_confusion_with.tex", svm_with_tex, check=check)

    # XGBoost regression table
    xgb_tex = _generate_xgboost_regression_table(experiments)
    _write_or_check(GENERATED_DIR / "xgb_regression_summary.tex", xgb_tex, check=check)

    # PC1 embeddings comparison (optional but supported when metrics.csv exists)
    pc1_metrics_path = OUTPUT_DIR / "etapa1" / "metrics.csv"
    pc1_tex_path = GENERATED_DIR / "pc1_embeddings_metrics.tex"
    pc1_present = pc1_metrics_path.exists() or pc1_tex_path.exists()
    pc1_hash = None
    if pc1_present:
        if not pc1_metrics_path.exists():
            raise SystemExit(f"Missing artifact: {pc1_metrics_path}")
        pc1_tex = _generate_pc1_embeddings_table(pc1_metrics_path)
        _write_or_check(pc1_tex_path, pc1_tex, check=check)
        pc1_hash = _sha256_bytes(pc1_metrics_path.read_bytes())

    # PC2 fine-tuning summary (required once PC2 is closed)
    pc2_tex, pc2_hash = _generate_pc2_finetune_summary(experiments)
    _write_or_check(GENERATED_DIR / "pc2_finetune_summary.tex", pc2_tex, check=check)

    # PC3 refinement comparison (requires etapa3 artifacts)
    pc3_tex, pc3_hashes = _generate_pc3_table(OUTPUT_DIR)
    _write_or_check(GENERATED_DIR / "pc3_comparison.tex", pc3_tex, check=check)

    # Metadata (helps auditing / drift investigation)
    meta = {
        "dataset_stats_sha256": _sha256_bytes(dataset_stats_path.read_bytes()),
        "training_experiments_sha256": _sha256_bytes(experiments_path.read_bytes()),
        "pc1_metrics_sha256": pc1_hash,
        "pc2_entry_sha256": pc2_hash,
        **pc3_hashes,
    }
    _write_or_check(GENERATED_DIR / "generated_meta.json", json.dumps(meta, indent=2) + "\n", check=check)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX tables for Artigo/main.tex from JSON artifacts.")
    parser.add_argument("--check", action="store_true", help="Fail if generated outputs differ from committed files.")
    parser.add_argument("--write", action="store_true", help="Write generated files to Artigo/generated/.")
    args = parser.parse_args()

    if args.check and args.write:
        raise SystemExit("Use only one of --check or --write")
    if not args.check and not args.write:
        args.write = True

    generate(check=args.check)
    mode = "CHECK" if args.check else "WRITE"
    print(f"[OK] {mode}: generated tables under {GENERATED_DIR}")


if __name__ == "__main__":
    main()
