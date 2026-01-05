# System Architecture

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)

## Purpose & Scope

This document presents the high-level architecture of the brain-mri-pipelines-py system, describing the organization of major components, their interactions, and the flow of data through the pipeline. It covers the system's layered design, entry points, core package structure, and integration patterns.

For detailed information about specific architectural patterns, see:

* Multi-stream multimodal network design: [#3.1](#3.1)
* Data processing and transformation pipeline: [#3.2](#3.2)
* Package organization and module responsibilities: [#3.3](#3.3)
* Subject-level data splitting implementation: [#3.4](#3.4)

For model-specific details, training procedures, and the three-stage research workflow, see [Models & Training](#5) and [Three-Stage Research Pipeline](#6).

## System Layers

The system is organized into four distinct layers that separate concerns and enable modular development:

```mermaid
flowchart TD

MAIN["main.py BrainMRIApp"]
CLI_BASE["run_baselines_cli.py"]
CLI_DEEP["run_deep_models_cli.py"]
PC1["run_pc1_embeddings.py"]
PC2["run_pc2_finetune.py"]
PC3["run_pc3_rl_refinement.py"]
TABLES["generate_article_tables"]
UI_PKG["ui/ NavigationMixin SegmentationMixin"]
ML_PKG["ml/ models, training, rl_refinement"]
EXP_PKG["experiments/ ExperimentTracker ResultsPlotter"]
UTILS_PKG["utils/ image processing"]
IMG["axl/, cor/, sag/ OAS2_*.nii.gz"]
CSV["oasis_longitudinal_ demographic.csv"]
OUT["output/ models/, logs/"]

MAIN -.-> UI_PKG
MAIN -.-> ML_PKG
PC1 -.-> ML_PKG
PC2 -.-> ML_PKG
PC3 -.-> ML_PKG
TABLES -.-> OUT

subgraph DATA ["Data Layer"]
    IMG
    CSV
    OUT
end

subgraph CORE ["Core Package: brain_mri/"]
    UI_PKG
    ML_PKG
    EXP_PKG
    UTILS_PKG
end

subgraph RESEARCH ["Research Pipeline Layer"]
    PC1
    PC2
    PC3
    TABLES
end

subgraph UI ["User Interface Layer"]
    MAIN
    CLI_BASE
    CLI_DEEP
end
```

**Sources:** [README.md L1-L218](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L1-L218)

### Layer Responsibilities

| Layer | Purpose | Key Components |
| --- | --- | --- |
| **User Interface** | Provides multiple access patterns for users | `main.py`, `run_baselines_cli.py`, `run_deep_models_cli.py` |
| **Research Pipeline** | Implements three-stage experimental workflow | `run_pc1_embeddings.py`, `run_pc2_finetune.py`, `run_pc3_rl_refinement.py` |
| **Core Package** | Reusable modules for ML, UI, and utilities | `brain_mri/ml/`, `brain_mri/ui/`, `brain_mri/experiments/`, `brain_mri/utils/` |
| **Data** | Input datasets and output artifacts | `axl/`, `cor/`, `sag/`, CSV files, `output/` directory |

**Sources:** [README.md L177-L196](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L196)

## Entry Points

The system provides three main entry points, each designed for different use cases:

```mermaid
flowchart TD

GUI["main.py"]
CLI_B["run_baselines_cli.py"]
CLI_D["run_deep_models_cli.py"]
EXPLORE["Data Exploration & Visualization"]
SEGMENT["Semi-automatic Segmentation"]
BASELINE["Classical ML Training"]
DEEP["Deep Learning Training"]
QUICK["Quick Single Experiments"]
BATCH["Batch Reproducible Experiments"]

GUI -.-> EXPLORE
GUI -.-> SEGMENT
GUI -.-> QUICK

subgraph TASKS ["Primary Tasks"]
    EXPLORE
    SEGMENT
    BASELINE
    DEEP
    QUICK
    BATCH
end

subgraph ENTRY ["Entry Points"]
    GUI
    CLI_B
    CLI_D
end
```

**Sources:** [README.md L83-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L83-L118)

### main.py - Interactive GUI

The graphical user interface is implemented in `main.py` using Tkinter. It instantiates the `BrainMRIApp` class which inherits from multiple mixins for modular functionality:

* **Navigation:** Browse MRI volumes, select subjects, mark non-viable studies
* **Segmentation:** Region-growing ventricle segmentation, descriptor extraction
* **Training:** Configure and execute single training runs
* **Visualization:** Slice viewing, overlay rendering, plot display

**Primary Use Cases:**

* Initial dataset exploration
* Visual quality assessment
* Manual segmentation tasks
* Rapid prototyping of model configurations

**Sources:** [README.md L83-L96](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L83-L96)

### run_baselines_cli.py - Classical ML

Command-line interface for training classical machine learning models. This script:

1. Generates subject-aware train/validation/test split CSV
2. Trains SVM models with morphological descriptors
3. Trains XGBoost for age estimation
4. Tests two scenarios: with and without MMSE/CDR scores (to analyze target proxy leakage)

**Primary Use Cases:**

* Baseline performance benchmarking
* Reproducible headless execution
* Target leakage analysis

**Sources:** [README.md L98-L109](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L98-L109)

### run_deep_models_cli.py - Deep Learning

Command-line interface for training deep learning models. Supports:

* Multiple backbones: `efficientnet`, `densenet`, `medicalnet`
* Multi-stream configuration (axial, coronal, sagittal planes)
* Multimodal fusion with clinical features
* Configurable hyperparameters via command-line arguments

Example invocations:

```
python run_deep_models_cli.py --seed 42 --epochs 40 --backbones efficientnet,medicalnet,densenetpython run_deep_models_cli.py --seed 42 --epochs 40 --backbones efficientnet --multimodal
```

**Sources:** [README.md L110-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L110-L118)

## Core Package Structure

The `brain_mri/` package contains the reusable modules that implement core functionality:

```mermaid
flowchart TD

S1["run_pc1_embeddings.py"]
NAV["NavigationMixin"]
IMG["image_processing.py"]
SEG["SegmentationMixin"]
TRAIN["training_loop.py"]
MODELS["medicalnet_models.py multistream_models.py model_definitions.py"]
DATA_ML["data_loader.py"]
TRACK["ExperimentTracker"]
RL["rl_refinement.py PPOAgent"]
S3["run_pc3_rl_refinement.py"]
S2["run_pc2_finetune.py"]
S4["generate_article_tables"]
META["metadata.py"]
PLOT["ResultsPlotter"]

subgraph PKG ["brain_mri/"]
    NAV -.-> IMG
    SEG -.-> IMG
    TRAIN -.-> TRACK
    S3 -.-> RL
    S1 -.-> ML
    S2 -.-> ML
    S4 -.-> EXP

subgraph SCRIPTS ["scripts/"]
    S1
    S3
    S2
    S4
end

subgraph UTILS ["utils/"]
    IMG
    META
end

subgraph EXP ["experiments/"]
    TRACK
    PLOT
end

subgraph ML ["ml/"]
    TRAIN
    MODELS
    DATA_ML
    RL
    TRAIN -.-> MODELS
    TRAIN -.-> DATA_ML
    RL -.-> TRAIN
end

subgraph UI ["ui/"]
    NAV
    SEG
end
end
```

**Sources:** [README.md L177-L196](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L196)

### Module Responsibilities

| Module | Key Components | Purpose |
| --- | --- | --- |
| `ui/` | `NavigationMixin`, `SegmentationMixin` | Tkinter GUI functionality split into mixins |
| `ml/` | Model definitions, training loops, RL agent | Core machine learning implementations |
| `experiments/` | `ExperimentTracker`, `ResultsPlotter` | Experiment logging, metrics tracking, visualization |
| `utils/` | Image processing, metadata parsing | Utility functions for data handling |
| `scripts/` | Stage-specific runners | Three-stage research pipeline implementations |

**Sources:** [README.md L177-L196](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L196)

### Key Files in brain_mri/ml/

```mermaid
flowchart TD

MED["medicalnet_models.py ResNet10/18/34/50/101/152/200 3D→2D conversion"]
MULTI["multistream_models.py MultiStreamModel MultimodalModel"]
DEF["model_definitions.py EfficientNet, DenseNet wrapper factories"]
TRAIN["training_loop.py train_model() evaluate_model()"]
RL_FILE["rl_refinement.py PPOAgent ActorCritic"]
DATA["data_loader.py create_data_loaders() WeightedRandomSampler"]

subgraph ML_DIR ["brain_mri/ml/"]
    MED
    MULTI
    DEF
    TRAIN
    RL_FILE
    DATA
    MULTI -.-> MED
    MULTI -.-> DEF
    TRAIN -.-> MULTI
    TRAIN -.-> DATA
end
```

**Sources:** [README.md L184-L188](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L184-L188)

The `ml/` directory centralizes all machine learning logic:

* **medicalnet_models.py:** Implements ResNet architectures with Med3D weight loading and mathematical 3D→2D kernel conversion
* **multistream_models.py:** Implements multi-view fusion logic, combining embeddings from multiple anatomical planes
* **model_definitions.py:** Factory functions for creating EfficientNet and DenseNet backbone instances
* **training_loop.py:** Core training and evaluation functions with class imbalance handling
* **rl_refinement.py:** PPO-based reinforcement learning agent for hyperparameter optimization
* **data_loader.py:** PyTorch DataLoader creation with subject-aware splitting

**Sources:** [README.md L184-L188](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L184-L188)

## Data Flow Architecture

The following diagram illustrates how data flows from raw files through processing stages to final outputs:

```mermaid
flowchart TD

NII["NIfTI Files axl/OAS2_XXXX_MRY_axl.nii.gz cor/OAS2_XXXX_MRY_cor.nii.gz sag/OAS2_XXXX_MRY_sag.nii.gz"]
CSV_IN["oasis_longitudinal_ demographic.csv"]
FNAME["Filename Parser Extract Subject_ID, MRI_ID"]
META["Metadata Loader Parse clinical features"]
VAL["Data Validator"]
SPLITTER["Subject-Aware Splitter group by Subject_ID prevent leakage"]
TRAIN_SET["Train Set"]
VAL_SET["Validation Set"]
TEST_SET["Test Set"]
DATASET["PyTorch Dataset"]
SAMPLER["WeightedRandomSampler handle class imbalance"]
LOADER["DataLoader"]
AUG["Augmentation rotation, flip, noise"]
BATCH["Batch Assembly multi-view + clinical"]
MODEL["Model Forward Pass"]
WEIGHTS["Trained Model Weights output/models/"]
LOGS["Training Logs output/logs/"]
PLOTS["Visualizations output/plots/"]

NII -.-> FNAME
VAL -.-> SPLITTER
LOADER -.-> AUG
MODEL -.-> WEIGHTS
MODEL -.-> LOGS
MODEL -.-> PLOTS

subgraph OUTPUT ["Outputs"]
    WEIGHTS
    LOGS
    PLOTS
end

subgraph PROCESS ["Processing & Training"]
    AUG
    BATCH
    MODEL
    AUG -.-> BATCH
    BATCH -.-> MODEL
end

subgraph LOAD ["Data Loading"]
    DATASET
    SAMPLER
    LOADER
    DATASET -.-> SAMPLER
    SAMPLER -.-> LOADER
end

subgraph SPLIT ["Subject-Level Splitting"]
    SPLITTER
    TRAIN_SET
    VAL_SET
    TEST_SET
    SPLITTER -.-> TRAIN_SET
    SPLITTER -.-> VAL_SET
    SPLITTER -.-> TEST_SET
end

subgraph PARSE ["Parsing & Validation"]
    FNAME
    META
    VAL
    FNAME -.-> VAL
    META -.-> VAL
end

subgraph INPUT ["Input Sources"]
    NII
    CSV_IN
end
```

**Sources:** [README.md L27-L50](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L27-L50)

### Critical: Subject-Level Splitting

The system enforces **subject-level splitting** to prevent data leakage. The filename pattern `OAS2_XXXX_MRY_plane.nii.gz` is parsed to extract:

* **Subject_ID:** `OAS2_XXXX` (e.g., `OAS2_0001`)
* **MRI_ID:** `OAS2_XXXX_MRY` (e.g., `OAS2_0001_MR1`)

All scans from a single `Subject_ID` (which may include multiple `MRI_ID` values representing different timepoints) remain strictly within one partition (Train, Validation, or Test). This prevents the common pitfall where different timepoint scans from the same patient leak across splits.

**Sources:** [README.md L40-L49](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L40-L49)

 [README.md L160-L174](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L160-L174)

## Component Interaction Patterns

### Training Pipeline Integration

The following diagram shows how components interact during a typical training workflow:

```mermaid
flowchart TD

CLI["CLI Entry Point run_deep_models_cli.py"]
PARSE_ARGS["Argument Parser seed, epochs, backbones"]
SPLIT_GEN["generate_split_csv() subject-level splitting"]
LOADER_CREATE["create_data_loaders() WeightedRandomSampler"]
BACKBONE_SEL["Backbone Selection efficientnet/densenet/medicalnet"]
MULTI_BUILD["MultiStreamModel() or MultimodalModel()"]
WEIGHTS_LOAD["Load Pretrained Weights ImageNet or Med3D"]
TRAIN_FUNC["train_model() from training_loop.py"]
FORWARD["Forward Pass multi-view fusion"]
LOSS_CALC["Loss Computation class-weighted, Focal"]
BACKWARD["Backward Pass optimizer.step()"]
EVAL_FUNC["evaluate_model() balanced accuracy"]
TRACKER["ExperimentTracker log metrics"]
PLOTTER["ResultsPlotter generate plots"]
SAVE_MODEL["Save checkpoint output/models/"]
SAVE_LOG["Save metrics output/logs/"]
SAVE_PLOT["Save figures output/plots/"]

CLI -.-> PARSE_ARGS
TRACKER -.-> SAVE_LOG
PLOTTER -.-> SAVE_PLOT

subgraph PERSIST ["Persistence"]
    SAVE_MODEL
    SAVE_LOG
    SAVE_PLOT
end

subgraph TRACKING ["Experiment Tracking"]
    TRACKER
    PLOTTER
    TRACKER -.-> PLOTTER
end

subgraph TRAINING ["Training Loop"]
    TRAIN_FUNC
    FORWARD
    LOSS_CALC
    BACKWARD
    EVAL_FUNC
    FORWARD -.-> LOSS_CALC
    BACKWARD -.-> EVAL_FUNC
end

subgraph MODEL_BUILD ["Model Construction"]
    BACKBONE_SEL
    MULTI_BUILD
    WEIGHTS_LOAD
end

subgraph SETUP ["Setup Phase"]
    PARSE_ARGS
    SPLIT_GEN
    LOADER_CREATE
end
```

**Sources:** [README.md L110-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L110-L118)

### RL Refinement Integration

The reinforcement learning refinement stage introduces a PPO agent that wraps the training process:

```mermaid
flowchart TD

PC3["run_pc3_rl_refinement.py"]
LOAD_BASE["Load Fine-tuned Model from Stage 2"]
INIT_PPO["Initialize PPOAgent ActorCritic network"]
INIT_STATE["Initial State: current hyperparameters"]
POLICY["Policy Network select action (Δlr, Δwd)"]
APPLY["Apply Hyperparameter Adjustment"]
MICRO_TRAIN["Micro-Epoch Training train_model() for N steps"]
REWARD["Compute Reward validation balanced accuracy"]
STORE["Store Transition (state, action, reward)"]
UPDATE_PPO["PPO Update actor & critic networks"]
SAVE_RL["Save RL-Optimized Model"]

PC3 -.-> LOAD_BASE

subgraph RL_LOOP ["RL Episode Loop"]
    POLICY
    APPLY
    MICRO_TRAIN
    REWARD
    STORE
    UPDATE_PPO
    POLICY -.-> APPLY
    APPLY -.-> MICRO_TRAIN
    REWARD -.-> STORE
    STORE -.-> UPDATE_PPO
end

subgraph RL_SETUP ["RL Setup"]
    LOAD_BASE
    INIT_PPO
    INIT_STATE
end
```

**Sources:** [README.md L142-L148](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L142-L148)

 [README.md L17-L18](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L17-L18)

The PPO agent treats hyperparameters (learning rate, weight decay) as actions that can be adjusted per micro-epoch. The validation balanced accuracy serves as the reward signal, creating an iterative optimization loop that refines the model beyond traditional hyperparameter tuning.

## Output Artifacts

The `output/` directory stores all generated artifacts:

| Subdirectory | Contents | Generated By |
| --- | --- | --- |
| `output/models/` | Trained model weights (`.pth` files) | `training_loop.py`, `rl_refinement.py` |
| `output/logs/` | Training metrics, experiment configs | `ExperimentTracker` |
| `output/plots/` | Loss curves, confusion matrices, ROC curves | `ResultsPlotter` |
| `output/tables/` | LaTeX tables for publication | `generate_article_tables` |

**Sources:** [README.md L37](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L37-L37)

## System Extensibility

The architecture supports extension through several mechanisms:

1. **New Backbones:** Add model definitions to `brain_mri/ml/model_definitions.py` and register in `multistream_models.py`
2. **New Augmentations:** Extend augmentation pipeline in `data_loader.py`
3. **New Metrics:** Add metric computation to `evaluate_model()` in `training_loop.py`
4. **New RL Algorithms:** Replace `PPOAgent` in `rl_refinement.py` with alternative RL implementations
5. **New CLI Tools:** Create new scripts in `brain_mri/scripts/` following the PC1/PC2/PC3 pattern

**Sources:** [README.md L177-L196](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L196)

Refresh this wiki

Last indexed: 5 January 2026 ([cd9d51](https://github.com/ThalesMMS/brain-mri-pipelines-py/commit/cd9d51a5))

### On this page

* [System Architecture](#3-system-architecture)
* [Purpose & Scope](#3-purpose-scope)
* [System Layers](#3-system-layers)
* [Layer Responsibilities](#3-layer-responsibilities)
* [Entry Points](#3-entry-points)
* [main.py - Interactive GUI](#3-mainpy---interactive-gui)
* [run_baselines_cli.py - Classical ML](#3-run_baselines_clipy---classical-ml)
* [run_deep_models_cli.py - Deep Learning](#3-run_deep_models_clipy---deep-learning)
* [Core Package Structure](#3-core-package-structure)
* [Module Responsibilities](#3-module-responsibilities)
* [Key Files in brain_mri/ml/](#3-key-files-in-brain_mriml)
* [Data Flow Architecture](#3-data-flow-architecture)
* [Critical: Subject-Level Splitting](#3-critical-subject-level-splitting)
* [Component Interaction Patterns](#3-component-interaction-patterns)
* [Training Pipeline Integration](#3-training-pipeline-integration)
* [RL Refinement Integration](#3-rl-refinement-integration)
* [Output Artifacts](#3-output-artifacts)
* [System Extensibility](#3-system-extensibility)

Ask Devin about brain-mri-pipelines-py