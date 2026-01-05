# Graphical User Interface (main.py)

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)

## Purpose & Scope

This page documents the Tkinter-based graphical user interface (GUI) provided by `main.py`, which serves as an interactive entry point for data exploration, visualization, and single-run training experiments. The GUI is designed for researchers who need to visually inspect MRI data, perform semi-automatic segmentation, and quickly prototype model configurations before committing to full-scale headless training runs.

For automated, reproducible training workflows, see [Baselines CLI](#7.2) and [Deep Models CLI](#7.3). For the underlying UI component implementations, see [Core Package Structure](#3.3).

**Sources:** [README.md L1-L218](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L1-L218)

---

## Overview

The GUI application is the primary interactive interface for the brain-mri-pipelines-py framework. It provides a visual workspace for navigating the OASIS-2 dataset, performing exploratory analysis, and configuring training experiments without requiring command-line arguments or configuration files.

### Key Capabilities

| Feature Category | Capabilities |
| --- | --- |
| **Data Navigation** | Browse MRI volumes slice-by-slice, mark non-viable studies, view patient metadata |
| **Segmentation Tools** | Semi-automatic region-growing algorithm for ventricle segmentation, morphological descriptor extraction |
| **Model Training** | Configure deep learning experiments, select backbones, adjust hyperparameters, launch single training runs |
| **Visualization** | Real-time slice rendering, segmentation overlay, basic image processing |

**Sources:** [README.md L83-L96](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L83-L96)

---

## Application Architecture

The GUI follows a modular mixin-based architecture, where specialized UI components from `brain_mri/ui/` are composed into the main application class.

```mermaid
flowchart TD

MainApp["Main Application Class (Tkinter root)"]
NavMixin["Navigation Mixin Slice browsing"]
SegMixin["Segmentation Mixin Region growing tools"]
TrainMixin["Training Mixin Model configuration"]
Models["multistream_models.py medicalnet_models.py"]
Training["Training loop logic"]
ImageProc["Image processing NIfTI loading"]
AxlDir["axl/ Required for GUI"]
CorDir["cor/ Optional"]
SagDir["sag/ Optional"]
CSV["oasis_longitudinal_ demographic.csv"]

MainApp -.-> NavMixin
MainApp -.-> SegMixin
MainApp -.-> TrainMixin
NavMixin -.-> ImageProc
SegMixin -.-> ImageProc
TrainMixin -.-> Models
TrainMixin -.-> Training
ImageProc -.-> AxlDir
ImageProc -.-> CorDir
ImageProc -.-> SagDir
NavMixin -.-> CSV

subgraph subGraph4 ["Data Layer"]
    AxlDir
    CorDir
    SagDir
    CSV
end

subgraph brain_mri/utils/ ["brain_mri/utils/"]
    ImageProc
end

subgraph brain_mri/ml/ ["brain_mri/ml/"]
    Models
    Training
end

subgraph subGraph1 ["brain_mri/ui/ Mixins"]
    NavMixin
    SegMixin
    TrainMixin
end

subgraph main.py ["main.py"]
    MainApp
end
```

**Diagram: GUI Component Architecture**

The main application inherits behavior from multiple UI mixins, each responsible for a distinct functional domain. This separation allows for modular development and testing of individual features.

**Sources:** [README.md L182-L183](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L182-L183)

---

## Launching the Application

### Prerequisites

Before launching the GUI, ensure:

1. **Tkinter is installed** on your system: * **Linux:** `sudo apt-get install python3-tk` * **macOS:** `brew install python-tk@3.11` * **Windows:** Typically bundled with Python
2. **Data is organized** with at minimum the `axl/` directory present in the repository root (see [Directory Organization](#4.3))
3. **Dependencies are installed** via `pip install -r requirements.txt`

### Starting the GUI

```
python main.py
```

This launches the Tkinter application window with the full feature set enabled.

**Sources:** [README.md L56-L77](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L56-L77)

 [README.md L83-L89](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L83-L89)

---

## Navigation Features

The navigation system allows researchers to browse through the MRI dataset, inspect individual volumes, and manage data quality annotations.

### Dataset Browser

The GUI provides controls for:

* **Subject Selection:** Dropdown or list widget to choose from available `OAS2_XXXX` subjects
* **MRI ID Selection:** For subjects with longitudinal scans (e.g., `MR1`, `MR2`), select specific timepoints
* **Plane Selection:** Switch between axial, coronal, and sagittal views (if available)
* **Slice Navigation:** Scroll through the 3D volume slice-by-slice

### Data Quality Management

Users can mark studies as **non-viable** for exclusion from training datasets. This is useful for identifying:

* Motion artifacts
* Acquisition errors
* Incomplete scans
* Other quality issues

**Sources:** [README.md L91-L92](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L91-L92)

---

## Segmentation Tools

The GUI implements a semi-automatic segmentation workflow based on **region-growing** algorithms, specifically designed for ventricle segmentation in brain MRI.

### Region-Growing Algorithm

```mermaid
flowchart TD

SeedPoint["User clicks seed point"]
Params["Set threshold parameters"]
Init["Initialize region with seed"]
Grow["Iteratively expand to neighbors"]
Check["Intensity threshold check"]
Converge["Convergence detection"]
Mask["Binary segmentation mask"]
Descriptors["Morphological descriptors"]
Export["Export results"]

SeedPoint -.-> Init
Params -.-> Check
Converge -.-> Mask

subgraph Output ["Output"]
    Mask
    Descriptors
    Export
    Mask -.-> Descriptors
    Descriptors -.-> Export
end

subgraph Algorithm ["Algorithm"]
    Init
    Grow
    Check
    Converge
    Init -.-> Grow
    Grow -.-> Check
    Check -.-> Grow
    Check -.-> Converge
end

subgraph subGraph0 ["User Interaction"]
    SeedPoint
    Params
end
```

**Diagram: Segmentation Workflow**

### Morphological Descriptor Extraction

After segmentation, the GUI computes geometric features from the ventricle regions:

* **Volume:** Total voxel count
* **Surface area:** Boundary extent
* **Shape metrics:** Circularity, elongation
* **Position:** Centroid coordinates

These descriptors feed into classical baseline models (see [Classical Machine Learning Baselines](#5.3)).

**Sources:** [README.md L93-L94](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L93-L94)

---

## Model Training Configuration

The sidebar training panel provides a graphical interface for configuring and launching deep learning experiments.

### Training Configuration Panel

The GUI exposes the following configuration options:

| Parameter | Options | Description |
| --- | --- | --- |
| **Backbone** | `efficientnet`, `densenet`, `medicalnet` | Deep learning architecture (see [Deep Learning Backbones](#5.1)) |
| **Planes** | `axl`, `cor`, `sag` (multi-select) | Which anatomical views to use |
| **Multimodal** | Checkbox | Enable fusion with clinical features |
| **Epochs** | Integer input | Training duration |
| **Batch Size** | Integer input | Samples per batch |
| **Learning Rate** | Float input | Initial optimizer learning rate |
| **Seed** | Integer input | Random seed for reproducibility |

### Execution Flow

When the user initiates training via the GUI:

```mermaid
flowchart TD

Start["User clicks 'Train Model'"]
Validate["Validate configuration Check data availability"]
BuildDataset["Initialize DataLoaders Apply subject-level split"]
InitModel["Instantiate model from multistream_models.py"]
TrainLoop["Execute training loop with progress updates"]
SaveArtifacts["Save model checkpoint Log metrics to output/"]
Display["Display results in GUI"]
ErrorMsg["Show error dialog"]

Start -.->|"Valid"| Validate
Validate -.-> BuildDataset
Validate -.->|"Invalid"| ErrorMsg
BuildDataset -.-> InitModel
InitModel -.-> TrainLoop
TrainLoop -.-> SaveArtifacts
SaveArtifacts -.-> Display
```

**Diagram: GUI Training Execution Flow**

The training process runs in the main thread with periodic UI updates to display progress. For long-running experiments, the CLI interfaces are recommended.

**Sources:** [README.md L95-L96](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L95-L96)

---

## Integration with Core Package

The GUI serves as a lightweight orchestrator that delegates specialized tasks to the core `brain_mri/` package modules.

### Module Dependencies

```mermaid
flowchart TD

GUIApp["GUI Application"]
NavMixin["Navigation Mixin"]
SegMixin["Segmentation Mixin"]
TrainMixin["Training Configuration"]
MultiStream["multistream_models.py MultiStreamClassifier"]
MedicalNet["medicalnet_models.py Med3D integration"]
Training["Training utilities"]
ExperimentTracker["Experiment logging Metrics tracking"]
ImageUtils["NIfTI loading Preprocessing"]
DataUtils["CSV parsing Subject extraction"]

GUIApp -.-> NavMixin
GUIApp -.-> SegMixin
GUIApp -.-> TrainMixin
NavMixin -.-> ImageUtils
NavMixin -.-> DataUtils
SegMixin -.-> ImageUtils
TrainMixin -.-> MultiStream
TrainMixin -.-> MedicalNet
TrainMixin -.-> Training
TrainMixin -.-> ExperimentTracker

subgraph brain_mri/utils/ ["brain_mri/utils/"]
    ImageUtils
    DataUtils
end

subgraph brain_mri/experiments/ ["brain_mri/experiments/"]
    ExperimentTracker
end

subgraph brain_mri/ml/ ["brain_mri/ml/"]
    MultiStream
    MedicalNet
    Training
end

subgraph brain_mri/ui/ ["brain_mri/ui/"]
    NavMixin
    SegMixin
    TrainMixin
end

subgraph main.py ["main.py"]
    GUIApp
end
```

**Diagram: GUI Integration with brain_mri Package**

This architecture ensures that the GUI remains a thin presentation layer, with all business logic residing in testable, reusable modules.

**Sources:** [README.md L180-L196](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L180-L196)

---

## Data Requirements

The GUI has specific data directory requirements that differ slightly from the CLI interfaces.

### Required Directory

The **`axl/` directory is mandatory** for GUI operation. The application uses axial slices as the default view for navigation and visualization.

### Optional Directories

If `cor/` and `sag/` directories are present, the GUI enables multi-plane viewing and multi-stream model training. Without these directories, only single-stream models using axial data can be trained.

### File Format Support

While the deep learning pipelines require NIfTI format (`.nii` or `.nii.gz`), the GUI may support additional formats for visualization:

* `.png`
* `.jpg`

All files must follow the naming convention: `OAS2_XXXX_MRY_plane.ext`

**Sources:** [README.md L29-L50](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L29-L50)

---

## Comparison with CLI Interfaces

The GUI and CLI interfaces serve complementary purposes in the workflow:

| Aspect | GUI (`main.py`) | Baselines CLI | Deep Models CLI |
| --- | --- | --- | --- |
| **Use Case** | Exploration, prototyping, visualization | Classical ML experiments | Full-scale deep learning |
| **Reproducibility** | Single-run, interactive | Fully scriptable | Fully scriptable |
| **Configuration** | Point-and-click | Command-line arguments | Command-line arguments |
| **Progress Tracking** | Real-time UI updates | Console output | Console output + logs |
| **Batch Experiments** | Not supported | Supported | Supported |
| **Segmentation** | Interactive region-growing | N/A | N/A |
| **Typical Duration** | Minutes (quick experiments) | Hours (SVM/XGBoost) | Hours to days (deep learning) |

### Recommended Workflow

1. **Use GUI for:** Initial data exploration, quality assessment, segmentation tasks, hyperparameter prototyping
2. **Use CLI for:** Final experiments, hyperparameter sweeps, multi-seed runs, automated pipelines

**Sources:** [README.md L83-L119](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L83-L119)

---

## Limitations & Considerations

### Single-Threading

The GUI runs training in the main thread, which means the interface becomes unresponsive during model training. For experiments longer than a few minutes, use the CLI interfaces instead.

### Limited Batch Operations

Unlike the CLI, the GUI does not support:

* Multi-seed runs
* Cross-validation folds
* Hyperparameter grid search
* Automatic result aggregation

### Memory Constraints

Loading and displaying large MRI volumes in the GUI can consume significant memory. The application loads entire volumes into memory for smooth slice navigation.

### Platform Dependencies

Tkinter behavior and appearance vary across operating systems. The GUI has been tested on:

* Ubuntu 20.04+ with `python3-tk`
* macOS 11+ with `python-tk@3.11`
* Windows 10+ with bundled Tkinter

**Sources:** [README.md L56-L62](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L56-L62)

---

## Technical Implementation Notes

### Mixin-Based Architecture

The GUI leverages Python's multiple inheritance to compose functionality from specialized mixins located in `brain_mri/ui/`. This design pattern allows:

* **Separation of Concerns:** Each mixin handles one domain (navigation, segmentation, training)
* **Reusability:** Mixins can be tested independently
* **Extensibility:** New features can be added as additional mixins

Example mixin structure referenced in the codebase:

* **Navigation Mixin:** Handles file browsing, slice selection, metadata display
* **Segmentation Mixin:** Implements region-growing, descriptor calculation, visualization
* **Training Mixin:** Manages model configuration, training execution, result display

### Event-Driven Updates

The GUI uses Tkinter's event loop to handle user interactions. Key events include:

* Slice navigation (keyboard/mouse)
* Seed point selection (mouse clicks)
* Training progress updates (periodic polling)

**Sources:** [README.md L182-L183](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L182-L183)

---

## Summary

The `main.py` GUI provides an essential interactive layer for the brain-mri-pipelines-py framework. It excels at exploratory tasks—data inspection, quality control, segmentation—and enables rapid prototyping of model configurations. However, for production experiments requiring reproducibility and batch processing, researchers should transition to the [Baselines CLI](#7.2) and [Deep Models CLI](#7.3) interfaces.

The mixin-based architecture ensures that GUI-specific code remains isolated from core machine learning logic, maintaining the framework's modularity and testability.

**Sources:** [README.md L1-L218](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L1-L218)

Refresh this wiki

Last indexed: 5 January 2026 ([cd9d51](https://github.com/ThalesMMS/brain-mri-pipelines-py/commit/cd9d51a5))

### On this page

* [Graphical User Interface (main.py)](#7.1-graphical-user-interface-mainpy)
* [Purpose & Scope](#7.1-purpose-scope)
* [Overview](#7.1-overview)
* [Key Capabilities](#7.1-key-capabilities)
* [Application Architecture](#7.1-application-architecture)
* [Launching the Application](#7.1-launching-the-application)
* [Prerequisites](#7.1-prerequisites)
* [Starting the GUI](#7.1-starting-the-gui)
* [Navigation Features](#7.1-navigation-features)
* [Dataset Browser](#7.1-dataset-browser)
* [Data Quality Management](#7.1-data-quality-management)
* [Segmentation Tools](#7.1-segmentation-tools)
* [Region-Growing Algorithm](#7.1-region-growing-algorithm)
* [Morphological Descriptor Extraction](#7.1-morphological-descriptor-extraction)
* [Model Training Configuration](#7.1-model-training-configuration)
* [Training Configuration Panel](#7.1-training-configuration-panel)
* [Execution Flow](#7.1-execution-flow)
* [Integration with Core Package](#7.1-integration-with-core-package)
* [Module Dependencies](#7.1-module-dependencies)
* [Data Requirements](#7.1-data-requirements)
* [Required Directory](#7.1-required-directory)
* [Optional Directories](#7.1-optional-directories)
* [File Format Support](#7.1-file-format-support)
* [Comparison with CLI Interfaces](#7.1-comparison-with-cli-interfaces)
* [Recommended Workflow](#7.1-recommended-workflow)
* [Limitations & Considerations](#7.1-limitations-considerations)
* [Single-Threading](#7.1-single-threading)
* [Limited Batch Operations](#7.1-limited-batch-operations)
* [Memory Constraints](#7.1-memory-constraints)
* [Platform Dependencies](#7.1-platform-dependencies)
* [Technical Implementation Notes](#7.1-technical-implementation-notes)
* [Mixin-Based Architecture](#7.1-mixin-based-architecture)
* [Event-Driven Updates](#7.1-event-driven-updates)
* [Summary](#7.1-summary)

Ask Devin about brain-mri-pipelines-py