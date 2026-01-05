# Data Layer

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)
> * [axl/OAS2_0001_MR1_axl.nii.gz](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0001_MR1_axl.nii.gz)

This document provides comprehensive technical documentation of the data layer in the brain-mri-pipelines-py framework. The data layer encompasses all aspects of data storage, organization, loading, and preprocessing for the OASIS-2 neuroimaging dataset.

**Scope**: This page covers the physical organization of MRI images and clinical metadata, file formats, naming conventions, and the data loading pipeline. For information about the subject-level splitting mechanism that prevents data leakage, see [Subject-Level Splitting & Leakage Prevention](#3.4). For model training configurations, see [Training Configuration](#5.4).

**Child Pages**: This section has several specialized sub-pages:

* [OASIS-2 Dataset Overview](#4.1) - Dataset structure and demographics
* [NIfTI File Format](#4.2) - Neuroimaging file format specifications
* [Directory Organization & File Naming](#4.3) - Physical layout and naming patterns
* [Clinical Metadata](#4.4) - Tabular clinical features
* [Data Loading & Augmentation](#4.5) - Loading pipeline and transformations

---

## Dataset Organization

The framework expects a specific directory structure in the repository root, with clear separation between input data and generated outputs.

### Directory Structure

```mermaid
flowchart TD

ROOT["Repository Root"]
AXL["axl/ Axial plane images Required"]
COR["cor/ Coronal plane images Optional"]
SAG["sag/ Sagittal plane images Optional"]
CSV["oasis_longitudinal_ demographic.csv Clinical metadata"]
OUT["output/ Models, logs, plots"]
SPLIT["subject_splits.csv Train/Val/Test assignments"]

ROOT -.-> AXL
ROOT -.-> COR
ROOT -.-> SAG
ROOT -.-> CSV
ROOT -.-> OUT

subgraph subGraph1 ["Generated Artifacts"]
    OUT
    SPLIT
    OUT -.-> SPLIT
end

subgraph subGraph0 ["Input Data"]
    AXL
    COR
    SAG
    CSV
end
```

**Sources**: [README.md L27-L38](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L27-L38)

| Directory | Purpose | Required | Content Type |
| --- | --- | --- | --- |
| `axl/` | Axial plane MRI scans | Yes (for GUI) | NIfTI files (`.nii`, `.nii.gz`) |
| `cor/` | Coronal plane MRI scans | No | NIfTI files (`.nii`, `.nii.gz`) |
| `sag/` | Sagittal plane MRI scans | No | NIfTI files (`.nii`, `.nii.gz`) |
| `oasis_longitudinal_demographic.csv` | Clinical and demographic data | Yes | CSV with subject-level records |
| `output/` | Training artifacts and results | Auto-generated | Models, logs, plots, tables |

---

## File Naming Convention

The system uses a strict naming convention to extract subject identifiers and MRI session information from filenames.

### Naming Pattern

Files follow the pattern: `OAS2_XXXX_MRY_plane.nii.gz`

**Components**:

* `OAS2`: Dataset identifier (OASIS-2)
* `XXXX`: 4-digit subject number (e.g., `0001`, `0002`)
* `MR`: Literal string indicating MRI scan
* `Y`: Session/timepoint number (e.g., `1`, `2`)
* `plane`: Anatomical plane (`axl`, `cor`, or `sag`)
* Extension: `.nii.gz` (compressed) or `.nii` (uncompressed)

### Identifier Extraction Diagram

```mermaid
flowchart TD

FILE["OAS2_0001_MR1_axl.nii.gz"]
PARSE["Filename Parser"]
SUBJ["Subject_ID: OAS2_0001"]
MRI["MRI_ID: OAS2_0001_MR1"]
PLANE["Plane: axl"]
SPLIT["Subject-Level Splitting"]
DATALOADER["DataLoader Sample Selection"]
STREAM["Multi-Stream Architecture"]

FILE -.-> PARSE
PARSE -.-> SUBJ
PARSE -.-> MRI
PARSE -.-> PLANE
SUBJ -.-> SPLIT
MRI -.-> DATALOADER
PLANE -.-> STREAM
```

**Sources**: [README.md L40-L49](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L40-L49)

### Examples

| Filename | Subject_ID | MRI_ID | Plane | Session |
| --- | --- | --- | --- | --- |
| `OAS2_0001_MR1_axl.nii.gz` | `OAS2_0001` | `OAS2_0001_MR1` | axl | 1 |
| `OAS2_0001_MR2_axl.nii.gz` | `OAS2_0001` | `OAS2_0001_MR2` | axl | 2 |
| `OAS2_0002_MR1_cor.nii.gz` | `OAS2_0002` | `OAS2_0002_MR1` | cor | 1 |
| `OAS2_0002_MR1_sag.nii.gz` | `OAS2_0002` | `OAS2_0002_MR1` | sag | 1 |

**Critical Note**: The `Subject_ID` (e.g., `OAS2_0001`) is used for train/validation/test splitting to prevent data leakage. Multiple scans from the same subject (`OAS2_0001_MR1`, `OAS2_0001_MR2`) must remain in the same partition.

**Sources**: [README.md L40-L49](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L40-L49)

 High-level Diagram 4

---

## NIfTI File Format

### Format Overview

The framework processes neuroimaging data stored in the NIfTI-1 format (Neuroimaging Informatics Technology Initiative). NIfTI files contain both image data and metadata in a single file.

**File Extensions**:

* `.nii` - Uncompressed NIfTI
* `.nii.gz` - Gzip-compressed NIfTI (preferred for storage efficiency)

### NIfTI Structure

```mermaid
flowchart TD

NIFTI["NIfTI File (.nii or .nii.gz)"]
HEADER["Header (348 bytes) Metadata & Dimensions"]
VOXELS["Voxel Data 3D/4D Image Array"]
DIM["dim[8] Array dimensions"]
PIXDIM["pixdim[8] Voxel sizes (mm)"]
DATATYPE["datatype Data type code"]
QFORM["qform_code Coordinate system"]
SFORM["sform_code Affine transform"]

NIFTI -.-> HEADER
NIFTI -.-> VOXELS
HEADER -.-> DIM
HEADER -.-> PIXDIM
HEADER -.-> DATATYPE
HEADER -.-> QFORM
HEADER -.-> SFORM

subgraph subGraph1 ["Header Fields"]
    DIM
    PIXDIM
    DATATYPE
    QFORM
    SFORM
end

subgraph subGraph0 ["File Components"]
    HEADER
    VOXELS
end
```

**Sources**: [axl/OAS2_0001_MR1_axl.nii.gz L1-L100](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0001_MR1_axl.nii.gz#L1-L100)

 (binary file structure inferred)

### Coordinate Systems

NIfTI files store affine transformations that map voxel indices to physical coordinates (typically in mm). The framework relies on these transformations for spatial consistency across scans.

**Key Coordinate Spaces**:

* **Voxel Space**: Integer indices (i, j, k) into the 3D array
* **Scanner Space**: Physical coordinates (x, y, z) in millimeters relative to the scanner isocenter
* **Anatomical Space**: Standard orientation (e.g., RAS: Right-Anterior-Superior)

---

## Clinical Metadata

The clinical metadata file `oasis_longitudinal_demographic.csv` contains per-subject demographic and morphometric features.

### Metadata Schema

```mermaid
flowchart TD

CSV["oasis_longitudinal_ demographic.csv"]
SUBJ_ID["Subject ID"]
MRI_ID["MRI ID"]
AGE["Age"]
EDUC["Education (years)"]
GENDER["M/F"]
NWBV["nWBV Normalized Whole Brain Volume"]
ETIV["eTIV Estimated Total Intracranial Volume"]
ASF["ASF Atlas Scaling Factor"]
CDR["CDR Clinical Dementia Rating"]
MMSE["MMSE Mini-Mental State Exam"]

CSV -.-> SUBJ_ID
CSV -.-> MRI_ID
CSV -.-> AGE
CSV -.-> EDUC
CSV -.-> GENDER
CSV -.-> NWBV
CSV -.-> ETIV
CSV -.-> ASF
CSV -.-> CDR
CSV -.-> MMSE

subgraph subGraph3 ["Clinical Scores"]
    CDR
    MMSE
end

subgraph Morphometrics ["Morphometrics"]
    NWBV
    ETIV
    ASF
end

subgraph Demographics ["Demographics"]
    AGE
    EDUC
    GENDER
end

subgraph subGraph0 ["Subject Identifiers"]
    SUBJ_ID
    MRI_ID
end
```

**Sources**: [README.md L12](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L12-L12)

 [README.md L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L168-L168)

### Clinical Features for Multimodal Fusion

The framework extracts five features for multimodal deep learning models:

| Feature | Description | Usage in Model | Range/Units |
| --- | --- | --- | --- |
| `age` | Subject age at scan | Continuous covariate | Years |
| `education` | Years of education | Continuous covariate | Years |
| `nwbv` | Normalized whole brain volume | Structural biomarker | [0, 1] normalized |
| `etiv` | Estimated total intracranial volume | Structural biomarker | cm³ |
| `asf` | Atlas scaling factor | Normalization factor | Scaling ratio |

**Sources**: [README.md L12](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L12-L12)

### MMSE and CDR: Target Proxy Warning

```mermaid
flowchart TD

MMSE["MMSE Score Mini-Mental State Exam"]
CDR["CDR Score Clinical Dementia Rating"]
TARGET["Target Label: AD vs Non-AD"]
LEAK["⚠️ TARGET PROXY LEAKAGE"]
WARN["Artificially inflates model performance"]

MMSE -.->|"Strong correlation"| TARGET
CDR -.->|"Strong correlation"| TARGET
MMSE -.-> LEAK
CDR -.-> LEAK
LEAK -.-> WARN
```

**Important**: The framework includes two SVM baseline scenarios:

1. **`svm_with_mmse_cdr`**: Includes MMSE/CDR scores (demonstrates leakage)
2. **`svm_without_mmse_cdr`**: Clean imaging-based approach (recommended)

MMSE and CDR are diagnostic tools that directly measure cognitive impairment, making them strong proxies for the AD diagnosis target. Including them creates methodologically unsound "shortcuts" that bypass the actual imaging analysis.

**Sources**: [README.md L107-L108](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L107-L108)

 [README.md L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L168-L168)

 High-level Diagram 5

---

## Subject-Level Splitting

### Critical Leakage Prevention Mechanism

The most critical aspect of the data layer is **subject-level splitting**, which prevents data leakage by ensuring all scans from a single patient remain in one partition.

### The Data Leakage Problem

```mermaid
flowchart TD

MR1["OAS2_0001_MR1 Baseline scan"]
MR2["OAS2_0001_MR2 Follow-up scan"]
TRAIN_BAD["Training Set"]
VAL_BAD["Validation Set"]
TRAIN_GOOD["Training Set"]
VAL_GOOD["Validation Set"]
LEAK["Model memorizes patient-specific patterns Performance is artificially inflated"]

MR1 -.->|"Same patient!"| TRAIN_BAD
MR2 -.->|"Same patient!"| VAL_BAD
MR1 -.-> TRAIN_GOOD
MR2 -.-> TRAIN_GOOD

subgraph subGraph2 ["✓ Correct Split (No Leakage)"]
    TRAIN_GOOD
    VAL_GOOD
end

subgraph subGraph1 ["❌ Incorrect Split (Leakage)"]
    TRAIN_BAD
    VAL_BAD
end

subgraph subGraph0 ["Patient OAS2_0001"]
    MR1
    MR2
end
```

**Why This Matters**: MRI scans from the same patient at different timepoints share patient-specific anatomical features (skull shape, brain structure, ventricular geometry). If one timepoint is in training and another in validation, the model can "recognize" the patient and achieve unrealistically high performance.

**Sources**: [README.md L23](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L23-L23)

 High-level Diagram 4

### Subject-Aware Splitting Pipeline

```mermaid
flowchart TD

FILES["~150 NIfTI Files Multiple sessions per subject"]
PARSE["Parse Filenames Extract Subject_ID"]
UNIQUE["Get Unique Subjects ~75-100 unique patients"]
SPLIT["Stratified Split by Subject_ID"]
TRAIN["Training Subjects e.g., OAS2_0001, 0002, 0003"]
VAL["Validation Subjects e.g., OAS2_0004, 0005"]
TEST["Test Subjects e.g., OAS2_0006, 0007"]
ENFORCE["Enforcement: DataLoader only loads files matching assigned subjects"]

FILES -.-> PARSE
PARSE -.-> UNIQUE
UNIQUE -.-> SPLIT
SPLIT -.-> TRAIN
SPLIT -.-> VAL
SPLIT -.-> TEST
TRAIN -.-> ENFORCE
VAL -.-> ENFORCE
TEST -.-> ENFORCE

subgraph subGraph0 ["Output: subject_splits.csv"]
    TRAIN
    VAL
    TEST
end
```

**Sources**: [README.md L23](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L23-L23)

 High-level Diagram 4

### Split Generation

The `run_baselines_cli.py` script generates the `output/subject_splits.csv` file that records the subject-to-partition assignment.

**Expected CSV Structure**:

```
Subject_ID,Split
OAS2_0001,train
OAS2_0002,train
OAS2_0003,train
OAS2_0004,val
OAS2_0005,val
OAS2_0006,test
OAS2_0007,test
```

**Sources**: [README.md L101-L108](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L101-L108)

---

## Data Loading Pipeline

### DataLoader Architecture

```mermaid
flowchart TD

SPLIT_CSV["subject_splits.csv Subject assignments"]
IMAGES["Image Files axl/, cor/, sag/"]
METADATA["oasis_longitudinal_ demographic.csv"]
FILTER["Filter Files by Split Assignment"]
IMBALANCE["Class Imbalance Handling"]
SAMPLER["WeightedRandomSampler Oversample minority class"]
DATALOADER["torch.utils.data.DataLoader Batch creation"]
AUGMENT["Data Augmentation Training only"]
BATCH["Mini-Batch Images + Clinical Features"]

IMAGES -.-> FILTER
METADATA -.-> FILTER
FILTER -.-> IMBALANCE
IMBALANCE -.-> SAMPLER
SAMPLER -.-> DATALOADER
DATALOADER -.-> AUGMENT
AUGMENT -.-> BATCH
```

**Sources**: [README.md L163-L167](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L163-L167)

 High-level Diagram 4

### Class Imbalance Handling

The OASIS-2 dataset exhibits class imbalance (more non-AD than AD cases). The framework employs multiple mechanisms to prevent model collapse:

| Mechanism | Purpose | Implementation |
| --- | --- | --- |
| `WeightedRandomSampler` | Oversample minority class during training | PyTorch DataLoader |
| Class-weighted loss | Penalize misclassification of minority class | Loss function weights |
| Focal Loss | Down-weight easy examples, focus on hard cases | Alternative loss option |
| Balanced Accuracy | Primary metric that accounts for imbalance | Evaluation metric |

**Sources**: [README.md L163-L167](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L163-L167)

 High-level Diagram 4

### WeightedRandomSampler Configuration

```mermaid
flowchart TD

DATASET["Training Dataset"]
COUNT["Count samples per class"]
WEIGHTS["Compute sample weights weight = 1 / class_count"]
SAMPLER["WeightedRandomSampler replacement=True"]
LOADER["DataLoader Balanced batches"]

DATASET -.-> COUNT
COUNT -.-> WEIGHTS
WEIGHTS -.-> SAMPLER
SAMPLER -.-> LOADER
```

**Example**: If training set has 70 non-AD and 30 AD samples:

* Non-AD weight: 1/70 ≈ 0.014
* AD weight: 1/30 ≈ 0.033

The sampler draws samples proportionally to these weights, effectively oversampling the minority class.

**Sources**: [README.md L163-L167](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L163-L167)

---

## Data Augmentation

Data augmentation is applied during training to increase dataset diversity and improve model generalization. Augmentations are **only applied to the training set**, not validation or test sets.

### Augmentation Pipeline

```mermaid
flowchart TD

INPUT["Input Image 2D Slice"]
ROT["Random Rotation ±15 degrees"]
FLIP["Random Horizontal Flip p=0.5"]
NOISE["Gaussian Noise σ=0.05"]
CONTRAST["Random Contrast ±20%"]
NORMALIZE["Normalization Mean=0, Std=1"]
OUTPUT["Augmented Image"]

INPUT -.-> ROT
FLIP -.-> NOISE
CONTRAST -.-> NORMALIZE
NORMALIZE -.-> OUTPUT

subgraph subGraph1 ["Intensity Augmentations"]
    NOISE
    CONTRAST
    NOISE -.-> CONTRAST
end

subgraph subGraph0 ["Spatial Augmentations"]
    ROT
    FLIP
    ROT -.-> FLIP
end
```

### Typical Augmentation Parameters

| Augmentation | Parameter | Rationale |
| --- | --- | --- |
| Rotation | ±15° | Accounts for head position variation |
| Horizontal Flip | p=0.5 | Brain has approximate bilateral symmetry |
| Gaussian Noise | σ=0.05 | Models scanner noise |
| Contrast Adjustment | ±20% | Accounts for scanner calibration differences |
| Normalization | z-score | Standardizes intensity ranges |

**Important**: Augmentations are implemented as part of the PyTorch Dataset transform pipeline. The random seed controls reproducibility of augmentation sequences.

**Sources**: High-level Diagram 4, [README.md L163-L167](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L163-L167)

---

## Multi-Plane Data Loading

### Three-Stream Architecture Data Requirements

```mermaid
flowchart TD

SUBJECT["Subject: OAS2_0001_MR1"]
AXL_IMG["Load axl/ OAS2_0001_MR1_axl.nii.gz"]
COR_IMG["Load cor/ OAS2_0001_MR1_cor.nii.gz"]
SAG_IMG["Load sag/ OAS2_0001_MR1_sag.nii.gz"]
META["Load from CSV: age, education, nwbv, etiv, asf"]
STREAM_A["Axial Stream"]
STREAM_C["Coronal Stream"]
STREAM_S["Sagittal Stream"]
CLINICAL["Clinical Branch"]

SUBJECT -.-> AXL_IMG
SUBJECT -.-> COR_IMG
SUBJECT -.-> SAG_IMG
SUBJECT -.-> META
META -.-> CLINICAL

subgraph subGraph2 ["Multi-Stream Model Input"]
    STREAM_A
    STREAM_C
    STREAM_S
    CLINICAL
end

subgraph subGraph1 ["Load Clinical Features"]
    META
end

subgraph subGraph0 ["Load Three Planes"]
    AXL_IMG
    COR_IMG
    SAG_IMG
end
```

**Note**: The framework supports flexible plane configurations:

* Single-stream: Only one plane (e.g., axial only)
* Multi-stream: Two or three planes simultaneously
* Multimodal: Multi-stream + clinical features

**Sources**: [README.md L10-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L15)

 [README.md L110-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L110-L118)

 High-level Diagram 2

---

## Data Validation

### File Integrity Checks

The data layer includes validation logic to ensure data integrity before training:

```mermaid
flowchart TD

START["Data Validation"]
CHECK1["Check file existence All required planes present?"]
CHECK2["Check filename format Matches OAS2_XXXX_MRY_plane?"]
CHECK3["Check NIfTI header Valid dimensions?"]
CHECK4["Check CSV integrity All subjects have metadata?"]
CHECK5["Check subject coverage Images match CSV records?"]
PASS["Validation Passed Proceed to training"]
FAIL["Validation Failed Report errors"]

START -.->|"Yes"| CHECK1
CHECK1 -.->|"Yes"| CHECK2
CHECK1 -.->|"No"| FAIL
CHECK2 -.->|"Yes"| CHECK3
CHECK2 -.->|"No"| FAIL
CHECK3 -.->|"Yes"| CHECK4
CHECK3 -.->|"No"| FAIL
CHECK4 -.->|"Yes"| CHECK5
CHECK4 -.->|"No"| FAIL
CHECK5 -.->|"No"| PASS
CHECK5 -.-> FAIL
```

**Sources**: High-level Diagram 4 (Data Parsing & Validation box)

---

## Output Directory Structure

Generated artifacts are stored in the `output/` directory, maintaining separation from input data.

### Output Organization

```markdown
output/
├── subject_splits.csv              # Train/Val/Test assignments
├── models/                         # Saved model checkpoints
│   ├── efficientnet_best.pth
│   ├── densenet_best.pth
│   └── medicalnet_best.pth
├── logs/                          # Training logs
│   ├── training_log.csv
│   └── tensorboard/
├── plots/                         # Visualization outputs
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── training_curves.png
└── tables/                        # LaTeX tables for publication
    └── results_table.tex
```

**Sources**: [README.md L37](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L37-L37)

 [README.md L177-L195](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L195)

---

## Data Layer Code Map

### Key Code Entities

| Code Entity | Location | Purpose |
| --- | --- | --- |
| Dataset classes | `brain_mri/ml/` | PyTorch Dataset implementations |
| Filename parser | `brain_mri/utils/` | Extract Subject_ID and MRI_ID |
| Data validator | `brain_mri/ml/` | Integrity checks |
| Subject splitter | `run_baselines_cli.py` | Generate subject_splits.csv |
| Image preprocessing | `brain_mri/utils/` | NIfTI loading and normalization |
| Augmentation transforms | `brain_mri/ml/` | Training-time augmentations |
| WeightedRandomSampler | PyTorch | Class imbalance handling |

**Sources**: [README.md L177-L195](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L177-L195)

---

## Summary

The data layer implements a rigorous structure for medical imaging data management with emphasis on:

1. **Subject-level splitting** to prevent data leakage
2. **Flexible multi-plane architecture** supporting axial, coronal, and sagittal views
3. **Multimodal fusion** combining imaging and clinical features
4. **Class imbalance handling** via weighted sampling and appropriate metrics
5. **Data augmentation** for improved generalization
6. **Strict validation** ensuring data integrity

The combination of NIfTI medical imaging format, subject-aware splitting, and the WeightedRandomSampler creates a methodologically sound foundation for training AD classification models.

**Sources**: [README.md L1-L218](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L1-L218)

 High-level Diagrams 1-6

Refresh this wiki

Last indexed: 5 January 2026 ([cd9d51](https://github.com/ThalesMMS/brain-mri-pipelines-py/commit/cd9d51a5))

### On this page

* [Data Layer](#4-data-layer)
* [Dataset Organization](#4-dataset-organization)
* [Directory Structure](#4-directory-structure)
* [File Naming Convention](#4-file-naming-convention)
* [Naming Pattern](#4-naming-pattern)
* [Identifier Extraction Diagram](#4-identifier-extraction-diagram)
* [Examples](#4-examples)
* [NIfTI File Format](#4-nifti-file-format)
* [Format Overview](#4-format-overview)
* [NIfTI Structure](#4-nifti-structure)
* [Coordinate Systems](#4-coordinate-systems)
* [Clinical Metadata](#4-clinical-metadata)
* [Metadata Schema](#4-metadata-schema)
* [Clinical Features for Multimodal Fusion](#4-clinical-features-for-multimodal-fusion)
* [MMSE and CDR: Target Proxy Warning](#4-mmse-and-cdr-target-proxy-warning)
* [Subject-Level Splitting](#4-subject-level-splitting)
* [Critical Leakage Prevention Mechanism](#4-critical-leakage-prevention-mechanism)
* [The Data Leakage Problem](#4-the-data-leakage-problem)
* [Subject-Aware Splitting Pipeline](#4-subject-aware-splitting-pipeline)
* [Split Generation](#4-split-generation)
* [Data Loading Pipeline](#4-data-loading-pipeline)
* [DataLoader Architecture](#4-dataloader-architecture)
* [Class Imbalance Handling](#4-class-imbalance-handling)
* [WeightedRandomSampler Configuration](#4-weightedrandomsampler-configuration)
* [Data Augmentation](#4-data-augmentation)
* [Augmentation Pipeline](#4-augmentation-pipeline)
* [Typical Augmentation Parameters](#4-typical-augmentation-parameters)
* [Multi-Plane Data Loading](#4-multi-plane-data-loading)
* [Three-Stream Architecture Data Requirements](#4-three-stream-architecture-data-requirements)
* [Data Validation](#4-data-validation)
* [File Integrity Checks](#4-file-integrity-checks)
* [Output Directory Structure](#4-output-directory-structure)
* [Output Organization](#4-output-organization)
* [Data Layer Code Map](#4-data-layer-code-map)
* [Key Code Entities](#4-key-code-entities)
* [Summary](#4-summary)

Ask Devin about brain-mri-pipelines-py