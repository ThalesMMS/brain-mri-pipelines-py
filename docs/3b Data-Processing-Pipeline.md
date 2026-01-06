# Data Processing Pipeline

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)
> * [axl/OAS2_0002_MR1_axl.nii.gz](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0002_MR1_axl.nii.gz)
> * [axl/OAS2_0002_MR2_axl.nii.gz](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0002_MR2_axl.nii.gz)

This document describes the end-to-end data processing pipeline that transforms raw NIfTI neuroimaging files into training-ready batches. The pipeline encompasses file parsing, clinical metadata integration, subject-aware splitting, dataset construction, data loading, and augmentation.

For architectural details about the multi-stream network that consumes this data, see [Multi-Stream Multimodal Network](3a%20Multi-Stream-Multimodal-Network.md). For the critical subject-level splitting mechanism that prevents data leakage, see [Subject-Level Splitting & Leakage Prevention](3d%20Subject-Level-Splitting-&-Leakage-Prevention.md). For comprehensive dataset documentation, see [Data Layer](4%20Data-Layer.md).

**Sources:** README.md

---

## Pipeline Overview

The data processing pipeline operates in distinct stages, each with specific responsibilities for transforming raw medical imaging data into model-ready tensors.

```mermaid
flowchart TD

NIFTI["NIfTI Files axl/, cor/, sag/"]
CSV["oasis_longitudinal_ demographic.csv"]
PARSER["Filename Parser Extract Subject_ID Extract MRI_ID"]
VALIDATOR["Data Validator Check file integrity"]
JOINER["Metadata Joiner Merge imaging + clinical"]
RECORDS["Unified Records {filepath, subject, mri_id, label, covariates}"]
SPLITTER["Subject-Aware Splitter Group by Subject_ID"]
TRAIN_SET["Training Set Subject partition 1"]
VAL_SET["Validation Set Subject partition 2"]
TEST_SET["Test Set Subject partition 3"]
DATASET["PyTorch Dataset getitem logic"]
SAMPLER["WeightedRandomSampler Handle class imbalance"]
LOADER["DataLoader Batch construction"]
AUG["Augmentation Pipeline Rotation, Flip, Noise"]
NORM["Normalization Intensity rescaling"]
TENSOR["Tensor Batches {images, clinical_features, labels}"]

CSV -.-> JOINER
PARSER -.-> VALIDATOR
RECORDS -.-> SPLITTER
LOADER -.-> AUG

subgraph subGraph4 ["Stage 5: Runtime Augmentation"]
    AUG
    NORM
    TENSOR
    AUG -.-> NORM
    NORM -.-> TENSOR
end

subgraph subGraph3 ["Stage 4: Dataset & DataLoader"]
    DATASET
    SAMPLER
    LOADER
    DATASET -.-> SAMPLER
    SAMPLER -.-> LOADER
end

subgraph subGraph2 ["Stage 3: Subject-Level Splitting"]
    SPLITTER
    TRAIN_SET
    VAL_SET
    TEST_SET
    SPLITTER -.-> TRAIN_SET
    SPLITTER -.-> VAL_SET
    SPLITTER -.-> TEST_SET
end

subgraph subGraph1 ["Stage 2: Data Organization"]
    VALIDATOR
    JOINER
    RECORDS
    VALIDATOR -.-> JOINER
    JOINER -.-> RECORDS
end

subgraph subGraph0 ["Stage 1: Raw Data Ingestion"]
    NIFTI
    CSV
    PARSER
    NIFTI -.-> PARSER
end
```

**Sources:** README.md

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L160-L169)

---

## Data Ingestion & File Parsing

### Filename Convention

The pipeline relies on a strict naming convention to extract metadata from filenames:

| Component | Pattern | Example | Description |
| --- | --- | --- | --- |
| **Subject ID** | `OAS2_XXXX` | `OAS2_0001` | Unique patient identifier |
| **MRI ID** | `OAS2_XXXX_MRY` | `OAS2_0001_MR1` | Specific scan identifier (Y = time point) |
| **Plane** | `{axl,cor,sag}` | `axl` | Anatomical plane |
| **Extension** | `.nii.gz` or `.nii` | `.nii.gz` | Compressed NIfTI format |

**Full Pattern:** `OAS2_XXXX_MRY_{axl,cor,sag}.nii.gz`

```mermaid
flowchart TD

FILE["OAS2_0002_MR1_axl.nii.gz"]
SUBJECT["Subject_ID: OAS2_0002"]
MRI["MRI_ID: OAS2_0002_MR1"]
PLANE["Plane: axl"]
SPLIT["Used for subject-level splitting"]
JOIN["Used to join with clinical metadata"]
STREAM["Used for multi-stream routing"]

FILE -.-> SUBJECT
FILE -.-> MRI
FILE -.-> PLANE
SUBJECT -.-> SPLIT
MRI -.-> JOIN
PLANE -.-> STREAM
```

**Sources:** README.md

### Clinical Metadata Integration

The pipeline joins imaging files with tabular clinical data from `oasis_longitudinal_demographic.csv`:

| Clinical Feature | Type | Description |
| --- | --- | --- |
| **Subject ID** | String | Join key to imaging data |
| **MRI ID** | String | Specific scan identifier |
| **Age** | Numeric | Patient age at scan time |
| **Education** | Numeric | Years of education |
| **nWBV** | Numeric | Normalized whole-brain volume |
| **eTIV** | Numeric | Estimated total intracranial volume |
| **ASF** | Numeric | Atlas scaling factor |
| **CDR** | Numeric | Clinical Dementia Rating (target proxy) |
| **MMSE** | Numeric | Mini-Mental State Examination (target proxy) |

**Sources:** [README.md L36](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L36-L36)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L106-L108)

---

## Subject-Level Splitting Strategy

The pipeline implements **strict subject-level partitioning** to prevent data leakage. This is critical because the OASIS-2 dataset contains longitudinal scans (multiple MRI sessions per patient over time).

```mermaid
flowchart TD

SCAN1["OAS2_0002_MR1_axl.nii.gz Time point 1"]
SCAN2["OAS2_0002_MR2_axl.nii.gz Time point 2"]
WRONG_TRAIN["Training Set OAS2_0002_MR1"]
WRONG_VAL["Validation Set OAS2_0002_MR2"]
LEAK["⚠️ DATA LEAKAGE Model sees same patient in train and validation"]
CORRECT_TRAIN["Training Set OAS2_0002_MR1 OAS2_0002_MR2 All scans from subject"]
CORRECT_VAL["Validation Set OAS2_0003_MR1 OAS2_0003_MR2 Different subjects only"]
NO_LEAK["✓ NO LEAKAGE All scans from a subject stay in one partition"]

SCAN1 -.-> WRONG_TRAIN
SCAN2 -.-> WRONG_VAL
SCAN1 -.-> CORRECT_TRAIN
SCAN2 -.-> CORRECT_TRAIN

subgraph subGraph2 ["CORRECT Splitting (No Leakage)"]
    CORRECT_TRAIN
    CORRECT_VAL
    NO_LEAK
end

subgraph subGraph1 ["INCORRECT Splitting (Leakage)"]
    WRONG_TRAIN
    WRONG_VAL
    LEAK
end

subgraph subGraph0 ["Example: Subject OAS2_0002 Scans"]
    SCAN1
    SCAN2
end
```

### Splitting Algorithm

1. **Group by Subject_ID**: All scans are grouped by `OAS2_XXXX`
2. **Partition subjects**: Subjects (not scans) are split into Train/Val/Test
3. **Assign scans**: All MRI sessions from a subject go to the same partition

**Sources:** [README.md L23](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L23-L23)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L160-L169)

---

## Dataset Construction

The pipeline constructs PyTorch `Dataset` objects that handle lazy loading and multi-view data assembly.

```mermaid
flowchart TD

INIT["init(split, planes, multimodal)"]
GETITEM["getitem(idx)"]
LEN["len()"]
RECORD["Record = {   'filepath_axl': path,   'filepath_cor': path,   'filepath_sag': path,   'subject_id': str,   'mri_id': str,   'label': int,   'age': float,   'education': float,   'nwbv': float,   'etiv': float,   'asf': float }"]
LOAD["Load NIfTI(s) nibabel.load()"]
SLICE["Extract 2D Slice Middle slice selection"]
MULTI["Concatenate Planes If multi-stream enabled"]
CLINICAL["Extract Clinical If multimodal enabled"]
TRANSFORM["Apply Transforms Augmentation pipeline"]

INIT -.-> RECORD
GETITEM -.-> LOAD

subgraph subGraph2 ["getitem Logic"]
    LOAD
    SLICE
    MULTI
    CLINICAL
    TRANSFORM
    LOAD -.-> SLICE
    SLICE -.-> MULTI
    MULTI -.-> CLINICAL
    CLINICAL -.-> TRANSFORM
end

subgraph subGraph1 ["Data Record Format"]
    RECORD
end

subgraph subGraph0 ["Dataset Class Structure"]
    INIT
    GETITEM
    LEN
end
```

### Multi-Stream Data Loading

When multiple anatomical planes are enabled, the dataset loads and stacks images from `axl/`, `cor/`, and `sag/` directories:

| Configuration | Planes Loaded | Output Shape |
| --- | --- | --- |
| **Single-stream** | Axial only | `(1, H, W)` |
| **Multi-stream** | Axial + Coronal + Sagittal | `(3, H, W)` |
| **Multimodal** | Multi-stream + Clinical features | Images: `(3, H, W)`Clinical: `(5,)` |

**Sources:** [README.md L10-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L15)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L112-L118)

---

## Data Loading & Class Imbalance Handling

### WeightedRandomSampler

The pipeline uses PyTorch's `WeightedRandomSampler` to address severe class imbalance in Alzheimer's detection (more non-AD than AD cases):

```mermaid
flowchart TD

AD["AD Cases Minority class ~30%"]
NON_AD["Non-AD Cases Majority class ~70%"]
WEIGHTS["Compute Sample Weights weight = 1 / class_count"]
SAMPLE["Sample with Replacement Higher probability for minority class"]
BALANCED["Balanced Batches ~50% AD ~50% Non-AD"]

AD -.-> WEIGHTS
SAMPLE -.-> BALANCED

subgraph subGraph2 ["Effective Batch Distribution"]
    BALANCED
end

subgraph WeightedRandomSampler ["WeightedRandomSampler"]
    WEIGHTS
    SAMPLE
    WEIGHTS -.-> SAMPLE
end

subgraph subGraph0 ["Class Distribution"]
    AD
    NON_AD
end
```

### DataLoader Configuration

```mermaid
flowchart TD

DATASET["PyTorch Dataset Train/Val/Test split"]
SAMPLER["WeightedRandomSampler Only for training"]
LOADER["DataLoader batch_size=32 num_workers=4 pin_memory=True"]
COLLATE["Collate Function Stack images Stack clinical features Stack labels"]
BATCH["Output Batch: {   'images': (B, C, H, W),   'clinical': (B, 5),   'labels': (B,) }"]

DATASET -.-> SAMPLER
SAMPLER -.-> LOADER
LOADER -.-> COLLATE
COLLATE -.-> BATCH
```

**Sources:** [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L164-L166)

---

## Augmentation Pipeline

Data augmentation is applied **during training only** (not validation/test) to increase dataset diversity and prevent overfitting.

### Augmentation Operations

| Transform | Parameters | Purpose |
| --- | --- | --- |
| **Random Rotation** | `±15°` | Simulate patient positioning variance |
| **Random Horizontal Flip** | `p=0.5` | Exploit left-right symmetry |
| **Random Vertical Flip** | `p=0.5` | Increase spatial diversity |
| **Gaussian Noise** | `σ=0.01` | Simulate scanner noise |
| **Intensity Scaling** | `[0.9, 1.1]` | Simulate scanner calibration variance |

```mermaid
flowchart TD

LOAD_VAL["Load Raw Image nibabel.load()"]
SLICE_VAL["Extract Slice Middle slice"]
NORM_VAL["Normalization Mean=0, Std=1"]
TENSOR_VAL["To Tensor (C, H, W)"]
LOAD["Load Raw Image nibabel.load()"]
SLICE["Extract Slice Middle slice"]
AUG["Augmentation Rotation, Flip, Noise"]
NORM["Normalization Mean=0, Std=1"]
TENSOR["To Tensor (C, H, W)"]

subgraph subGraph1 ["Validation/Test Pipeline"]
    LOAD_VAL
    SLICE_VAL
    NORM_VAL
    TENSOR_VAL
end

subgraph subGraph0 ["Training Pipeline"]
    LOAD
    SLICE
    AUG
    NORM
    TENSOR
    LOAD -.-> SLICE
    SLICE -.-> AUG
    AUG -.-> NORM
    NORM -.-> TENSOR
end
```

**Sources:** [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L164-L166)

---

## Pipeline Integration Points

### Training Loop Integration

The data processing pipeline integrates with the training loop through the following interfaces:

```mermaid
flowchart TD

BUILD["Build Datasets Train/Val/Test splits"]
CREATE_LOADERS["Create DataLoaders With WeightedRandomSampler"]
ITER["for batch in train_loader:"]
UNPACK["images, clinical, labels = batch"]
FORWARD["model(images, clinical)"]
LOSS["loss = criterion(outputs, labels)"]
BACKWARD["loss.backward()"]
ITER_VAL["for batch in val_loader:"]
UNPACK_VAL["images, clinical, labels = batch"]
FORWARD_VAL["model(images, clinical)"]
METRICS["balanced_accuracy, etc."]

subgraph subGraph2 ["Validation Loop"]
    ITER_VAL
    UNPACK_VAL
    FORWARD_VAL
    METRICS
end

subgraph subGraph1 ["Training Epoch Loop"]
    ITER
    UNPACK
    FORWARD
    LOSS
    BACKWARD
    ITER -.-> UNPACK
    UNPACK -.-> FORWARD
    FORWARD -.-> LOSS
    LOSS -.-> BACKWARD
end

subgraph subGraph0 ["Initialization Phase"]
    BUILD
    CREATE_LOADERS
    BUILD -.-> CREATE_LOADERS
end
```

### Multi-Stream Model Input

For multi-stream architectures, the pipeline prepares separate data paths for each anatomical plane:

```mermaid
flowchart TD

BATCH["Batch from DataLoader {images: (B, 3, H, W)}"]
SPLIT["Split by Plane"]
AXL["Axial Stream (B, 1, H, W)"]
COR["Coronal Stream (B, 1, H, W)"]
SAG["Sagittal Stream (B, 1, H, W)"]
BACKBONE_A["Backbone EfficientNet/DenseNet/ MedicalNet"]
BACKBONE_C["Backbone EfficientNet/DenseNet/ MedicalNet"]
BACKBONE_S["Backbone EfficientNet/DenseNet/ MedicalNet"]
EMB_A["Embeddings (B, 1280)"]
EMB_C["Embeddings (B, 1280)"]
EMB_S["Embeddings (B, 1280)"]
CLINICAL["Clinical Features (B, 5)"]
CONCAT["Concatenate (B, 3845)"]
CLASSIFIER["Classification Head (B, 2)"]

BATCH -.-> SPLIT
SPLIT -.-> AXL
SPLIT -.-> COR
SPLIT -.-> SAG
AXL -.-> BACKBONE_A
COR -.-> BACKBONE_C
SAG -.-> BACKBONE_S
CLINICAL -.-> CONCAT
CONCAT -.-> CLASSIFIER
```

**Sources:** [README.md L10-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L15)

---

## Data Validation & Error Handling

The pipeline includes validation steps to ensure data integrity:

| Validation Check | Action on Failure |
| --- | --- |
| **NIfTI file exists** | Skip sample, log warning |
| **File can be loaded** | Skip sample, log error |
| **Expected dimensions** | Skip sample, log error |
| **Subject_ID matches** | Skip sample, log error |
| **Clinical data available** | Use default values or skip |
| **Label is valid** | Skip sample, log error |

```mermaid
flowchart TD

LOAD["Attempt Load nibabel.load(filepath)"]
CHECK_DIM["Check Dimensions Expected: (176, 208, 176)"]
CHECK_LABEL["Check Label Must be 0 or 1"]
CHECK_CLINICAL["Check Clinical Data Required if multimodal=True"]
VALID["Valid Sample Return to DataLoader"]
SKIP["Skip Sample Log error"]

LOAD -.->|"Valid"| CHECK_DIM
```

**Sources:** README.md

---

## Performance Optimizations

### Caching Strategy

The pipeline does not implement explicit caching of loaded images due to memory constraints. Each `__getitem__` call loads data from disk. For faster experimentation, users can:

1. Store data on SSD rather than HDD
2. Use `pin_memory=True` in DataLoader
3. Increase `num_workers` for parallel loading
4. Pre-process data to a faster format (not implemented)

### Memory Management

| Configuration | Memory Usage | Speed |
| --- | --- | --- |
| `num_workers=0` | Low (sequential) | Slowest |
| `num_workers=4` | Medium (parallel) | Faster |
| `pin_memory=True` | Higher (CUDA pinned) | Fastest for GPU |
| `prefetch_factor=2` | Higher (pre-fetch) | Reduced waiting |

**Sources:** README.md

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L160-L169)

---

## Summary: Pipeline Stages

The complete data processing pipeline can be summarized in these stages:

1. **Ingestion**: Parse NIfTI filenames → Extract Subject_ID, MRI_ID, Plane
2. **Integration**: Join imaging paths with clinical metadata from CSV
3. **Validation**: Check file integrity, label validity, dimension correctness
4. **Splitting**: Group by Subject_ID → Partition subjects into Train/Val/Test
5. **Dataset**: Construct PyTorch `Dataset` with `__getitem__` for lazy loading
6. **Sampling**: Apply `WeightedRandomSampler` to training set (class balance)
7. **Loading**: Create `DataLoader` with batching and multi-worker parallelism
8. **Augmentation**: Apply transforms during training (rotation, flip, noise)
9. **Normalization**: Standardize intensity values (mean=0, std=1)
10. **Delivery**: Yield batches to training loop as `{images, clinical, labels}`

This pipeline ensures rigorous data handling with emphasis on preventing leakage through subject-level splitting, addressing class imbalance through weighted sampling, and supporting flexible multi-stream, multimodal architectures.

**Sources:** README.md





### On this page

* [Data Processing Pipeline](#3.2-data-processing-pipeline)
* [Pipeline Overview](#3.2-pipeline-overview)
* [Data Ingestion & File Parsing](#3.2-data-ingestion-file-parsing)
* [Filename Convention](#3.2-filename-convention)
* [Clinical Metadata Integration](#3.2-clinical-metadata-integration)
* [Subject-Level Splitting Strategy](#3.2-subject-level-splitting-strategy)
* [Splitting Algorithm](#3.2-splitting-algorithm)
* [Dataset Construction](#3.2-dataset-construction)
* [Multi-Stream Data Loading](#3.2-multi-stream-data-loading)
* [Data Loading & Class Imbalance Handling](#3.2-data-loading-class-imbalance-handling)
* [WeightedRandomSampler](#3.2-weightedrandomsampler)
* [DataLoader Configuration](#3.2-dataloader-configuration)
* [Augmentation Pipeline](#3.2-augmentation-pipeline)
* [Augmentation Operations](#3.2-augmentation-operations)
* [Pipeline Integration Points](#3.2-pipeline-integration-points)
* [Training Loop Integration](#3.2-training-loop-integration)
* [Multi-Stream Model Input](#3.2-multi-stream-model-input)
* [Data Validation & Error Handling](#3.2-data-validation-error-handling)
* [Performance Optimizations](#3.2-performance-optimizations)
* [Caching Strategy](#3.2-caching-strategy)
* [Memory Management](#3.2-memory-management)
* [Summary: Pipeline Stages](#3.2-summary-pipeline-stages)

Ask Devin about brain-mri-pipelines-py