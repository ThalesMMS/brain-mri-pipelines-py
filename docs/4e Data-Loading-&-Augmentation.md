# Data Loading & Augmentation

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)
> * [axl/OAS2_0008_MR1_axl.nii.gz](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0008_MR1_axl.nii.gz)
> * [axl/OAS2_0009_MR1_axl.nii.gz](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/axl/OAS2_0009_MR1_axl.nii.gz)

## Purpose and Scope

This page documents the data loading pipeline and augmentation techniques used during model training. It covers how MRI images are loaded from disk, transformed into training batches, and augmented to improve model generalization. For information about the dataset structure and file organization, see [Directory Organization & File Naming](#4.3). For details on subject-level splitting to prevent data leakage, see [Subject-Level Splitting & Leakage Prevention](#3.4).

---

## Data Loading Pipeline Overview

The data loading pipeline transforms raw NIfTI files into batched tensors suitable for training deep learning models. The pipeline handles multi-view data (axial, coronal, sagittal), applies augmentation, and manages class imbalance through weighted sampling.

### End-to-End Data Flow

```mermaid
flowchart TD

AXL["axl/ OAS2_*.nii.gz"]
COR["cor/ OAS2_*.nii.gz"]
SAG["sag/ OAS2_*.nii.gz"]
CSV["oasis_longitudinal_ demographic.csv"]
SPLIT["subject_split.csv (Train/Val/Test assignments)"]
PARSE["Filename Parser Extract Subject_ID & MRI_ID"]
FILTER["Filter by split (train/validation/test)"]
DATASET["Custom Dataset Class getitem returns: - image tensors (3 views) - clinical features - label"]
CACHE["Optional image caching for repeated epochs"]
SAMPLER["WeightedRandomSampler Balances AD/Non-AD classes"]
WEIGHTS["Per-sample weights inversely proportional to class frequency"]
AUG_ROT["Random Rotation ±10-15 degrees"]
AUG_FLIP["Random Horizontal Flip probability=0.5"]
AUG_NOISE["Gaussian Noise Addition std=0.01-0.05"]
AUG_NORM["Intensity Normalization z-score standardization"]
LOADER["PyTorch DataLoader batch_size=16-32 num_workers=4"]
COLLATE["Collate function Stack images, clinical features, labels"]
BATCH["Batched Tensors images: [B, C, H, W] clinical: [B, 5] labels: [B]"]

AXL -.-> PARSE
COR -.-> PARSE
SAG -.-> PARSE
CSV -.-> PARSE
FILTER -.-> DATASET
DATASET -.-> SAMPLER
DATASET -.-> AUG_ROT
SAMPLER -.-> LOADER
COLLATE -.-> BATCH

subgraph subGraph6 ["7. Training Input"]
    BATCH
end

subgraph subGraph5 ["6. Batch Creation"]
    LOADER
    COLLATE
    LOADER -.-> COLLATE
end

subgraph subGraph4 ["5. Data Augmentation"]
    AUG_ROT
    AUG_FLIP
    AUG_NOISE
    AUG_NORM
end

subgraph subGraph3 ["4. Weighted Sampling"]
    SAMPLER
    WEIGHTS
    WEIGHTS -.-> SAMPLER
end

subgraph subGraph2 ["3. PyTorch Dataset"]
    DATASET
    CACHE
end

subgraph subGraph1 ["2. Dataset Initialization"]
    SPLIT
    PARSE
    FILTER
    SPLIT -.-> FILTER
    PARSE -.-> FILTER
end

subgraph subGraph0 ["1. Raw Data Storage"]
    AXL
    COR
    SAG
    CSV
end
```

**Sources:** [README.md L27-L51](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L27-L51)

 [README.md L162-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L162-L168)

 High-level Diagram 4 from context

---

## Dataset Class Implementation

The custom dataset class is responsible for loading individual samples and returning them in a format suitable for model training. It handles multi-view loading and integrates clinical metadata.

### Dataset Structure

```mermaid
flowchart TD

IMG_DICT["images: dict {'axl': tensor, 'cor': tensor, 'sag': tensor}"]
IDX["Input: index"]
LOAD_AXL["Load axl/_axl.nii.gz"]
LOAD_COR["Load cor/_cor.nii.gz"]
LOAD_SAG["Load sag/_sag.nii.gz"]
CLIN_LOOKUP["Lookup in demographic CSV"]
CDR_LOOKUP["Lookup CDR score"]
SLICE["Extract middle slice or aggregate volume"]
CLIN_FEAT["Extract: age, education, nwbv, etiv, asf"]
CLIN_NORM["Normalize features (pre-computed stats)"]
CLIN_TENSOR["clinical: tensor[5]"]
LABEL["Binary label: 0=Non-AD 1=AD (CDR > 0)"]
LABEL_TENSOR["label: int"]

subgraph Dataset.__getitem__(idx) ["Dataset.getitem(idx)"]
    IDX
    IDX -.-> LOAD_AXL
    IDX -.-> LOAD_COR
    IDX -.-> LOAD_SAG
    IDX -.-> CLIN_LOOKUP
    IDX -.-> CDR_LOOKUP
    SLICE -.-> IMG_DICT
    LABEL -.-> LABEL_TENSOR

subgraph Output ["Output"]
    IMG_DICT
    CLIN_TENSOR
    LABEL_TENSOR
end

subgraph subGraph2 ["Label Assignment"]
    CDR_LOOKUP
    LABEL
end

subgraph subGraph1 ["Clinical Features"]
    CLIN_LOOKUP
    CLIN_FEAT
    CLIN_NORM
end

subgraph subGraph0 ["Image Loading"]
    LOAD_AXL
    LOAD_COR
    LOAD_SAG
    SLICE
end
end
```

**Sources:** [README.md L36-L50](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L36-L50)

 High-level Diagram 2 from context

### Key Implementation Details

| Component | Description | Configuration |
| --- | --- | --- |
| **Image Format** | NIfTI-1 format (`.nii.gz`) | 3D volumes, middle slice extracted |
| **Preprocessing** | Intensity normalization | Z-score standardization per image |
| **Clinical Features** | 5 demographic variables | `age, education, nwbv, etiv, asf` |
| **Label Derivation** | CDR score from demographic CSV | Binary: CDR > 0 indicates AD |
| **Caching Strategy** | Optional in-memory cache | Enabled for repeated epoch training |

**Sources:** [README.md L36-L50](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L36-L50)

 [README.md L23-L24](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L23-L24)

---

## WeightedRandomSampler for Class Imbalance

Medical datasets typically exhibit severe class imbalance (fewer AD cases than healthy controls). The `WeightedRandomSampler` ensures balanced representation during training by oversampling the minority class.

### Sampling Mechanism

```mermaid
flowchart TD

LABELS["Training Labels [0,0,1,0,0,1,0,...]"]
COUNT["Count per class class_0: 120 class_1: 30"]
FREQ["Class frequencies freq_0: 0.80 freq_1: 0.20"]
INV_FREQ["Inverse frequency weight_0: 1/0.80 = 1.25 weight_1: 1/0.20 = 5.0"]
ASSIGN["Assign weight to each sample sample with label=0 → 1.25 sample with label=1 → 5.0"]
SAMPLER["WeightedRandomSampler replacement=True num_samples=len(dataset)"]
PROB["Sampling probability ∝ sample weight"]
BATCH["Balanced batch ~50% class_0 ~50% class_1"]
EFFECT["Effect: Minority class seen more frequently"]

FREQ -.-> INV_FREQ
ASSIGN -.-> SAMPLER
PROB -.-> BATCH

subgraph subGraph3 ["4. Training Batch"]
    BATCH
    EFFECT
    BATCH -.-> EFFECT
end

subgraph subGraph2 ["3. Weighted Random Sampling"]
    SAMPLER
    PROB
    SAMPLER -.-> PROB
end

subgraph subGraph1 ["2. Calculate Sample Weights"]
    INV_FREQ
    ASSIGN
end

subgraph subGraph0 ["1. Compute Class Frequencies"]
    LABELS
    COUNT
    FREQ
    LABELS -.-> COUNT
    COUNT -.-> FREQ
end
```

**Sources:** [README.md L165-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L165-L168)

 High-level Diagram 4 from context

### WeightedRandomSampler Configuration

The sampler is configured as follows:

```
# Pseudo-code representation of implementationweights = compute_sample_weights(train_labels)  # inverse class frequencysampler = WeightedRandomSampler(    weights=weights,    num_samples=len(train_dataset),    replacement=True  # Allow same sample multiple times per epoch)train_loader = DataLoader(    train_dataset,    batch_size=32,    sampler=sampler,  # Replaces shuffle=True    num_workers=4)
```

**Key Parameters:**

| Parameter | Value | Rationale |
| --- | --- | --- |
| `replacement` | `True` | Allows oversampling minority class |
| `num_samples` | `len(dataset)` | Maintains consistent epoch length |
| `weights` | Inverse class frequency | Balances class representation |

**Sources:** [README.md L165-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L165-L168)

 High-level Diagram 4 from context

---

## Image Augmentation Techniques

Data augmentation artificially increases training set diversity, improving model generalization and reducing overfitting. The pipeline applies geometric and intensity-based transformations.

### Augmentation Pipeline

```mermaid
flowchart TD

IMG_IN["Raw MRI Slice [H, W] or [1, H, W]"]
ROT["Random Rotation angle ∈ [-15°, +15°] probability: 0.5"]
FLIP["Random Horizontal Flip probability: 0.5"]
NOISE["Gaussian Noise μ=0, σ=0.01-0.05 probability: 0.3"]
BRIGHT["Random Brightness factor ∈ [0.9, 1.1] probability: 0.3"]
ZNORM["Z-score Normalization (x - μ) / σ"]
CLIP["Clip outliers [-3σ, +3σ]"]
IMG_OUT["Augmented Tensor [C, H, W]"]

FLIP -.-> NOISE
BRIGHT -.-> ZNORM
CLIP -.-> IMG_OUT

subgraph Output ["Output"]
    IMG_OUT
end

subgraph Normalization ["Normalization"]
    ZNORM
    CLIP
    ZNORM -.-> CLIP
end

subgraph subGraph2 ["Intensity Transforms"]
    NOISE
    BRIGHT
    NOISE -.-> BRIGHT
end

subgraph subGraph1 ["Geometric Transforms"]
    ROT
    FLIP
    ROT -.-> FLIP
end

subgraph Input ["Input"]
    IMG_IN
end
```

**Sources:** High-level Diagram 4 from context, [README.md L20-L22](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L20-L22)

### Augmentation Techniques Summary

| Technique | Parameters | Applied To | Justification |
| --- | --- | --- | --- |
| **Random Rotation** | ±10-15° | All views | Brain orientation varies across scans |
| **Horizontal Flip** | p=0.5 | Axial, coronal | Left-right symmetry in brain structure |
| **Gaussian Noise** | σ=0.01-0.05 | All views | Simulates scanner noise variability |
| **Brightness Adjustment** | factor ∈ [0.9, 1.1] | All views | Models intensity variations |
| **Z-score Normalization** | Always applied | All views | Standardizes intensity distributions |

**Important Notes:**

* **Training Only:** Augmentation is applied only to training data, not validation/test sets
* **Per-View Augmentation:** Each anatomical view (axial/coronal/sagittal) is augmented independently
* **Random Seed Control:** Reproducible augmentation via seeded random number generators

**Sources:** High-level Diagram 4 from context, [README.md L20-L22](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L20-L22)

---

## Integration with Training Pipeline

The data loading and augmentation components integrate seamlessly with the training loop, providing batched, balanced, and augmented data to the model.

### Training Loop Data Flow

```mermaid
flowchart TD

SPLIT_FILE["Load subject_split.csv"]
INIT_DATASET["Initialize Dataset objects (train, val, test)"]
CALC_WEIGHTS["Calculate class weights for WeightedRandomSampler"]
CREATE_LOADER["Create DataLoaders with sampler, batch_size, num_workers"]
EPOCH_START["Epoch start"]
SHUFFLE["WeightedRandomSampler generates new sample order"]
GET_BATCH["DataLoader yields batch"]
AUGMENT["Apply augmentations (training only)"]
TO_DEVICE["Move to GPU .to(device)"]
FORWARD["Model forward pass"]
LOSS["Compute loss (class-weighted)"]
BACKWARD["Backward pass"]
UPDATE["Update weights"]
EPOCH_END["Epoch end"]
VAL_START["Validation start"]
VAL_BATCH["Load validation batch (no augmentation, no sampling)"]
VAL_FORWARD["Model forward (eval mode)"]
VAL_METRICS["Compute metrics (Balanced Accuracy)"]

subgraph subGraph3 ["Validation (Per Epoch)"]
    VAL_START
    VAL_BATCH
    VAL_FORWARD
    VAL_METRICS
end

subgraph subGraph2 ["Training Loop (Per Epoch)"]
    EPOCH_START
    SHUFFLE
    EPOCH_END
    SHUFFLE -.-> GET_BATCH
    UPDATE -.-> EPOCH_END

subgraph subGraph1 ["Batch Loop"]
    GET_BATCH
    AUGMENT
    TO_DEVICE
    FORWARD
    LOSS
    BACKWARD
    UPDATE
    AUGMENT -.-> TO_DEVICE
    FORWARD -.-> LOSS
    LOSS -.-> BACKWARD
    BACKWARD -.-> UPDATE
    UPDATE -.-> GET_BATCH
end
end

subgraph subGraph0 ["Initialization (Once per Training Run)"]
    SPLIT_FILE
    INIT_DATASET
    CALC_WEIGHTS
    CREATE_LOADER
end
```

**Sources:** [README.md L162-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L162-L168)

 High-level Diagram 4 from context

### DataLoader Configuration

Typical DataLoader configuration used across training scripts:

| Parameter | Training | Validation/Test | Notes |
| --- | --- | --- | --- |
| `batch_size` | 16-32 | 32-64 | Larger batches for eval (no gradients) |
| `shuffle` | `False` | `False` | Sampler handles training order |
| `sampler` | `WeightedRandomSampler` | `None` | Balancing only for training |
| `num_workers` | 4-8 | 2-4 | Parallel data loading |
| `pin_memory` | `True` | `True` | Faster GPU transfer |
| `drop_last` | `True` | `False` | Consistent batch sizes in training |

**Sources:** [README.md L162-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L162-L168)

 [README.md L112-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L112-L118)

---

## Multi-View Data Loading

For multi-stream models, the data loader must provide synchronized views from all three anatomical planes. Each sample returns a dictionary of image tensors.

### Multi-View Batch Structure

```mermaid
flowchart TD

SAMPLE["Dataset.getitem(idx)"]
AXL_IMG["'axl': tensor[1, H, W]"]
COR_IMG["'cor': tensor[1, H, W]"]
SAG_IMG["'sag': tensor[1, H, W]"]
CLIN["clinical: tensor[5]"]
LABEL["label: int"]
BATCH["Collated Batch"]
AXL_BATCH["images['axl']: tensor[B, 1, H, W]"]
COR_BATCH["images['cor']: tensor[B, 1, H, W]"]
SAG_BATCH["images['sag']: tensor[B, 1, H, W]"]
CLIN_BATCH["clinical: tensor[B, 5]"]
LABEL_BATCH["labels: tensor[B]"]

CLIN -.-> CLIN_BATCH
LABEL -.-> LABEL_BATCH

subgraph subGraph3 ["Batched Output (from DataLoader)"]
    BATCH
    CLIN_BATCH
    LABEL_BATCH
    BATCH -.->|"batch"| AXL_BATCH
    BATCH -.->|"batch"| COR_BATCH
    BATCH -.->|"batch"| SAG_BATCH
    BATCH -.-> CLIN_BATCH
    BATCH -.-> LABEL_BATCH

subgraph subGraph2 ["Stacked Images"]
    AXL_BATCH
    COR_BATCH
    SAG_BATCH
end
end

subgraph subGraph1 ["Single Sample Output"]
    SAMPLE
    CLIN
    LABEL
    SAMPLE -.->|"batch"| AXL_IMG
    SAMPLE -.->|"batch"| COR_IMG
    SAMPLE -.-> SAG_IMG
    SAMPLE -.-> CLIN
    SAMPLE -.-> LABEL

subgraph subGraph0 ["Image Dictionary"]
    AXL_IMG
    COR_IMG
    SAG_IMG
end
end
```

**Sources:** High-level Diagram 2 from context, [README.md L10-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L15)

### Custom Collate Function

A custom collate function is required to handle the dictionary structure:

```css
# Pseudo-code for custom collate functiondef multiview_collate_fn(batch):    """    Collates multi-view samples into batched tensors.        Args:        batch: List of tuples (images_dict, clinical, label)        Returns:        images: Dict of batched tensors {'axl': [B,1,H,W], ...}        clinical: Batched tensor [B, 5]        labels: Batched tensor [B]    """    images_dict = {}    clinical_list = []    labels_list = []        for sample in batch:        images, clinical, label = sample        for view in ['axl', 'cor', 'sag']:            if view not in images_dict:                images_dict[view] = []            images_dict[view].append(images[view])        clinical_list.append(clinical)        labels_list.append(label)        # Stack into batched tensors    batched_images = {        view: torch.stack(tensors)         for view, tensors in images_dict.items()    }    batched_clinical = torch.stack(clinical_list)    batched_labels = torch.tensor(labels_list)        return batched_images, batched_clinical, batched_labels
```

**Sources:** High-level Diagram 2 from context, [README.md L10-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L15)

---

## Anti-Collapse Mechanisms

The data loading pipeline implements multiple safeguards to prevent model collapse (predicting only majority class):

### Anti-Collapse Strategy Layers

| Layer | Mechanism | Location | Purpose |
| --- | --- | --- | --- |
| **1. Sampling** | `WeightedRandomSampler` | DataLoader | Balance batch composition |
| **2. Loss Weighting** | Class-weighted cross-entropy | Training loop | Penalize majority class errors |
| **3. Focal Loss** | Focus on hard examples | Loss function | Reduce easy negative contribution |
| **4. Metric** | Balanced Accuracy | Evaluation | Detect imbalance-induced collapse |

**Combined Effect:** These mechanisms work together to ensure the model learns discriminative features for both classes despite severe imbalance (typical ratio: 3:1 or 4:1 Non-AD:AD).

**Sources:** [README.md L162-L168](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L162-L168)

 High-level Diagram 4 from context

---

## Performance Considerations

Efficient data loading is critical for GPU utilization and overall training speed.

### Optimization Techniques

```mermaid
flowchart TD

SSD["SSD storage for dataset"]
COMPRESSED["Keep .gz files Use nibabel lazy loading"]
SEQUENTIAL["Sequential read patterns when possible"]
CPU_AUG["CPU-based augmentation in worker processes"]
BATCH_AUG["Batch augmentation where possible"]
DETERMINISTIC["Reproducible RNG for debugging"]
WORKERS["Multi-process Loading num_workers=4-8"]
PIN["Pin Memory Faster CPU→GPU transfer"]
PREFETCH["Prefetch Factor Load next batch while GPU processes current"]
CACHE["In-Memory Caching For small datasets"]

subgraph subGraph2 ["I/O Optimizations"]
    SSD
    COMPRESSED
    SEQUENTIAL
    SSD -.-> COMPRESSED
    COMPRESSED -.-> SEQUENTIAL
end

subgraph subGraph1 ["Augmentation Optimizations"]
    CPU_AUG
    BATCH_AUG
    DETERMINISTIC
end

subgraph subGraph0 ["Data Loading Optimizations"]
    WORKERS
    PIN
    PREFETCH
    CACHE
    WORKERS -.-> PIN
    PIN -.-> PREFETCH
    PREFETCH -.-> CACHE
end
```

**Sources:** [README.md L55-L77](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L55-L77)

 High-level Diagram 4 from context

### Bottleneck Analysis

Common bottlenecks and solutions:

| Bottleneck | Symptom | Solution |
| --- | --- | --- |
| **Disk I/O** | Low GPU utilization (<80%) | Increase `num_workers`, use SSD |
| **CPU Augmentation** | Slow batch loading | Simplify augmentation, use GPU-based transforms |
| **Memory Copying** | Delays before training step | Enable `pin_memory=True` |
| **Small Batch Size** | Underutilized GPU | Increase batch size if memory allows |

**Sources:** [README.md L55-L77](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L55-L77)

---

## Reproducibility and Debugging

Ensuring reproducible data loading and augmentation is critical for scientific validity.

### Seed Management

```sql
# Pseudo-code for reproducible data loadingdef set_seeds(seed=42):    """Set all random seeds for reproducibility."""    random.seed(seed)    np.random.seed(seed)    torch.manual_seed(seed)    torch.cuda.manual_seed_all(seed)    def get_reproducible_loader(dataset, seed=42):    """Create DataLoader with reproducible sampling."""    g = torch.Generator()    g.manual_seed(seed)        sampler = WeightedRandomSampler(        weights=compute_weights(dataset),        num_samples=len(dataset),        replacement=True,        generator=g  # Reproducible sampling order    )        return DataLoader(        dataset,        batch_size=32,        sampler=sampler,        num_workers=4,        worker_init_fn=seed_worker,  # Seed each worker        generator=g    )def seed_worker(worker_id):    """Seed each DataLoader worker process."""    worker_seed = torch.initial_seed() % 2**32    np.random.seed(worker_seed)    random.seed(worker_seed)
```

**Key Points:**

* **Generator Object:** Pass `torch.Generator` to `WeightedRandomSampler` and `DataLoader`
* **Worker Seeding:** Use `worker_init_fn` to seed each data loading worker
* **Augmentation Seeding:** Control random transforms with explicit seeds
* **Documentation:** Record seeds in experiment logs for reproducibility

**Sources:** [README.md L112-L118](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L112-L118)

 [README.md L126-L148](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L126-L148)

---

## Summary

The data loading and augmentation pipeline implements a robust system for preparing MRI data for training:

1. **Multi-View Loading:** Supports axial, coronal, and sagittal planes simultaneously
2. **Class Balancing:** `WeightedRandomSampler` ensures balanced training despite severe imbalance
3. **Augmentation:** Geometric and intensity transforms improve generalization
4. **Integration:** Seamless connection with training loop via PyTorch `DataLoader`
5. **Performance:** Multi-process loading and optimizations for efficient GPU utilization
6. **Reproducibility:** Comprehensive seed management for scientific validity

This pipeline forms the foundation for training the multi-stream, multimodal deep learning models described in [Deep Learning Backbones](#5.1) and [Multi-Stream Multimodal Network](#3.1).

**Sources:** All sections above, [README.md L1-L218](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L1-L218)

 High-level Diagrams 1-6 from context

Refresh this wiki

Last indexed: 5 January 2026 ([cd9d51](https://github.com/ThalesMMS/brain-mri-pipelines-py/commit/cd9d51a5))

### On this page

* [Data Loading & Augmentation](#4.5-data-loading-augmentation)
* [Purpose and Scope](#4.5-purpose-and-scope)
* [Data Loading Pipeline Overview](#4.5-data-loading-pipeline-overview)
* [End-to-End Data Flow](#4.5-end-to-end-data-flow)
* [Dataset Class Implementation](#4.5-dataset-class-implementation)
* [Dataset Structure](#4.5-dataset-structure)
* [Key Implementation Details](#4.5-key-implementation-details)
* [WeightedRandomSampler for Class Imbalance](#4.5-weightedrandomsampler-for-class-imbalance)
* [Sampling Mechanism](#4.5-sampling-mechanism)
* [WeightedRandomSampler Configuration](#4.5-weightedrandomsampler-configuration)
* [Image Augmentation Techniques](#4.5-image-augmentation-techniques)
* [Augmentation Pipeline](#4.5-augmentation-pipeline)
* [Augmentation Techniques Summary](#4.5-augmentation-techniques-summary)
* [Integration with Training Pipeline](#4.5-integration-with-training-pipeline)
* [Training Loop Data Flow](#4.5-training-loop-data-flow)
* [DataLoader Configuration](#4.5-dataloader-configuration)
* [Multi-View Data Loading](#4.5-multi-view-data-loading)
* [Multi-View Batch Structure](#4.5-multi-view-batch-structure)
* [Custom Collate Function](#4.5-custom-collate-function)
* [Anti-Collapse Mechanisms](#4.5-anti-collapse-mechanisms)
* [Anti-Collapse Strategy Layers](#4.5-anti-collapse-strategy-layers)
* [Performance Considerations](#4.5-performance-considerations)
* [Optimization Techniques](#4.5-optimization-techniques)
* [Bottleneck Analysis](#4.5-bottleneck-analysis)
* [Reproducibility and Debugging](#4.5-reproducibility-and-debugging)
* [Seed Management](#4.5-seed-management)
* [Summary](#4.5-summary)

Ask Devin about brain-mri-pipelines-py