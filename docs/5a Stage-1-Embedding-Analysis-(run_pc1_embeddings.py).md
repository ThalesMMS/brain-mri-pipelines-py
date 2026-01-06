# Stage 1: Embedding Analysis (run_pc1_embeddings.py)

> **Relevant source files**
> * [README.md](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md)

## Purpose and Scope

Stage 1 of the three-stage research pipeline validates the quality of deep learning embeddings before proceeding to full model fine-tuning. This stage compares pretrained backbone embeddings against handcrafted morphological descriptors using lightweight classifiers to determine whether learned representations capture meaningful information for Alzheimer's disease classification.

This page documents the embedding analysis methodology. For information about the backbone architectures themselves, see [Deep Learning Backbones](#5.1). For the subsequent transfer learning stage, see [Stage 2: Transfer Learning & Fine-Tuning](6b%20Baselines-CLI-%28run_baselines_cli.py%29.md). For classical baseline implementation details, see [Classical Machine Learning Baselines](5c%20Stage-3-RL-Hyperparameter-Refinement-%28run_pc3_rl_refinement.py%29.md).

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L126-L132)

---

## Overview

Stage 1 implements a **feature quality assessment** methodology that answers the question: "Do pretrained deep learning backbones extract better representations than traditional feature engineering for AD detection?" This validation step justifies the computational expense of full fine-tuning in Stage 2.

The stage follows this workflow:

1. **Extract embeddings** from frozen pretrained backbones (EfficientNet-B0, DenseNet121, MedicalNet ResNet)
2. **Extract handcrafted features** (morphological descriptors based on ventricle geometry)
3. **Train lightweight classifiers** (e.g., logistic regression, SVM) on both feature types
4. **Compare performance** using balanced accuracy as the primary metric

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L126-L132)

---

## Stage 1 Pipeline Architecture

```mermaid
flowchart TD

RAW["Raw MRI Scans (axl/, cor/, sag/)"]
META["Clinical Metadata oasis_longitudinal_demographic.csv"]
BACKBONE_EFF["EfficientNet-B0 Frozen Pretrained"]
BACKBONE_DENSE["DenseNet121 Frozen Pretrained"]
BACKBONE_MED["MedicalNet ResNet Frozen Pretrained"]
EMBED["Embedding Vectors (e.g., 1280-dim for EfficientNet)"]
SEG["Ventricle Segmentation Region Growing"]
MORPH["Morphological Descriptors Geometry Features"]
CLASSIFIER_DL["Lightweight Classifier on DL Embeddings"]
CLASSIFIER_HAND["Lightweight Classifier on Handcrafted Features"]
METRICS["Performance Metrics Balanced Accuracy"]
COMPARE["Statistical Comparison Embedding vs Handcrafted"]

RAW -.-> BACKBONE_EFF
RAW -.-> BACKBONE_DENSE
RAW -.-> BACKBONE_MED
RAW -.-> SEG
META -.-> MORPH
EMBED -.-> CLASSIFIER_DL
MORPH -.-> CLASSIFIER_HAND

subgraph subGraph4 ["Evaluation & Comparison"]
    METRICS
    COMPARE
    METRICS -.-> COMPARE
end

subgraph subGraph3 ["Classification Layer"]
    CLASSIFIER_DL
    CLASSIFIER_HAND
end

subgraph subGraph2 ["Feature Extraction: Classical Path"]
    SEG
    MORPH
    SEG -.-> MORPH
end

subgraph subGraph1 ["Feature Extraction: Deep Learning Path"]
    BACKBONE_EFF
    BACKBONE_DENSE
    BACKBONE_MED
    EMBED
end

subgraph subGraph0 ["Input Layer"]
    RAW
    META
end
```

**Pipeline Flow**: The dual-path architecture enables direct comparison between learned representations and engineered features. Both paths use the same subject-level train/validation/test splits to ensure fair comparison.

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L126-L132)

---

## Embedding Extraction Process

### Frozen Backbone Configuration

Stage 1 uses **frozen pretrained backbones** to extract embeddings without any task-specific training. This isolates the quality of the pretrained representations from the classification task.

```mermaid
flowchart TD

INPUT["MRI Slice (e.g., 224x224)"]
NORM["Normalization ImageNet stats"]
CONV["Convolutional Layers (weights frozen)"]
POOL["Global Average Pooling"]
FEATURES["Feature Vector (embedding)"]
CACHE["Embedding Cache (optional optimization)"]

NORM -.-> CONV
FEATURES -.-> CACHE

subgraph Storage ["Storage"]
    CACHE
end

subgraph subGraph1 ["Frozen Backbone"]
    CONV
    POOL
    FEATURES
    CONV -.-> POOL
    POOL -.-> FEATURES
end

subgraph subGraph0 ["Input Processing"]
    INPUT
    NORM
    INPUT -.-> NORM
end
```

**Backbone Selection**: The `--dl-backbone` argument specifies which pretrained model to use:

* `efficientnet`: EfficientNet-B0 (1280-dimensional embeddings)
* `densenet`: DenseNet121 (1024-dimensional embeddings)
* `medicalnet`: MedicalNet ResNet (512-dimensional embeddings for ResNet-18)

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L131-L131)

### Multi-View Embedding Aggregation

When multiple anatomical planes are available, embeddings from each plane can be aggregated:

| Aggregation Strategy | Description | Dimensionality |
| --- | --- | --- |
| **Concatenation** | Stack embeddings from axial, coronal, sagittal | 3 × embedding_dim |
| **Average Pooling** | Element-wise mean across planes | embedding_dim |
| **Max Pooling** | Element-wise maximum across planes | embedding_dim |
| **Single Plane** | Use only one plane (typically axial) | embedding_dim |

Stage 1 typically uses **single plane** (axial) extraction for simplicity and computational efficiency.

Sources: [README.md L10-L12](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L10-L12)

---

## Handcrafted Feature Baseline

### Morphological Descriptor Extraction

The classical feature path extracts **geometry-based descriptors** from ventricle segmentation masks. These features represent traditional neuroimaging analysis approaches.

```mermaid
flowchart TD

INPUT["Axial MRI Slice"]
SEED["Seed Point Selection (ventricle region)"]
REGION["Region Growing Algorithm"]
MASK["Binary Segmentation Mask"]
AREA["Ventricle Area"]
PERIM["Perimeter Length"]
CIRC["Circularity (4π × Area / Perimeter²)"]
MOMENTS["Shape Moments (Hu invariants)"]
RATIO["Aspect Ratio"]
FEAT_VEC["Feature Vector (~10-20 dimensions)"]
AGE["Age"]
EDU["Education Years"]
NWBV["Normalized Whole Brain Volume"]
ETIV["Estimated Total Intracranial Volume"]
ASF["Atlas Scaling Factor"]
CLIN_VEC["Clinical Vector (5 dimensions)"]
COMBINED["Combined Feature Vector (morphological + clinical)"]

MASK -.-> AREA
MASK -.-> PERIM
MASK -.-> CIRC
MASK -.-> MOMENTS
MASK -.-> RATIO

subgraph subGraph3 ["Feature Fusion"]
    COMBINED
end

subgraph subGraph2 ["Clinical Features"]
    AGE
    EDU
    NWBV
    ETIV
    ASF
    CLIN_VEC
    AGE -.-> CLIN_VEC
    EDU -.-> CLIN_VEC
    NWBV -.-> CLIN_VEC
    ETIV -.-> CLIN_VEC
    ASF -.-> CLIN_VEC
end

subgraph subGraph1 ["Feature Computation"]
    AREA
    PERIM
    CIRC
    MOMENTS
    RATIO
    FEAT_VEC
    AREA -.-> FEAT_VEC
    PERIM -.-> FEAT_VEC
    CIRC -.-> FEAT_VEC
    MOMENTS -.-> FEAT_VEC
    RATIO -.-> FEAT_VEC
end

subgraph subGraph0 ["Segmentation Pipeline"]
    INPUT
    SEED
    REGION
    MASK
    INPUT -.-> SEED
    SEED -.-> REGION
    REGION -.-> MASK
end
```

**Key Insight**: Handcrafted features are **low-dimensional** (typically 15-25 dimensions) compared to deep learning embeddings (512-1280 dimensions), yet may capture domain-specific geometric properties that pretrained networks don't emphasize.

Sources: [README.md L14-L15](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L14-L15)

 [README.md L36](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L36-L36)

### Feature Vector Composition

The handcrafted feature vector combines:

1. **Ventricle Geometry** (10-20 dimensions): * Area, perimeter, circularity * Shape moments (Hu invariants) * Aspect ratio, compactness * Centroid coordinates
2. **Clinical Covariates** (5 dimensions): * Age * Education (years) * nWBV (normalized whole brain volume) * eTIV (estimated total intracranial volume) * ASF (atlas scaling factor)

**Warning**: Stage 1 should **not** include MMSE or CDR scores in the feature vector, as these are strong proxies for the target label and would create methodological issues (see [Classical Machine Learning Baselines](5c%20Stage-3-RL-Hyperparameter-Refinement-%28run_pc3_rl_refinement.py%29.md) for discussion of target leakage).

Sources: [README.md L12](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L12-L12)

 [README.md L36](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L36-L36)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L168-L168)

---

## Lightweight Classifier Training

### Classifier Selection

Stage 1 uses **simple, non-parametric classifiers** to isolate embedding quality from classifier complexity:

| Classifier | Configuration | Rationale |
| --- | --- | --- |
| **Logistic Regression** | L2 regularization, C=1.0 | Linear baseline, interpretable coefficients |
| **Linear SVM** | C=1.0, balanced class weights | Robust to outliers, handles imbalance |
| **k-Nearest Neighbors** | k=5, distance weighting | Non-parametric, no training required |

The primary classifier is typically **Logistic Regression** for simplicity and interpretability.

Sources: [README.md L14](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L14-L14)

### Training Configuration

```mermaid
flowchart TD

SPLIT["Subject-Level Split Train/Val/Test"]
BALANCE["WeightedRandomSampler (handle class imbalance)"]
TRAIN_DL["Train on DL Embeddings LogisticRegression max_iter=1000"]
TRAIN_HC["Train on Handcrafted LogisticRegression max_iter=1000"]
GRID["Grid Search C: [0.01, 0.1, 1.0, 10.0]"]
VAL["Validation Set Selection"]
TEST_DL["Test Set DL Embedding Model"]
TEST_HC["Test Set Handcrafted Model"]
METRICS["Balanced Accuracy Precision, Recall, F1"]

BALANCE -.-> TRAIN_DL
BALANCE -.-> TRAIN_HC
VAL -.-> TEST_DL
VAL -.-> TEST_HC

subgraph Evaluation ["Evaluation"]
    TEST_DL
    TEST_HC
    METRICS
end

subgraph subGraph2 ["Hyperparameter Tuning"]
    GRID
    VAL
    GRID -.-> VAL
end

subgraph subGraph1 ["Classifier Training"]
    TRAIN_DL
    TRAIN_HC
end

subgraph subGraph0 ["Data Preparation"]
    SPLIT
    BALANCE
    SPLIT -.-> BALANCE
end
```

**Key Configuration Choices**:

* **Class weights**: Set to `'balanced'` to handle AD/Non-AD imbalance
* **Regularization**: L2 penalty with C=1.0 as default, tuned via grid search
* **Convergence**: `max_iter=1000` ensures convergence on small datasets
* **Metric**: Balanced accuracy (see [Evaluation Metrics](#5.6))

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L164-L167)

---

## Command-Line Interface

### Basic Usage

```
python brain_mri/scripts/run_pc1_embeddings.py --dl-backbone efficientnet
```

### Available Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--dl-backbone` | str | `efficientnet` | Backbone for embedding extraction: `efficientnet`, `densenet`, `medicalnet` |
| `--seed` | int | `42` | Random seed for reproducibility |
| `--plane` | str | `axl` | Anatomical plane to use: `axl`, `cor`, `sag` |
| `--split-csv` | str | `output/subject_split.csv` | Path to subject-level split file |
| `--output-dir` | str | `output/stage1/` | Directory for results |

### Example Commands

```
# Compare EfficientNet embeddings vs handcrafted featurespython brain_mri/scripts/run_pc1_embeddings.py --dl-backbone efficientnet --seed 42# Compare DenseNet embeddingspython brain_mri/scripts/run_pc1_embeddings.py --dl-backbone densenet --seed 42# Compare MedicalNet embeddingspython brain_mri/scripts/run_pc1_embeddings.py --dl-backbone medicalnet --seed 42
```

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L130-L132)

---

## Outputs and Interpretation

### Generated Artifacts

Stage 1 produces the following outputs in `output/stage1/`:

| File | Description |
| --- | --- |
| `embedding_results_{backbone}.json` | Performance metrics for DL embeddings |
| `handcrafted_results.json` | Performance metrics for morphological features |
| `comparison_plot.png` | Bar chart comparing balanced accuracy |
| `confusion_matrices.png` | Confusion matrices for both approaches |
| `feature_importance.png` | (For handcrafted) Feature importance scores |

### Interpretation Guidelines

**Successful Validation Criteria**:

1. **DL embeddings should outperform handcrafted features** on balanced accuracy
2. **Performance gap should be substantial** (e.g., >5% improvement)
3. **Both approaches should exceed random baseline** (50% balanced accuracy)

**Example Results**:

```yaml
Deep Learning Embeddings (EfficientNet):
  Balanced Accuracy: 0.72
  Precision: 0.68
  Recall: 0.75
  F1 Score: 0.71

Handcrafted Morphological Features:
  Balanced Accuracy: 0.64
  Precision: 0.61
  Recall: 0.68
  F1 Score: 0.64

Conclusion: EfficientNet embeddings provide 8% improvement over
handcrafted features, justifying full fine-tuning in Stage 2.
```

**Decision Logic**:

* **If DL embeddings win**: Proceed to [Stage 2: Transfer Learning & Fine-Tuning](6b%20Baselines-CLI-%28run_baselines_cli.py%29.md)
* **If handcrafted features win**: Re-evaluate backbone choice or pretrained weights
* **If both perform poorly**: Check data quality, splits, and preprocessing

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L126-L132)

---

## Implementation Flow Diagram

```mermaid
flowchart TD

ARGS["Parse CLI Arguments"]
SEED["Set Random Seed"]
LOAD_SPLIT["Load subject_split.csv"]
LOAD_IMAGES["Load MRI Images per subject"]
LOAD_META["Load Clinical Metadata from CSV"]
INIT_BACKBONE["Initialize Frozen Backbone from brain_mri.ml"]
EXTRACT_EMB["Extract Embeddings for all subjects"]
CACHE_EMB["Cache Embeddings (optional)"]
SEGMENT["Run Ventricle Segmentation from brain_mri.ui"]
COMPUTE_MORPH["Compute Morphological Descriptors"]
MERGE_CLIN["Merge Clinical Covariates"]
TRAIN_LR_DL["Train LogisticRegression on DL embeddings"]
TRAIN_LR_HC["Train LogisticRegression on handcrafted features"]
GRID_SEARCH["Hyperparameter Tuning via GridSearchCV"]
PREDICT_DL["Predict on Test Set (DL model)"]
PREDICT_HC["Predict on Test Set (handcrafted model)"]
COMPUTE_METRICS["Compute Metrics balanced_accuracy_score"]
PLOT["Generate Comparison Plots"]
SAVE_JSON["Save Metrics to JSON"]
LOG["Log to Console"]

subgraph Results ["Results"]
    PLOT
    SAVE_JSON
    LOG
end

subgraph Evaluation ["Evaluation"]
    PREDICT_DL
    PREDICT_HC
    COMPUTE_METRICS
end

subgraph subGraph4 ["Classifier Training"]
    TRAIN_LR_DL
    TRAIN_LR_HC
    GRID_SEARCH
end

subgraph subGraph3 ["Feature Extraction: Classical Path"]
    SEGMENT
    COMPUTE_MORPH
    MERGE_CLIN
    SEGMENT -.-> COMPUTE_MORPH
end

subgraph subGraph2 ["Feature Extraction: DL Path"]
    INIT_BACKBONE
    EXTRACT_EMB
    CACHE_EMB
end

subgraph subGraph1 ["Data Loading"]
    LOAD_IMAGES
    LOAD_META
end

subgraph Initialization ["Initialization"]
    ARGS
    SEED
    LOAD_SPLIT
    ARGS -.-> SEED
    SEED -.-> LOAD_SPLIT
end
```

**Code Entities Referenced**:

* Backbone models: `brain_mri/ml/medicalnet_models.py`, `brain_mri/ml/multistream_models.py`
* Segmentation: `brain_mri/ui/` (GUI segmentation mixins)
* Metrics: `sklearn.metrics.balanced_accuracy_score`
* Split loading: `output/subject_split.csv`

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L180-L196)

---

## Integration with Research Pipeline

Stage 1 serves as the **validation gate** for the three-stage pipeline:

```mermaid
flowchart TD

S1["Stage 1: Embedding Analysis run_pc1_embeddings.py"]
S2["Stage 2: Transfer Learning run_pc2_finetune.py"]
S3["Stage 3: RL Refinement run_pc3_rl_refinement.py"]

S1 -.->|"Embeddings validated"| S2
S2 -.->|"Model fine-tuned"| S3
S1 -.-> S2
S2 -.->|"If embeddings fail: Re-evaluate backbone"| S3
S3 -.-> S1
```

**Sequential Dependency**: Stage 2 should only be executed if Stage 1 demonstrates that the chosen backbone extracts meaningful features. This prevents wasting computational resources on ineffective architectures.

**Iterative Refinement**: If Stage 1 results are unsatisfactory, iterate by:

1. Trying different backbones (`efficientnet`, `densenet`, `medicalnet`)
2. Adjusting preprocessing (normalization, augmentation)
3. Verifying data quality and split integrity

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L122-L156)

---

## Methodological Considerations

### Why Lightweight Classifiers?

Stage 1 intentionally uses **simple classifiers** to:

1. **Isolate embedding quality** from classifier capacity
2. **Reduce computational cost** (no GPU needed for training)
3. **Enable fair comparison** between DL and handcrafted features
4. **Prevent overfitting** on small medical datasets

A complex classifier (e.g., deep neural network) could mask poor embeddings by learning task-specific transformations.

### Subject-Level Splitting Critical

Stage 1 inherits the subject-aware split from the data preparation phase (see [Subject-Level Splitting & Leakage Prevention](3d%20Subject-Level-Splitting-&-Leakage-Prevention.md)). This ensures:

* **No data leakage**: Multiple scans from the same patient don't appear in both train and test
* **Realistic performance**: Test metrics reflect generalization to unseen patients
* **Fair comparison**: Both DL and handcrafted paths use identical splits

Sources: [README.md L23](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L23-L23)

 **Sources**: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L164-L167)

### Balanced Accuracy as Primary Metric

Stage 1 uses **balanced accuracy** rather than standard accuracy because:

* The OASIS-2 dataset has **class imbalance** (fewer AD cases than controls)
* Standard accuracy would reward predicting the majority class
* Balanced accuracy averages recall across classes, penalizing imbalanced predictions

See [Evaluation Metrics](#5.6) for detailed explanation.

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L164-L164)

---

## Troubleshooting

### Low Performance on Both Paths

**Symptom**: Both DL embeddings and handcrafted features achieve <60% balanced accuracy.

**Possible Causes**:

1. **Data leakage in split**: Verify `subject_split.csv` has no duplicate subjects across partitions
2. **Label noise**: Check `oasis_longitudinal_demographic.csv` for missing or incorrect CDR labels
3. **Insufficient data**: OASIS-2 has limited AD-positive cases
4. **Preprocessing issues**: Verify image normalization matches backbone expectations

### DL Embeddings Underperform Handcrafted Features

**Symptom**: Handcrafted features achieve higher balanced accuracy than DL embeddings.

**Possible Causes**:

1. **Wrong normalization**: ImageNet-pretrained models expect specific normalization
2. **Domain mismatch**: 2D slices may not leverage 3D pretrained weights effectively
3. **Feature dimensionality**: High-dimensional embeddings may require more regularization

**Solutions**:

* Try different backbones (e.g., switch from ImageNet to MedicalNet)
* Reduce embedding dimensionality via PCA
* Increase regularization strength (decrease C parameter)

### Memory Issues

**Symptom**: Out-of-memory errors during embedding extraction.

**Solutions**:

1. **Reduce batch size** for embedding extraction
2. **Process embeddings incrementally** and save to disk
3. **Use embedding caching** to avoid re-extraction

---

## Summary

Stage 1 provides a **lightweight validation mechanism** that determines whether pretrained deep learning backbones extract meaningful representations for AD classification. By comparing embeddings against handcrafted morphological features using simple classifiers, Stage 1 establishes a performance baseline and justifies the computational expense of full fine-tuning in subsequent stages.

**Key Takeaways**:

* Stage 1 is a **feature quality assessment**, not a full training pipeline
* Simple classifiers isolate embedding quality from model complexity
* Subject-level splitting prevents data leakage
* Balanced accuracy accounts for class imbalance
* Results inform the choice of backbone for Stage 2

**Next Steps**: If Stage 1 validates the embeddings, proceed to [Stage 2: Transfer Learning & Fine-Tuning](6b%20Baselines-CLI-%28run_baselines_cli.py%29.md) for full end-to-end training.

Sources: [Project overview and setup](https://github.com/ThalesMMS/brain-mri-pipelines-py/blob/cd9d51a5/README.md#L122-L156)





### On this page

* [Stage 1: Embedding Analysis (run_pc1_embeddings.py)](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Purpose and Scope](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Overview](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Stage 1 Pipeline Architecture](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Embedding Extraction Process](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Frozen Backbone Configuration](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Multi-View Embedding Aggregation](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Handcrafted Feature Baseline](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Morphological Descriptor Extraction](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Feature Vector Composition](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Lightweight Classifier Training](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Classifier Selection](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Training Configuration](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Command-Line Interface](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Basic Usage](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Available Arguments](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Example Commands](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Outputs and Interpretation](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Generated Artifacts](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Interpretation Guidelines](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Implementation Flow Diagram](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Integration with Research Pipeline](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Methodological Considerations](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Why Lightweight Classifiers?](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Subject-Level Splitting Critical](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Balanced Accuracy as Primary Metric](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Troubleshooting](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Low Performance on Both Paths](6a%20Graphical-User-Interface-%28main.py%29.md)
* [DL Embeddings Underperform Handcrafted Features](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Memory Issues](6a%20Graphical-User-Interface-%28main.py%29.md)
* [Summary](6a%20Graphical-User-Interface-%28main.py%29.md)

Ask Devin about brain-mri-pipelines-py