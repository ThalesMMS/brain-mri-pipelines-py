# Brain MRI Analysis Pipelines for Alzheimer's Disease Detection

Python framework for brain MRI analysis and Alzheimer’s disease (AD) classification using the OASIS-2 dataset. The project implements a multi-stream, multimodal architecture that combines multi-view MRI data (axial, coronal, sagittal) with clinical tabular data, with an optional reinforcement-learning (PPO) component for automated hyperparameter adjustment.



## Key Features

### 1. Model Suite
- **Deep Learning (Multi-Stream):** Supports up to three anatomical planes (`axl`, `cor`, `sag`) simultaneously.
    - **Backbones:** EfficientNet-B0, DenseNet121, and MedicalNet ResNet (Med3D weights converted from 3D to 2D).
    - **Multimodal fusion:** Concatenates visual embeddings with clinical features (`age`, `education`, `nwbv`, `etiv`, `asf`) before classification.
- **Classical baselines:**
    - **SVM:** Classification based on morphological descriptors (ventricle geometry) and covariates.
    - **XGBoost:** Regression for age estimation.

### 2. Reinforcement Learning (RL) Refinement
A PPO (Proximal Policy Optimization) agent can adjust hyperparameters (learning rate, weight decay) per micro-epoch to improve validation balanced accuracy.

### 3. Comprehensive Tooling
- **Tkinter GUI:** For interactive slice navigation, semi-automatic segmentation (region growing), and single-run training.
- **Reproducible CLI Scripts:** For headless execution of baselines, fine-tuning, and RL experiments.
- **Leakage Prevention:** Enforces **subject-level splits** to ensure all MRI scans from a single patient remain strictly within one partition (Train, Validation, or Test).

---

## Data Layout & Setup

This repository does not bundle OASIS-2 data. You must organize the dataset in the project root as follows:

```text
<repo-root>/
├── axl/                        # Axial images (Required for GUI)
├── cor/                        # Coronal images (Optional, supported by deep models)
├── sag/                        # Sagittal images (Optional, supported by deep models)
├── oasis_longitudinal_demographic.csv
└── output/                     # Generated artifacts (logs, models, plots)
```

### File Naming Convention

The pipeline relies on this naming structure to map images to subjects:

-   `OAS2_0001_MR1_axl.nii.gz` (or `.nii`, `.png`, `.jpg`)
    
-   `Subject_ID`: `OAS2_0001`
    
-   `MRI_ID`: `OAS2_0001_MR1`
    

* * *

## Installation

### Prerequisites

-   Python **3.11+**
    
-   `pip`
    
-   **Tkinter** (Required for GUI. On Linux: `sudo apt-get install python3-tk`; macOS: `brew install python-tk@3.11`)
    

### Setup Environment

```Bash
    git clone https://github.com/ThalesMMS/brain-mri-pipelines-py.git
    cd brain-mri-pipelines-py

    # Create virtual environment
    python3.11 -m venv .venv
    source .venv/bin/activate        # macOS/Linux
    # .venv\Scripts\activate         # Windows PowerShell

    # Install dependencies
    pip install -r requirements.txt
```

* * *

## Usage

### 1\. Graphical User Interface (GUI)

Useful for data exploration, visual segmentation, and quick experiments.

```Bash
    python main.py
```

-   **Navigation:** Browse MRI volumes and mark non-viable studies.
    
-   **Segmentation:** Perform region-growing ventricle segmentation and extract descriptors.
    
-   **Training:** Configure and train models via the sidebar.
    
### 2\. Headless CLI Workflows

Recommended for reproducibility and long training runs.

**A) Build Dataset & Run Baselines** Generates the subject-aware split CSV and runs classical models.

```Bash
    python run_baselines_cli.py
```

-   *Note: This trains SVMs in two scenarios: with and without MMSE/CDR to analyze target proxy leakage.*
    

**B) Train Deep Models** Trains the deep backbones on the current split.

```Bash
    # Standard training
    python run_deep_models_cli.py --seed 42 --epochs 40 --backbones efficientnet,medicalnet,densenet
    
    # With Multimodal Fusion (Clinical Data)
    python run_deep_models_cli.py --seed 42 --epochs 40 --backbones efficientnet --multimodal
```

* * *

## Research Pipeline Stages

The repository includes specific scripts for a three-stage experimental workflow located in `brain_mri/scripts/`.

### Stage 1: Embeddings Analysis

Compares Deep Learning embeddings against handcrafted descriptors using a lightweight classifier.

```Bash
    python brain_mri/scripts/run_pc1_embeddings.py --dl-backbone efficientnet
```

### Stage 2: Transfer Learning & Fine-tuning

Runs deep training with an explicit "frozen backbone" warmup phase followed by unfreezing.

```Bash
    python brain_mri/scripts/run_pc2_finetune.py --backbone efficientnet --seed 42 --epochs 6 --warmup-epochs 2
```

### Stage 3: RL Refinement

Uses the PPO Actor-Critic agent to refine the model trained in Stage 2.

```Bash
    python brain_mri/scripts/run_pc3_rl_refinement.py --backbone efficientnet --seed 42 --episodes 4 --horizon 4
```

### Paper Artifacts

Generate LaTeX tables from the experiment logs:

```Bash
    python -m brain_mri.scripts.generate_article_tables --write
```

* * *

## Methodology Notes

### Metric & Safeguards

-   **Primary Metric:** Balanced Accuracy (due to class imbalance).
    
-   **Anti-Collapse:** The pipeline uses `WeightedRandomSampler`, Class-Weighted Loss, and Focal Loss to prevent the model from predicting only the majority class.
    
-   **Warning:** MMSE and CDR scores are strong proxies for dementia. While the codebase supports using them, we recommend the `svm_without_mmse_cdr` scenario for methodologically cleaner imaging-based analysis.
    

### MedicalNet (3D → 2D)

We utilize pre-trained weights from the **Med3D** project (ResNet architectures trained on 23 medical datasets). These are downloaded via `huggingface_hub` to `~/.cache/medicalnet` and mathematically converted from 3D kernels to 2D for slice-based analysis.

* * *

## Repository Structure

```text
    <repo-root>/
    ├── brain_mri/                       # Main package
    │   ├── ui/                          # Tkinter GUI mixins (Navigation, Segmentation)
    │   ├── experiments/                 # Experiment tracking & Visualization
    │   ├── ml/                          # Core ML logic (Models, RL Agent, Training Loop)
    │   │   ├── medicalnet_models.py     # Med3D implementation
    │   │   ├── multistream_models.py    # Multi-view fusion logic
    │   │   └── rl_refinement.py         # PPO Agent
    │   ├── scripts/                     # Reproducible stage scripts
    │   └── utils/                       # Image processing utilities
    ├── axl/                             # Dataset directory (user provided)
    ├── tests/                           # Unit tests
    ├── main.py                          # GUI Entrypoint
    ├── run_baselines_cli.py             # Classical ML runner
    ├── run_deep_models_cli.py           # Deep Learning runner
    └── requirements.txt
```

## Authors

-   **[Antônio Soares Couto Neto](https://github.com/nietus)**
    
-   **[Giovanna Naves Ribeiro](https://github.com/GiovannaNaves)**
    
-   **[Julia Rodrigues Vasconcellos Melo](https://github.com/Juliarvm)**
    
-   **[Thales Matheus Mendonça Santos](https://github.com/ThalesMMS)**
    

## Citation

If you use the MedicalNet weights integration, please cite:

> Chen, S., Ma, K., & Zheng, Y. (2019). Med3D: Transfer Learning for 3D Medical Image Analysis. *arXiv preprint* arXiv:1904.00625.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
