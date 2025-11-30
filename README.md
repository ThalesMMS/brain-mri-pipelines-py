# Brain MRI Analysis Pipelines for Alzheimer's Disease Detection

A comprehensive machine learning pipeline for analyzing brain MRI scans to detect and classify Alzheimer's disease using the OASIS-2 dataset. This project implements both traditional machine learning and deep learning approaches for classification and regression tasks.

## Overview

This repository contains the implementation of image processing and machine learning techniques for Alzheimer's disease analysis, including:

- **Image preprocessing and segmentation** of brain MRI scans
- **Feature extraction** from segmented regions
- **Traditional ML classifiers** (SVM) and regressors (XGBoost)
- **Deep learning models** (EfficientNet-B0 baseline) with strong regularization/augmentations
- **Comprehensive GUI** for interactive analysis

## Authors

- **[Antônio Soares Couto Neto](https://github.com/nietus)**
- **[Giovanna Naves Ribeiro](https://github.com/GiovannaNaves)**
- **[Julia Rodrigues Vasconcellos Melo](https://github.com/Juliarvm)**
- **[Thales Matheus Mendonça Santos](https://github.com/ThalesMMS)**

## Dataset

- **Source**: OASIS-2 (Open Access Series of Imaging Studies)
- **Format**: Axial brain MRI scans (`.nii.gz`)
- **Demographics**: Clinical and demographic data included
- **Practical constraint**: The cohort is small, so every experiment must be treated as low-data—heavy regularization, patient-aware splits, and cautious claims are essential.

## Limitations & Overfitting Warning

- Expect severe overfitting with the provided splits: the dataset is tiny, class balance is fragile, and standard DenseNet baselines memorize quickly.
- Use strong regularization/augmentation (mixup, dropout, weight decay), conservative early stopping, and patient-aware splits; prefer reporting validation/test metrics only.
- Treat any reported gains as exploratory; reproducibility and external validity are limited until more data is added.

## Key Features

### Machine Learning Models
- **Classification**: SVM (shallow), EfficientNet-B0 (deep)
- **Regression**: XGBoost for age estimation, EfficientNet-B0 for deep regression
- **Optimization**: EfficientNet-B0 baseline with dropout, label smoothing, weight decay, and strong augmentations; mixup optional (off by default) to mitigate overfitting on a small dataset

### Advanced Capabilities
- **Automated segmentation** with Otsu thresholding
- **Manual segmentation** with interactive tools
- **Feature extraction**: Area, circularity, eccentricity, and additional descriptors
- **Learning curves** and performance metrics
- **Cross-validation** with patient-specific data splitting

### Recent Enhancements
- **EfficientNet-B0 baseline**: Dropout 0.25, lr 5e-5 (classificação), label smoothing leve, augmentações fortes, early stopping por balanced accuracy, CosineAnnealing LR
- **Model checkpointing**: Best model saving and EMA weights
- **Performance tracking**: Incremental training history and learning curves

### Default EfficientNet settings
- Input 224x224; pretrained EfficientNet-B0 backbone
- Dropout: 0.25; Label smoothing: 0.05 (classificação)
- Learning rate: 5e-5 (classificação), 1e-3 (regressão); Weight decay: 1e-4 (classificação)
- Mixup: desligado por padrão; habilite via `DENSENET_MIXUP` (mantido por compatibilidade)
- Early stopping: balanced accuracy de validação; seeds e splits fixos (`random_state=42`)
- Métrica: balanced accuracy como métrica principal de validação/early-stopping (accuracy simples também logada para referência)

## Repository Structure

```
brain-mri-pipelines/
├── README.md                           # This file
├── main.py                             # Main GUI application
├── axl/                                # OASIS-2 dataset (axial slices, .nii.gz)
├── oasis_longitudinal_demographic.csv  # Clinical and demographic data
├── requirements.txt                    # Python dependencies
├── output/                             # Model outputs and results
└── Artigo/                             # Research paper (LaTeX)
```

## Installation

### Prerequisites
- Python 3.11+
- pip
- Tkinter (GUI library)

### Install Tkinter
- **macOS**: `brew install python-tk@3.11`
- **Ubuntu/Debian**: `sudo apt-get install python3-tk`
- **Windows**: Included with Python installation

### Setup Environment
```bash
# Clone the repository
git clone https://github.com/ThalesMMS/brain-mri-pipelines-py.git
cd brain-mri-pipelines-py

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Usage

### Basic Workflow
1. **Launch the GUI**: Run `python main.py`
2. **Create Dataset**: Use the "Criar Dataset" button to process MRI data
3. **Train Models**: Select and train desired models (SVM, EfficientNet, XGBoost)
4. **Analyze Results**: View performance metrics and learning curves

### Advanced: RL-based EfficientNet Optimization
1. Generate the dataset using "Criar Dataset"
2. Train the base EfficientNet classification model
3. Click **"Refinar EfficientNet (RL)"** to optimize with PPO:
   - Optimizes: Learning rate, weight decay, dropout, mixup, label smoothing
   - Outputs: Best model, training policy, optimization history
4. Monitor progress through learning curves and metrics

## Model Performance

### Classification
- **SVM**: Traditional ML with handcrafted features
- **EfficientNet-B0**: Deep learning with end-to-end training
- **Metrics**: Accuracy, sensitivity, specificity, confusion matrix

### Regression
- **XGBoost**: Age estimation from extracted features
- **EfficientNet-B0**: Deep regression for temporal coherence analysis

## Technical Implementation

### Image Processing
- NIfTI image loading and visualization
- Multi-threshold segmentation (Otsu + manual)
- Region-specific analysis (lateral ventricles)

### Feature Engineering
- **Geometric descriptors**: Area, circularity, eccentricity
- **Advanced metrics**: Additional morphological features
- **Statistical analysis**: Bivariate scatter plots by class

### Model Architecture
- **Data splitting**: Patient-aware train/validation/test (80/20)
- **Class balance**: 4:1 ratio maintained
- **Cross-validation**: Prevents data leakage from same patients

## Contributing

This project was developed as part of academic research on Alzheimer's disease detection. For questions or collaborations, please contact the authors.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **OASIS Project** for providing the brain MRI dataset
- **Open Access Series of Imaging Studies** (OASIS-2)
- Research community for Alzheimer's disease imaging analysis
