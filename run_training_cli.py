
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import tkinter as tk


# Ensure brain_mri is in path
sys.path.append(os.getcwd())

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from brain_mri.ml.ml_training import MLTrainingMixin

# Mocking GUI components for headless execution
class MockRoot:
    def update(self): pass
    def title(self, _): pass
    def geometry(self, _): pass

class MockToplevel:
    def __init__(self, master, **kwargs): pass
    def update(self): pass
    def destroy(self): pass

class MockLabel:
    def __init__(self, master, text="", **kwargs): pass
    def pack(self, **kwargs): pass

class HeadlessApp(MLTrainingMixin):
    def __init__(self, base_dir):
        self.root = MockRoot()
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "axl"  # This drives dataset loading logic
        self.output_dir = self.base_dir / "output"
        self.descriptors_csv = self.output_dir / "ventricle_descriptors.csv"
        self.experiment_history_path = self.output_dir / "training_experiments.json"
        
        # Ensure output dir exists
        self.output_dir.mkdir(exist_ok=True)
        
        # Patch tkinter
        self._patch_tkinter()

    def _patch_tkinter(self):
        from brain_mri.ml import ml_training
        
        ml_training.tk.Toplevel = MockToplevel
        ml_training.tk.Label = MockLabel
        
        # Mock messageboxes
        def showinfo(title, message): print(f"[INFO] {title}: {message}")
        def showwarning(title, message): print(f"[WARN] {title}: {message}")
        def showerror(title, message): print(f"[ERROR] {title}: {message}")
        def askyesno(title, message): return True
        
        ml_training.messagebox.showinfo = showinfo
        ml_training.messagebox.showwarning = showwarning
        ml_training.messagebox.showerror = showerror
        ml_training.messagebox.askyesno = askyesno

    def _show_plot_window(self, title, figure):
        safe_title = title.replace(' ', '_').replace('/', '-')
        save_path = self.output_dir / f"plot_{safe_title}.png"
        print(f"[PLOT] Saving plot to {save_path}")
        figure.savefig(save_path)

def main():
    base_dir = os.getcwd()
    print(f"Running headless training in {base_dir}")
    
    app = HeadlessApp(base_dir)
    
    # Configure environment for this run
    # Use real data split if available, otherwise default
    # The default logic in ml_training looks for 'exam_level_dataset_split.csv' in output_dir
    # OR uses SPLIT_CSV_PATH.
    # The user has 'oasis_longitudinal_demographic.csv' in root.
    # It seems ml_training has logic for 'split' column in that CSV, but typically data preparation
    # creates 'exam_level_dataset_split.csv'. 
    # Let's check if 'exam_level_dataset_split.csv' exists.
    
    split_csv = Path(base_dir) / "output" / "exam_level_dataset_split.csv"
    if not split_csv.exists():
        print(f"[WARN] Split CSV {split_csv} NOT found. Training might fail or require running setup first.")
        # Try to point to the main csv if it has 'split' column?
        # Typically the app flow creates the split. If the user ran before, it should exist.
    
    # Set hyperparams for the run
    os.environ["USE_MULTIMODAL"] = "1"
    os.environ["EPOCHS"] = "40" # Increased to 40 for better convergence
    
    # Unfreeze backbone to allow learning features from scratch/fine-tuning
    os.environ["RESNET_FREEZE"] = "0"

    # Hyperparameters for Stability (Data Science Tuning)
    # OneCycleLR will start at 1e-3/25 = 4e-5, peak at 1e-3, and decay.
    os.environ["RESNET_LR"] = "1e-3"           # Peak LR for OneCycle
    os.environ["RESNET_WEIGHT_DECAY"] = "1e-2" # Stronger regularization (1e-2 is typical for OneCycle)
    os.environ["RESNET_DROPOUT"] = "0.5"       # Higher dropout for small dataset
    os.environ["RESNET_PATIENCE"] = "25"       # Increase patience to survive warmup (40 epochs total)

    
    print("\n--- Starting ResNet18 (MedicalNet) Run ---")
    try:
        app._train_pytorch_model(
            mode='classification',
            backbone='medicalnet'
        )
    except Exception as e:
        print(f"Error running ResNet18: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
