
import os
import sys
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import tkinter as tk

# Mocking mixin dependencies
# We need to make sure brain_mri is in path
sys.path.append(os.getcwd())

from brain_mri.ml.ml_training import MLTrainingMixin

class MockRoot:
    def update(self): pass
    def title(self, _): pass
    def geometry(self, _): pass

class MockApp(MLTrainingMixin):
    def __init__(self, base_dir):
        self.root = MockRoot()
        self.base_dir = Path(base_dir)
        self.dataset_dir = self.base_dir / "axl"
        self.output_dir = self.base_dir / "output"
        self.descriptors_csv = self.output_dir / "ventricle_descriptors.csv"
        self.experiment_history_path = self.output_dir / "training_experiments.json"
        
        # Ensure directories exist
        self.output_dir.mkdir(exist_ok=True)
        
        # Mock messagebox
        self._mock_messagebox()

    def _mock_messagebox(self):
        # We replace the imported messagebox in ml_training with a mock
        import tkinter.messagebox
        def showinfo(title, message): print(f"[INFO] {title}: {message}")
        def showwarning(title, message): print(f"[WARN] {title}: {message}")
        def showerror(title, message): print(f"[ERROR] {title}: {message}")
        def askyesno(title, message): return True
        
        # Patching the module where MLTrainingMixin is defined
        from brain_mri.ml import ml_training
        
        # Mocking Toplevel and Label since they are used for simple popups
        class MockToplevel:
            def __init__(self, master, **kwargs): pass
            def update(self): pass
            def destroy(self): pass

        class MockLabel:
            def __init__(self, master, text="", **kwargs): pass
            def pack(self, **kwargs): pass
        
        # We need to replace tk.Toplevel in the module namespace, 
        # but ml_training code calls tk.Toplevel directly from imported tkinter as tk
        # So we patch tk in sys.modules or monkeypatch attribute on the imported module if possible.
        # But ml_training does `import tkinter as tk`.
        # Easier to patch tk module itself in sys.modules if we imported it first?
        # Or patch ml_training.tk attribute.
        
        ml_training.tk.Toplevel = MockToplevel
        ml_training.tk.Label = MockLabel
        
        ml_training.messagebox.showinfo = showinfo
        ml_training.messagebox.showwarning = showwarning
        ml_training.messagebox.showerror = showerror
        ml_training.messagebox.askyesno = askyesno

    def _show_plot_window(self, title, figure):
        print(f"[PLOT] Saving plot '{title}' instead of showing.")
        figure.savefig(self.output_dir / f"{title.replace(' ', '_')}.png")

def create_dummy_data(base_path):
    base = Path(base_path)
    if base.exists():
        shutil.rmtree(base)
    base.mkdir()
    
    (base / "axl").mkdir()
    (base / "cor").mkdir()
    (base / "sag").mkdir()
    (base / "output").mkdir()

    # Subjects
    subjects = [f"sub{i:03d}" for i in range(10)]
    data = []

    print("Creating dummy images...")
    for sub in subjects:
        mri_id = f"{sub}_MR1"
        # Create images
        for orient in ["axl", "cor", "sag"]:
            # create 30x30 dummy image to be fast (resize in pipeline might fail if too small, but pipeline delegates to transforms)
            # Actually dataset.py removed resize, so we should rely on what pytorch transforms expect if any default
            # But let's make them 64x64
            img_arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            img = Image.fromarray(img_arr)
            # Filename format expected: {mri_id}_{orient}.png (or nii)
            # _list_orientation_paths checks {mri_id}_{orient}{ext}
            save_path = base / orient / f"{mri_id}_{orient}.png"
            img.save(save_path)
        
        # Clinical data
        data.append({
            "Subject Accession": sub,
            "MRI_ID": mri_id,
            "Group": np.random.choice(["Nondemented", "Demented"]),
            "Visit": 1,
            "MR Delay": 0,
            "M/F": np.random.choice(["M", "F"]),
            "Hand": "R",
            "Age": np.random.randint(60, 90),
            "EDUC": 15,
            "SES": 2,
            "MMSE": 28,
            "CDR": 0.0,
            "eTIV": 1500,
            "nWBV": 0.75,
            "ASF": 1.0,
            # ml_training.py expects lowercase names: 'age', 'education', 'nwbv', 'etiv', 'asf'
            "age": np.random.randint(60, 90),
            "education": 15,
            "nwbv": 0.75,
            "etiv": 1500,
            "asf": 1.0,
            "split": np.random.choice(["train", "validation", "test"], p=[0.6, 0.2, 0.2]),
            "Final_Group": np.random.choice(["Nondemented", "Demented"]), # Required by ml_training checking
        })
    
    # Force at least one in each split
    data[0]['split'] = 'train'
    data[1]['split'] = 'validation'
    data[2]['split'] = 'test'
    data[0]['Final_Group'] = 'Demented'
    data[1]['Final_Group'] = 'Nondemented'

    df = pd.DataFrame(data)
    csv_path = base / "oasis_longitudinal_demographic.csv"
    df.to_csv(csv_path, index=False)
    
    # Create dummy descriptors csv
    desc_data = []
    for sub in subjects:
        mri_id = f"{sub}_MR1"
        desc_data.append({
            "MRI_ID": mri_id,
            "area": 100.0, "perimeter": 50.0, "circularity": 0.8,
            "eccentricity": 0.5, "solidity": 0.9, "major_axis_length": 10.0, "minor_axis_length": 8.0
        })
    pd.DataFrame(desc_data).to_csv(base / "output" / "ventricle_descriptors.csv", index=False)

    return csv_path

def main():
    base_dir = Path("dummy_data_verification")
    print(f"Setting up dummy data in {base_dir}...")
    csv_path = create_dummy_data(base_dir)
    
    print("Initializing MockApp...")
    app = MockApp(base_dir)
    
    # Override dataset creation mixin behavior if needed, but MLTrainingMixin relies on self.descriptors_csv and csv_path
    # We need to monkeypatch self.csv_path (it's not in __init__ of mixin, but usually in App)
    app.csv_path = csv_path
    
    print("Starting training verification...")
    try:
        # We need to ensure USE_MULTIMODAL is considered if we want to test it
        # The logic in ml_training.py reads os.getenv("USE_MULTIMODAL", "1")
        # We want to test WITH clinical features
        os.environ["USE_MULTIMODAL"] = "1"
        os.environ["EPOCHS"] = "1" # Fast training
        os.environ["SPLIT_CSV_PATH"] = str(csv_path) # Force usage of our dummy csv
        
        # Run training
        # We mock defaults to be very light
        app._train_pytorch_model(
            mode='classification',
            backbone='medicalnet',
            # clinical_features argument removed as it's not accepted
        )
        print("\n\nSUCCESS: Training verification loop completed without errors.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n\nFAILURE: Training verification failed with error: {e}")
    finally:
        # Cleanup
        if base_dir.exists():
            shutil.rmtree(base_dir)

if __name__ == "__main__":
    main()
