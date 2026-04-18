try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None
    messagebox = None

from brain_mri.ml.classical_training import DEFAULT_SVM_FEATURES, DEFAULT_XGB_FEATURES

SVM_FEATURES = DEFAULT_SVM_FEATURES
XGBOOST_FEATURES = DEFAULT_XGB_FEATURES


class FeatureSelectionMixin:
    def open_feature_selection_dialog(self):
        """
        Open a feature selection dialog configured for SVM and invoke the instance's SVM training callback with the selected feature names.
        """
        self._generic_feature_selector("SVM", self.train_svm_classifier)

    def open_feature_selection_dialog_xgboost(self):
        """
        Open a feature-selection dialog configured for XGBoost and start XGBoost training with the chosen features.
        
        When the user confirms a non-empty selection, the dialog is closed and `self.train_xgboost_regressor(selected_features)` is called with the list of selected feature names.
        """
        self._generic_feature_selector("XGBoost", self.train_xgboost_regressor)

    def _generic_feature_selector(self, title, callback):
        """
        Open a modal-like Tkinter window that lets the user select which features to use and then calls the provided callback with the chosen feature names.
        
        Raises:
            RuntimeError: If Tkinter is unavailable or if `self.root` is missing or None.
        
        Parameters:
            title (str): Short descriptor used in the dialog title (e.g., "SVM" or "XGBoost").
            callback (Callable[[list[str]], Any]): Function to invoke with the list of selected feature names when the user confirms.
        
        Behavior:
            - Renders checkboxes for a fixed set of features (all checked by default).
            - If the user confirms with no features selected, an error dialog is shown and the selector remains open.
            - If at least one feature is selected, the dialog is closed and `callback(selected_features)` is called.
        """
        if tk is None:
            raise RuntimeError("Tkinter is not available in this environment.")
        if not hasattr(self, "root") or self.root is None:
            raise RuntimeError("Feature selection dialog requires an active Tk root.")

        win = tk.Toplevel(self.root)
        win.title(f"Features para {title}")

        vars_dict = {}
        features = XGBOOST_FEATURES if title == "XGBoost" else SVM_FEATURES

        for feature in features:
            variable = tk.BooleanVar(value=True)
            tk.Checkbutton(win, text=feature, variable=variable).pack(anchor="w")
            vars_dict[feature] = variable

        def run():
            """
            Collect the checked feature names from the dialog and either invoke the callback with them or show an error if none are selected.
            
            If no features are selected, displays an error dialog titled "Seleção obrigatória" with message "Selecione pelo menos uma feature para treinar." and leaves the selection window open. If one or more features are selected, closes the selection window and calls the surrounding scope's `callback` with a list of the selected feature names.
            """
            selected = [name for name, variable in vars_dict.items() if variable.get()]
            if not selected:
                messagebox.showerror("Seleção obrigatória", "Selecione pelo menos uma feature para treinar.")
                return
            win.destroy()
            callback(selected)

        tk.Button(win, text="Treinar", command=run).pack(pady=10)
