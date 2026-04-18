"""Tests for brain_mri/ui/feature_selection.py and brain_mri/ui/__init__.py (new in this PR)."""

from __future__ import annotations

import sys
import types

import pytest


# ---------------------------------------------------------------------------
# Install a stub tkinter module that is used for headless CI
# ---------------------------------------------------------------------------

def _install_tkinter_stub() -> None:
    if "tkinter" in sys.modules:
        return
    tk = types.ModuleType("tkinter")
    tk.Toplevel = None
    tk.BooleanVar = None
    tk.Checkbutton = None
    tk.Button = None
    messagebox = types.ModuleType("tkinter.messagebox")
    messagebox.showinfo = lambda *args, **kwargs: None
    messagebox.showwarning = lambda *args, **kwargs: None
    messagebox.showerror = lambda *args, **kwargs: None
    messagebox.askyesno = lambda *args, **kwargs: True
    tk.messagebox = messagebox
    sys.modules["tkinter"] = tk
    sys.modules["tkinter.messagebox"] = messagebox


_install_tkinter_stub()


# ---------------------------------------------------------------------------
# Tests for brain_mri/ui/__init__.py
# ---------------------------------------------------------------------------


def test_ui_init_exports_feature_selection_mixin():
    """brain_mri.ui must export FeatureSelectionMixin."""
    from brain_mri.ui import FeatureSelectionMixin

    assert FeatureSelectionMixin is not None


def test_ui_all_contains_feature_selection_mixin():
    """__all__ in brain_mri.ui must list FeatureSelectionMixin."""
    import brain_mri.ui as ui_module

    assert "FeatureSelectionMixin" in ui_module.__all__


def test_ui_feature_selection_mixin_importable_from_module():
    """FeatureSelectionMixin imported from brain_mri.ui.feature_selection must be the same object."""
    from brain_mri.ui import FeatureSelectionMixin
    from brain_mri.ui.feature_selection import FeatureSelectionMixin as Direct

    assert FeatureSelectionMixin is Direct


# ---------------------------------------------------------------------------
# Tests for FeatureSelectionMixin behaviour
# ---------------------------------------------------------------------------


class _MinimalApp:
    """Minimal stub that mixes in FeatureSelectionMixin without a real Tk root."""

    root = None  # Simulate headless / no Tk root

    def train_svm_classifier(self, features):
        self._last_svm_features = features

    def train_xgboost_regressor(self, features):
        self._last_xgb_features = features


def test_feature_selection_mixin_generic_selector_raises_without_tk(monkeypatch):
    """_generic_feature_selector must raise RuntimeError when tkinter is unavailable."""
    from brain_mri.ui import feature_selection as fs_module
    from brain_mri.ui.feature_selection import FeatureSelectionMixin

    # Simulate tk being None (unavailable)
    monkeypatch.setattr(fs_module, "tk", None)

    class _App(FeatureSelectionMixin):
        root = None

        def train_svm_classifier(self, features):
            pass

    app = _App()

    with pytest.raises(RuntimeError, match=r"[Tt]kinter"):
        app._generic_feature_selector("SVM", app.train_svm_classifier)


def test_feature_selection_mixin_generic_selector_raises_without_root(monkeypatch):
    """_generic_feature_selector must raise RuntimeError when app.root is None."""
    import brain_mri.ui.feature_selection as fs_module

    # Restore a non-None tk stub so the first guard passes
    fake_tk = types.ModuleType("tkinter")
    monkeypatch.setattr(fs_module, "tk", fake_tk)

    from brain_mri.ui.feature_selection import FeatureSelectionMixin

    class _App(FeatureSelectionMixin):
        root = None

        def train_svm_classifier(self, features):
            pass

    app = _App()
    with pytest.raises(RuntimeError, match="root"):
        app._generic_feature_selector("SVM", app.train_svm_classifier)


def test_open_feature_selection_dialog_delegates_to_svm(monkeypatch):
    """open_feature_selection_dialog must call _generic_feature_selector with 'SVM'."""
    from brain_mri.ui.feature_selection import FeatureSelectionMixin

    calls = []

    class _App(FeatureSelectionMixin):
        root = None

        def train_svm_classifier(self, features):
            pass

        def _generic_feature_selector(self, title, callback):
            calls.append((title, callback))

    app = _App()
    app.open_feature_selection_dialog()

    assert len(calls) == 1
    assert calls[0][0] == "SVM"
    assert calls[0][1] == app.train_svm_classifier


def test_open_feature_selection_dialog_xgboost_delegates(monkeypatch):
    """open_feature_selection_dialog_xgboost must call _generic_feature_selector with 'XGBoost'."""
    from brain_mri.ui.feature_selection import FeatureSelectionMixin

    calls = []

    class _App(FeatureSelectionMixin):
        root = None

        def train_xgboost_regressor(self, features):
            pass

        def _generic_feature_selector(self, title, callback):
            calls.append((title, callback))

    app = _App()
    app.open_feature_selection_dialog_xgboost()

    assert len(calls) == 1
    assert calls[0][0] == "XGBoost"
    assert calls[0][1] == app.train_xgboost_regressor


def test_feature_selection_mixin_feature_list_contains_expected_items():
    """UI feature defaults must match the canonical trainer defaults."""
    from brain_mri.ml.classical_training import DEFAULT_SVM_FEATURES, DEFAULT_XGB_FEATURES
    from brain_mri.ui.feature_selection import SVM_FEATURES, XGBOOST_FEATURES

    assert SVM_FEATURES == DEFAULT_SVM_FEATURES
    assert XGBOOST_FEATURES == DEFAULT_XGB_FEATURES
