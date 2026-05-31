"""
ICT Trade Outcome Predictor — Institutional Edition
=====================================================
LightGBM binary classifier: P(TP1 hit before SL) given ICT + quant features.

Upgrades over v1 (XGBoost):
  • LightGBM   — faster, handles small datasets better, native NaN support
  • Optuna     — 50-trial hyperparameter search on TimeSeriesSplit CV
  • Calibration — Platt sigmoid scaling → probabilities actually mean P(win)
  • Multi-symbol — trains on all top-10 symbols, trades only configured subset
"""
from __future__ import annotations
import os
import pickle

import numpy as np
import pandas as pd

from ml.feature_builder import FEATURE_COLS

MODEL_PATH = "models/ict_predictor.pkl"


class ICTTradePredictor:

    def __init__(self):
        self.model       = None   # calibrated pipeline
        self._base_model = None   # raw LightGBM
        self.cols        = FEATURE_COLS
        self._fitted     = False

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df: pd.DataFrame, test_size: float = 0.20,
              n_optuna_trials: int = 50) -> dict:
        """
        Time-based 80/20 split. Optuna optimises on the 80% train with
        5-fold TimeSeriesSplit. Final model is calibrated on a held-out
        20% slice.
        """
        from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        X = df[self.cols].fillna(0).astype(float)
        y = df["label"].astype(int)

        n       = len(df)
        n_train = int(n * (1 - test_size))
        X_tr, X_te = X.iloc[:n_train], X.iloc[n_train:]
        y_tr, y_te = y.iloc[:n_train], y.iloc[n_train:]

        print(f"\n  Train: {len(X_tr):4d} samples  WR={y_tr.mean():.1%}")
        print(f"  Test : {len(X_te):4d} samples  WR={y_te.mean():.1%}")

        # ── Optuna hyperparameter search ──────────────────────────────────────
        print(f"\n  Running Optuna ({n_optuna_trials} trials × 5-fold TimeSeriesCV)...")
        best_params = self._optuna_search(X_tr, y_tr, n_optuna_trials)
        print(f"  Best params: {best_params}")

        # ── Train final model on full train set ───────────────────────────────
        pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
        base = self._build_lgbm(best_params, pos_weight)
        base.fit(X_tr, y_tr)
        self._base_model = base

        # ── Platt calibration on held-out test set ────────────────────────────
        calibrated = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        calibrated.fit(X_te, y_te)
        self.model   = calibrated
        self._fitted = True

        # ── Evaluation ────────────────────────────────────────────────────────
        proba_te = calibrated.predict_proba(X_te)[:, 1]
        pred_te  = (proba_te >= 0.5).astype(int)

        metrics = {
            "auc":      roc_auc_score(y_te, proba_te) if len(y_te.unique()) > 1 else 0.5,
            "accuracy": accuracy_score(y_te, pred_te),
            "n_train":  n_train,
            "n_test":   len(X_te),
            "train_wr": float(y_tr.mean()),
            "test_wr":  float(y_te.mean()),
        }
        print(f"\n  AUC (calibrated) : {metrics['auc']:.3f}")
        print(f"  Accuracy         : {metrics['accuracy']:.1%}")
        print(f"\n{classification_report(y_te, pred_te, target_names=['Loss','Win'], zero_division=0)}")

        # Probability distribution check
        print(f"  Probability range: [{proba_te.min():.3f}, {proba_te.max():.3f}]"
              f"  median={np.median(proba_te):.3f}")

        return metrics

    # ── Hyperparameter search ─────────────────────────────────────────────────

    @staticmethod
    def _optuna_search(X_tr: pd.DataFrame, y_tr: pd.Series,
                       n_trials: int) -> dict:
        try:
            import optuna
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        except ImportError:
            print("  Optuna not installed — using default params")
            return {}

        from sklearn.model_selection import TimeSeriesSplit, cross_val_score

        def objective(trial):
            params = {
                "n_estimators":      trial.suggest_int("n_estimators",    100, 600),
                "max_depth":         trial.suggest_int("max_depth",        2,   6),
                "learning_rate":     trial.suggest_float("lr",            0.01, 0.15, log=True),
                "subsample":         trial.suggest_float("subsample",     0.6,  1.0),
                "colsample_bytree":  trial.suggest_float("colsample",     0.5,  1.0),
                "min_child_samples": trial.suggest_int("min_child",        5,  40),
                "reg_lambda":        trial.suggest_float("lambda",        2.0, 20.0, log=True),
                "reg_alpha":         trial.suggest_float("alpha",         0.0,  5.0),
            }
            mdl = ICTTradePredictor._build_lgbm(params, scale_pos_weight=1.0)
            tscv = TimeSeriesSplit(n_splits=5)
            try:
                scores = cross_val_score(mdl, X_tr, y_tr, cv=tscv,
                                         scoring="roc_auc", error_score=0.5)
                mean_score = float(np.nanmean(scores))
                return mean_score if np.isfinite(mean_score) else 0.5
            except Exception:
                return 0.5

        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        try:
            return study.best_params
        except ValueError:
            print("  Optuna: all trials failed — using default params")
            return {}

    @staticmethod
    def _build_lgbm(params: dict, scale_pos_weight: float = 1.0):
        try:
            from lightgbm import LGBMClassifier
            p = dict(params)  # copy
            # Remap trial param names to LightGBM param names
            if "lr" in p:       p["learning_rate"]    = p.pop("lr")
            if "colsample" in p: p["colsample_bytree"] = p.pop("colsample")
            if "min_child" in p: p["min_child_samples"] = p.pop("min_child")
            if "lambda" in p:   p["reg_lambda"]        = p.pop("lambda")
            if "alpha" in p:    p["reg_alpha"]          = p.pop("alpha")
            return LGBMClassifier(
                **p,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
        except ImportError:
            pass

        # Fallback: XGBoost
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=42, eval_metric="logloss",
                use_label_encoder=False,
            )
        except Exception:
            pass

        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=150, max_depth=3, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )

    # ── Feature importances ───────────────────────────────────────────────────

    def feature_importances(self) -> pd.Series:
        mdl = self._base_model
        if mdl is None:
            return pd.Series(dtype=float)
        if hasattr(mdl, "feature_importances_"):
            imp = pd.Series(mdl.feature_importances_, index=self.cols)
            return (imp / imp.sum()).sort_values(ascending=False)
        return pd.Series(dtype=float)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba_single(self, features: dict) -> float:
        """Returns P(win) ∈ [0, 1]; 0.5 if model not loaded."""
        if self.model is None:
            return 0.5
        row = pd.DataFrame([{col: features.get(col, 0) for col in self.cols}])
        return float(self.model.predict_proba(row.fillna(0))[:, 1][0])

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model":       self.model,
                "base_model":  self._base_model,
                "cols":        self.cols,
            }, f)
        print(f"  Model saved → {path}")

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.model       = obj.get("model")
        self._base_model = obj.get("base_model")
        self.cols        = obj.get("cols", FEATURE_COLS)
        self._fitted     = self.model is not None
        return self._fitted
