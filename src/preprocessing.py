"""
Per-fold, leakage-free preprocessing for the classical branch.

Everything is *fit on the training split only* and then applied to validation and
test:
  1. clean   - replace inf, impute NaN with the train column mean, drop columns
               that are constant on train,
  2. scale   - z-score (StandardScaler),
  3. select  - univariate SelectKBest(f_classif) capped at SELECT_K features,
  4. pca     - PCA representation for the SVM-RBF model.

Two representations are returned: ``*_sel`` (scaled + selected; used by Random
Forest and Linear-SVM) and ``*_pca`` (used by SVM-RBF).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler

from . import config as C


@dataclass
class Prep:
    Xtr_sel: np.ndarray
    Xval_sel: np.ndarray
    Xte_sel: np.ndarray
    Xtr_pca: np.ndarray
    Xval_pca: np.ndarray
    Xte_pca: np.ndarray
    n_selected: int
    n_pca: int

    def get(self, representation: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if representation == "sel":
            return self.Xtr_sel, self.Xval_sel, self.Xte_sel
        if representation == "pca":
            return self.Xtr_pca, self.Xval_pca, self.Xte_pca
        raise ValueError(representation)


def _clean_fit(Xtr: np.ndarray):
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan)
    col_mean = np.nanmean(Xtr, axis=0)
    col_mean = np.nan_to_num(col_mean, nan=0.0)
    filled = np.where(np.isnan(Xtr), col_mean, Xtr)
    keep = filled.std(axis=0) > 0                      # drop constant columns
    if not keep.any():                                 # degenerate guard
        keep = np.ones(filled.shape[1], dtype=bool)
    return col_mean, keep


def _clean_apply(X: np.ndarray, col_mean: np.ndarray, keep: np.ndarray) -> np.ndarray:
    X = np.where(np.isfinite(X), X, np.nan)
    X = np.where(np.isnan(X), col_mean, X)
    return X[:, keep]


def prepare_classical_fold(Xtr: np.ndarray, ytr: np.ndarray,
                           Xval: np.ndarray, Xte: np.ndarray) -> Prep:
    # 1. clean (fit on train)
    col_mean, keep = _clean_fit(Xtr)
    Xtr_c = _clean_apply(Xtr, col_mean, keep)
    Xval_c = _clean_apply(Xval, col_mean, keep)
    Xte_c = _clean_apply(Xte, col_mean, keep)

    # 2. scale (fit on train)
    scaler = StandardScaler().fit(Xtr_c)
    Xtr_s = scaler.transform(Xtr_c)
    Xval_s = scaler.transform(Xval_c)
    Xte_s = scaler.transform(Xte_c)

    # 3. univariate selection (fit on train + train labels)
    k = min(C.SELECT_K, Xtr_s.shape[1])
    selector = SelectKBest(f_classif, k=k).fit(Xtr_s, ytr)
    Xtr_sel = selector.transform(Xtr_s)
    Xval_sel = selector.transform(Xval_s)
    Xte_sel = selector.transform(Xte_s)

    # 4. PCA representation for SVM-RBF (fit on selected train features)
    n_comp = min(C.PCA_MAX, Xtr_sel.shape[1], max(Xtr_sel.shape[0] - 1, 1))
    pca = PCA(n_components=n_comp, random_state=C.SEED).fit(Xtr_sel)
    Xtr_pca = pca.transform(Xtr_sel)
    Xval_pca = pca.transform(Xval_sel)
    Xte_pca = pca.transform(Xte_sel)

    return Prep(Xtr_sel, Xval_sel, Xte_sel, Xtr_pca, Xval_pca, Xte_pca,
                n_selected=Xtr_sel.shape[1], n_pca=Xtr_pca.shape[1])
