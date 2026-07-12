"""
Phase classifiers on frozen encoder features — three heads, compared honestly.

Heads:
  1. 'lda'        — shrinkage LDA (solver='lsqr', shrinkage='auto'): regularised
                    for the small-sample / high-dim (p≈n) regime that made the
                    prior unregularised LDA hit train=100% / poor generalisation.
  2. 'pca_lda'    — PCA (fit on train only) → shrinkage LDA. Extra regularisation
                    by dimensionality reduction; also cheaper if reused as a loss.
  3. 'linear_nn'  — a trainable linear (or 1-hidden-layer) torch head with
                    cross-entropy + weight decay + early stopping on a validation
                    split. Directly comparable to the LDA variants.

Evaluation (identical protocol for all three):
  - Honest generalisation via GroupKFold over scan_id (a patient never spans two
    folds), reported as mean±std accuracy + macro-F1.
  - A final fit on all train+val features, evaluated once on the held-out TEST
    features (also patient-disjoint, guaranteed upstream by split_by_patient).

All feature scaling / PCA is fit on training data only inside each fold and for
the final test fit — no leakage of test statistics into preprocessing.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# sklearn heads (LDA, PCA→LDA) as leakage-safe pipelines
# ---------------------------------------------------------------------------

def _make_sklearn_head(kind: str, pca_components: int = 40) -> Pipeline:
    if kind == 'lda':
        steps = [
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ]
    elif kind == 'pca_lda':
        steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_components, random_state=0)),
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
        ]
    else:
        raise ValueError(kind)
    return Pipeline(steps)


# ---------------------------------------------------------------------------
# Linear NN head (torch) — trained with early stopping on an internal val split
# ---------------------------------------------------------------------------

class _LinearNNHead:
    """Linear or 1-hidden-layer torch classifier with a scikit-like API.

    Kept dependency-light (numpy in, numpy out); training uses a small internal
    validation split (grouped) for early stopping so it doesn't overfit like the
    unregularised LDA did.
    """

    def __init__(self, n_classes: int, hidden: int = 0, epochs: int = 200,
                 lr: float = 1e-3, weight_decay: float = 1e-3, patience: int = 20,
                 seed: int = 0):
        self.n_classes = n_classes
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.seed = seed
        self.scaler = StandardScaler()
        self.model = None

    def _build(self, in_dim: int):
        import torch.nn as nn
        if self.hidden > 0:
            return nn.Sequential(
                nn.Linear(in_dim, self.hidden), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(self.hidden, self.n_classes),
            )
        return nn.Linear(in_dim, self.n_classes)

    def fit(self, X, y, groups=None):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        X = self.scaler.fit_transform(X).astype(np.float32)
        y = np.asarray(y)

        # Grouped internal train/val split for early stopping (fall back to a
        # plain split if groups are unavailable).
        rng = np.random.default_rng(self.seed)
        if groups is not None:
            uniq = np.array(sorted(set(groups)))
            rng.shuffle(uniq)
            n_val = max(1, int(0.2 * len(uniq)))
            val_g = set(uniq[:n_val].tolist())
            val_mask = np.array([g in val_g for g in groups])
        else:
            val_mask = rng.random(len(y)) < 0.2
        if val_mask.all() or (~val_mask).all():   # degenerate tiny inputs
            val_mask = np.zeros(len(y), bool); val_mask[: max(1, len(y) // 5)] = True

        Xtr, ytr = torch.tensor(X[~val_mask]), torch.tensor(y[~val_mask])
        Xva, yva = torch.tensor(X[val_mask]),  torch.tensor(y[val_mask])

        self.model = self._build(X.shape[1])
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        best_val, best_state, bad = float('inf'), None, 0
        for _ in range(self.epochs):
            self.model.train()
            opt.zero_grad()
            loss = loss_fn(self.model(Xtr), ytr)
            loss.backward(); opt.step()

            self.model.eval()
            with torch.no_grad():
                v = loss_fn(self.model(Xva), yva).item()
            if v < best_val - 1e-4:
                best_val, best_state, bad = v, {k: t.clone() for k, t in self.model.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self

    def predict(self, X):
        import torch
        Xs = self.scaler.transform(X).astype(np.float32)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.tensor(Xs))
            return logits.argmax(dim=1).numpy()


# ---------------------------------------------------------------------------
# Unified evaluation
# ---------------------------------------------------------------------------

@dataclass
class HeadResult:
    name: str
    cv_acc_mean: float
    cv_acc_std: float
    cv_f1_mean: float
    test_acc: float
    test_f1: float
    test_confusion: List[List[int]] = field(default_factory=list)
    test_report: dict = field(default_factory=dict)


def _fit_predict(kind: str, n_classes: int, Xtr, ytr, gtr, Xte,
                 pca_components: int) -> np.ndarray:
    if kind in ('lda', 'pca_lda'):
        head = _make_sklearn_head(kind, pca_components)
        head.fit(Xtr, ytr)
        return head.predict(Xte)
    head = _LinearNNHead(n_classes=n_classes)
    head.fit(Xtr, ytr, groups=gtr)
    return head.predict(Xte)


def evaluate_head(
    kind: str,
    X_trainval: np.ndarray, y_trainval: np.ndarray, groups_trainval: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
    n_splits: int = 5,
    pca_components: int = 40,
) -> HeadResult:
    """GroupKFold CV on train+val, plus a single held-out test fit/eval."""
    n_classes = int(max(y_trainval.max(), y_test.max())) + 1
    n_groups = len(set(groups_trainval))
    n_splits = min(n_splits, n_groups)

    # --- patient-grouped CV ---
    accs, f1s = [], []
    gkf = GroupKFold(n_splits=n_splits)
    for tr, va in gkf.split(X_trainval, y_trainval, groups_trainval):
        pred = _fit_predict(kind, n_classes, X_trainval[tr], y_trainval[tr],
                            groups_trainval[tr], X_trainval[va], pca_components)
        accs.append(accuracy_score(y_trainval[va], pred))
        f1s.append(f1_score(y_trainval[va], pred, average='macro', zero_division=0))

    # --- final fit on all train+val, eval once on held-out test ---
    test_pred = _fit_predict(kind, n_classes, X_trainval, y_trainval,
                             groups_trainval, X_test, pca_components)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)
    labels = sorted(set(y_test.tolist()) | set(test_pred.tolist()))
    names = [class_names[i] for i in labels] if class_names else [str(i) for i in labels]
    report = classification_report(y_test, test_pred, labels=labels,
                                   target_names=names, zero_division=0, output_dict=True)
    conf = confusion_matrix(y_test, test_pred, labels=labels).tolist()

    res = HeadResult(
        name=kind,
        cv_acc_mean=float(np.mean(accs)), cv_acc_std=float(np.std(accs)),
        cv_f1_mean=float(np.mean(f1s)),
        test_acc=float(test_acc), test_f1=float(test_f1),
        test_confusion=conf, test_report=report,
    )
    log.info(f"[{kind}] CV acc={res.cv_acc_mean:.4f}±{res.cv_acc_std:.4f} "
             f"CV macroF1={res.cv_f1_mean:.4f} | test acc={res.test_acc:.4f} "
             f"test macroF1={res.test_f1:.4f}")
    return res


def compare_heads(
    X_trainval, y_trainval, groups_trainval, X_test, y_test,
    class_names: Optional[List[str]] = None,
    kinds: List[str] = ('lda', 'pca_lda', 'linear_nn'),
    n_splits: int = 5,
    pca_components: int = 40,
) -> Dict[str, HeadResult]:
    """Run every head under the identical protocol; return {kind: HeadResult}."""
    return {
        k: evaluate_head(k, X_trainval, y_trainval, groups_trainval, X_test, y_test,
                         class_names=class_names, n_splits=n_splits,
                         pca_components=pca_components)
        for k in kinds
    }


__all__ = ['HeadResult', 'evaluate_head', 'compare_heads']
