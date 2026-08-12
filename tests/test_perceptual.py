"""
perceptual.py's FID arithmetic and slice selection.

WHY THIS EXISTS. FID and LPIPS were added for comparability with the published
NCCT→CECT literature, and both are easy to get quietly, unfalsifiably wrong:

  1. The Fréchet formula. `sqrtm` on a near-singular 2048x2048 covariance (which
     is what a few thousand correlated CT slices produce) returns a complex
     matrix. Dropping the imaginary part unconditionally hides a genuine
     numerical failure; the closed-form cases below pin the arithmetic so a
     regression there cannot pass as "FID is just a big number".

  2. Slice selection. A CT volume's first and last slices are almost pure
     scanner air. Those frames are byte-identical in the generated and real
     volumes, so including them pulls every model's FID toward 0 by the same
     amount and destroys the column's ability to separate models. The selection
     must also be derived from the REAL volume only, or two models would be
     scored on different slice sets and their FIDs would not be comparable.

These run without torch, lpips or pytorch-fid: `frechet_distance` is
numpy/scipy-only and `_select` is exercised unbound, so the test suite does not
require the optional --perceptual dependency stack.

Run directly:
    python tests/test_perceptual.py
"""
import sys
import pathlib
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from perceptual import PerceptualScorer, frechet_distance


# ---------------------------------------------------------------------------
# Fréchet distance — closed-form cases
# ---------------------------------------------------------------------------

def test_fid_zero_for_identical_gaussians():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 32))
    mu, S = X.mean(0), np.cov(X, rowvar=False)
    assert abs(frechet_distance(mu, S, mu, S)) < 1e-6


def test_fid_pure_mean_shift_is_squared_distance():
    """Same covariance, means d apart -> FID == ||d||^2 exactly."""
    n = 32
    S = np.eye(n)
    d = np.zeros(n)
    d[0], d[1] = 3.0, 4.0                      # ||d|| = 5
    assert abs(frechet_distance(np.zeros(n), S, d, S) - 25.0) < 1e-6


def test_fid_isotropic_scale():
    """N(0,I) vs N(0,s^2 I) -> FID == n*(1-s)^2."""
    n, s = 32, 2.0
    I = np.eye(n)
    got = frechet_distance(np.zeros(n), I, np.zeros(n), s ** 2 * I)
    assert abs(got - n * (1 - s) ** 2) < 1e-6


def test_fid_is_symmetric():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(500, 24))
    B = rng.normal(size=(500, 24)) * 1.4 + 0.7
    ab = frechet_distance(A.mean(0), np.cov(A, rowvar=False),
                          B.mean(0), np.cov(B, rowvar=False))
    ba = frechet_distance(B.mean(0), np.cov(B, rowvar=False),
                          A.mean(0), np.cov(A, rowvar=False))
    assert abs(ab - ba) < 1e-8
    assert ab > 0


def test_fid_finite_on_rank_deficient_covariance():
    """Fewer samples than feature dims — the real case at n=20 volumes.

    np.cov is then singular and sqrtm goes complex; the eps-conditioned retry
    must still return a finite real number rather than nan.
    """
    rng = np.random.default_rng(2)
    A = rng.normal(size=(40, 128))             # 40 samples, 128 dims
    B = rng.normal(size=(40, 128)) + 0.3
    got = frechet_distance(A.mean(0), np.cov(A, rowvar=False),
                           B.mean(0), np.cov(B, rowvar=False))
    assert np.isfinite(got), 'FID went non-finite on a rank-deficient covariance'


# ---------------------------------------------------------------------------
# Slice selection — no torch needed, _select touches only numpy attributes
# ---------------------------------------------------------------------------

def _fake(min_body_frac=0.02, max_slices=None, slice_axis=-1):
    return SimpleNamespace(min_body_frac=min_body_frac,
                           max_slices_per_case=max_slices,
                           slice_axis=slice_axis)


def _mask(n_air_lead, n_body, n_air_tail, body_frac=0.4, hw=20):
    """(hw, hw, D) body mask: air slices at both ends, body in the middle."""
    D = n_air_lead + n_body + n_air_tail
    m = np.zeros((hw, hw, D), dtype=np.float32)
    rows = max(1, int(round(body_frac * hw)))
    m[:rows, :, n_air_lead:n_air_lead + n_body] = 1.0
    return m


def test_select_drops_air_slices():
    """The whole point: near-empty frames are identical in gen and real."""
    m = _mask(5, 10, 7)
    idx = PerceptualScorer._select(_fake(), m)
    assert idx.tolist() == list(range(5, 15))


def test_select_threshold_is_body_fraction_of_the_frame():
    # body covers 10% of each in-plane frame; a 0.2 threshold must reject it,
    # a 0.05 threshold must keep it.
    m = _mask(2, 6, 2, body_frac=0.1)
    assert PerceptualScorer._select(_fake(min_body_frac=0.20), m).size == 10   # fallback
    assert PerceptualScorer._select(_fake(min_body_frac=0.05), m).tolist() == \
        list(range(2, 8))


def test_select_falls_back_to_all_slices_when_mask_is_empty():
    """A degenerate body mask must not silently produce a zero-slice FID set."""
    m = np.zeros((16, 16, 9), dtype=np.float32)
    assert PerceptualScorer._select(_fake(), m).tolist() == list(range(9))


def test_select_subsamples_evenly_and_keeps_the_endpoints():
    m = _mask(0, 40, 0)
    idx = PerceptualScorer._select(_fake(max_slices=5), m)
    assert idx.size == 5
    assert idx[0] == 0 and idx[-1] == 39
    assert list(idx) == sorted(set(idx)), 'subsample produced duplicate/unordered slices'


def test_select_is_model_independent():
    """Derived from the real volume's mask alone, so every model gets the same
    slice set and their FIDs are comparable."""
    m = _mask(3, 12, 4)
    a = PerceptualScorer._select(_fake(), m)
    b = PerceptualScorer._select(_fake(), m.copy())
    assert a.tolist() == b.tolist()


def test_select_honours_slice_axis():
    """metrics.py's convention is slice_axis=-1; a first-axis volume must work too."""
    m = np.moveaxis(_mask(4, 8, 3), -1, 0)
    idx = PerceptualScorer._select(_fake(slice_axis=0), m)
    assert idx.tolist() == list(range(4, 12))


if __name__ == '__main__':
    fns = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f'  PASS  {fn.__name__}')
        except AssertionError as e:
            failed += 1
            print(f'  FAIL  {fn.__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
