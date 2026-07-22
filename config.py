"""
Configuration for the literature-baseline NCCT → CECT experiment.

Data pipeline follows autoenc_fresh:
  • Files found via file_tag glob (no external module import)
  • HU clip → [0, 1] normalisation
  • Validity filter in HU space (min_patch_std / min_patch_mean / min_patch_max)
  • Random sub-sample then full RAM preload of patches
  • __getitem__ = plain list lookup (zero disk I/O during training)

2-D vs 3-D switch:
  patch_depth = 1   → 2-D slices, dims = 2   (default, same as autoenc_fresh)
  patch_depth = 8   → 3-D patches, dims = 3  (future extension)
  Set both patch_depth and dims together.

BASELINE LOSSES (most common in NCCT→CECT literature):
  L1 (λ=100, auto-reduced to 25 whenever adversarial/perceptual/feature_matching
  is active — see LAMBDA_L1_REDUCED below) + Adversarial LSGAN (λ=2) +
  Perceptual VGG16 (λ=10) + Feature matching (λ=10)
  References: Liu22, Hau21, Cho21, Yan24c, Yan22, Yan23

All extra losses are off by default — toggle via use_* flags.
"""

import json
from pathlib import Path
import torch

# Repo root — used to absolute-ise paths that must resolve regardless of cwd.
_HERE = Path(__file__).resolve().parent

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = '../sample_data_reg/ncct_cect/vindr_ds/all_baseline_algorithms/B2_deeds__aligned'
LABELS_CSV = '../sample_data_reg/ncct_cect/vindr_ds/labels.csv'
OUTPUT_DIR = Path('../out_synthesis_train/literature_baseline')
# Fixed, independent of OUTPUT_DIR: preloaded patches are cached here keyed by
# data/geometry config (not loss flags), so scenario runs that only toggle
# loss flags reuse the same cache instead of re-preloading from scratch.
CACHE_DIR  = Path('../out_synthesis_train/patch_cache')

# ── Data / glob ───────────────────────────────────────────────────────────────
FILE_TAG     = '_deeds'           # suffix before .nii.gz (same as autoenc_fresh glob)
# Organ mask suffix: '..._deeds{SEG_SUFFIX}.nii.gz'. '_seg_full' = the
# regenerated full TotalSegmentator masks (CTPhase-XGBoost/run_ts_masks.sh),
# which include the aorta/heart/IVC (confirmed via retrain_out_full: OOF
# accuracy 87.8%->97.07%, aorta coverage 0/410->410/410 vs '_seg_reg').
SEG_SUFFIX   = '_seg_full'
TARGET_PHASE = 'venous'           # 'arterial' | 'venous' | 'portal' | 'delayed'

# ── HU normalisation ──────────────────────────────────────────────────────────
# Data-driven via analyze_hu_range.py, but NOT its raw p0.5-p99.5 output
# ([-900, 690]) — that pooled percentile is dominated by a huge low-density
# mass (lung bases / fat, sitting right above the script's -900 "air floor",
# see p0.1=-899 .. p5=-881 in the histogram log) that carries essentially no
# NCCT->CECT contrast-enhancement signal (contrast agent doesn't accumulate
# in lung/fat). Adopting it as-is would spend most of the normalised [0,1]
# range on that non-informative tail and compress the actually-informative
# soft-tissue/vessel enhancement band (p25=-113 .. p99=525, i.e. roughly
# where liver/kidney/vessel HU values and their contrast enhancement live)
# into a much smaller slice of the network's effective dynamic range.
# HU_MIN=-200 already excludes most of that lung/fat tail (only ~25% of the
# pooled "tissue" mass falls below it, consistent with p25=-113). HU_MAX is
# raised from the original 300 to 400 to keep more of the arterial/venous
# vessel enhancement peak (p95=175 .. p99=525) without pulling in the
# calcification/bone range that starts around p99.5=691 / p99.9=1049.
HU_MIN = -200                     # clip lower bound
HU_MAX =  400                     # clip upper bound  → rescale to [0, 1]

# ── Patch extraction ─────────────────────────────────────────────────────────
PATCH_SIZE   = 128                # int → square, or (H, W) tuple
PATCH_DEPTH  = 1                  # 1 = 2-D slice  |  >1 = 3-D sub-volume
OVERLAP      = 0.5                # stride = patch_size × (1 - overlap)
DIMS         = 2                  # must match: PATCH_DEPTH==1 → 2, else 3

# ── Validity filter (applied in HU space on source/NCCT patch) ───────────────
MIN_PATCH_STD  =  10.0            # reject near-constant patches (air, padding)
MIN_PATCH_MEAN = -800.0           # reject pure-air patches
MIN_PATCH_MAX  = -500.0           # must have at least some tissue

# ── Organ-focused sampling ────────────────────────────────────────────────────
# Fraction of patches whose CENTRE sits on an organ/vessel voxel (rest stay
# uniform-grid). 0.0 = legacy behaviour. Raise to ~0.5 when using organ or
# phase-consistency losses so patches actually contain the aorta/portal vein/IVC
# those losses depend on. ORGAN_FOCUS_LABELS restricts the focus to specific
# TotalSegmentator label ids (needs MULTILABEL masks); None = any mask>0 voxel.
ORGAN_FOCUS_FRAC   = 0.0
ORGAN_FOCUS_LABELS = None         # e.g. aorta/IVC/portal-vein ids for phase work
MAX_FOCUS_CAND_PER_VOL = 3000

# ── RAM preload budget ────────────────────────────────────────────────────────
MAX_TRAIN_PATCHES = 20_000
MAX_VAL_PATCHES   =  4_000

# ── Train/val/test split ─────────────────────────────────────────────────────
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42

# ── Training schedule ────────────────────────────────────────────────────────
BATCH_SIZE   = 16
# Measured on the l1_only / bowel_zero / organ_curriculum runs: organ-region SSIM
# plateaus by epoch ~13 and the remaining ~55 epochs change nothing (train L1 flat
# at 0.0137 raw from epoch 13 to 69). 45 leaves margin for the decay curriculum to
# finish at epoch 30 and anneal afterwards, without paying for dead epochs.
EPOCHS       = 45
LR_GEN       = 2e-4
LR_DISC      = 1e-4
BETAS        = (0.5, 0.999)
WEIGHT_DECAY = 1e-5

# Cosine annealing with WARM RESTARTS was measurably harmful: with T0=15/TMULT=2
# the restarts land at epochs 15 and 46, and each one costs performance it then
# spends ~15 epochs recovering (org-SSIM 0.93644 @ep13 → 0.93440 @ep17; a second
# dip after the ep46 restart). Every best value occurred at the LOW-LR end of a
# cycle, never after a restart — the restarts never found a better optimum.
# Setting T0 >= EPOCHS gives a single monotone anneal, i.e. no restart.
USE_COSINE    = True
COSINE_T0     = 45
COSINE_TMULT  = 2
COSINE_ETA    = 5e-7

DISC_UPDATE_FREQ    = 1     # discriminator steps per generator step
NUM_WORKERS         = 0     # 0 = fastest when patches are fully in RAM

# ── Architecture ─────────────────────────────────────────────────────────────
GEN_BASE_CH  = 64
GEN_DROPOUT  = 0.20

# Generator normalisation: 'instance' | 'group' | 'batch'  (models._norm)
#
# This is a TILING decision, not a regularisation one. InstanceNorm rescales each
# patch by its own spatial statistics, so the same voxel seen by two overlapping
# tiles gets two different content-dependent affine transforms — the tiles then
# disagree by a DC offset that overlap-blending cannot cancel, which shows up as a
# visible patch border in the reconstructed volume (`metrics.seam_energy`).
# 'batch' applies fixed running statistics at eval() and is the only option here
# that is tile-invariant by construction; 'group' is the middle ground.
#
# Left at 'instance' so existing checkpoints keep loading — a run's own
# run_config.json carries this forward to inference. Quantify before switching:
#     python norm_attribution.py --scenario_dir <run>
GEN_NORM     = 'instance'

# ── Baseline loss flags (pix2pixHD combination) ───────────────────────────────
USE_ADVERSARIAL      = False
USE_PERCEPTUAL       = False
USE_FEATURE_MATCHING = False

# ── Extra loss flags (ablation switches) ─────────────────────────────────────
USE_SSIM             = False
USE_GRADIENT         = False
USE_FREQUENCY        = False
USE_ORGAN            = False
# Per-organ MEAN-HU loss. The phase classifier reads per-organ median HU and
# nothing else, so this optimises its input features directly rather than hoping
# a per-voxel loss gets there indirectly. Needs multi-label masks (i.e.
# ORGAN_WEIGHTS set) to be worth much. Never use it alone — a patch can score 0
# on it while looking nothing like the target; it fixes an organ's LEVEL, while
# the organ/L1 terms fix its texture.
USE_HU_PROFILE       = False
USE_SALIENCY         = False
USE_CYCLE            = False
USE_SEG_CONSISTENCY  = False

# ── Loss weights ─────────────────────────────────────────────────────────────
# lambda_l1 is parametric (see CompositeLoss.__init__ in losses.py): L1 stays
# at full strength (LAMBDA_L1) when nothing else competes with pixel fidelity,
# but is reduced to LAMBDA_L1_REDUCED whenever adversarial/perceptual/
# feature_matching is active — those three specifically trade realism for
# exact pixel fidelity (unlike ssim/gradient/frequency/etc, which refine
# *what* L1-style fidelity means rather than compete with it), so L1 needs to
# back off for them to have any actual influence. At the original LAMBDA_L1=100
# ratio, adv_only's own results were the floor of every tested scenario and
# every adversarial-inclusive scenario converged to near-identical metrics
# regardless of which extra losses were added — see scenario_results_overview.md.
# LAMBDA_L1_REDUCED=25 / LAMBDA_ADV=2 matches this codebase's own cited
# baselines (Yan22/Yan23/Yan24c: lambda_l1=25, lambda_adv=2 → 12.5:1 ratio,
# vs the original 100:1).
LAMBDA_L1            = 100.0
LAMBDA_L1_REDUCED    =  25.0
LAMBDA_ADV           =   2.0
# ── L1 decay curriculum ──────────────────────────────────────────────────────
# Three-stage curriculum: structure (L1) → contrast (organ) → texture (adv).
# Global L1 covers every voxel, so while it is at full strength a zero entry in
# ORGAN_WEIGHTS is nearly a no-op — L1 still fits that region hard. Decaying L1
# is what gives those zeros force, handing the late-training gradient budget to
# the organ term.
#
# It decays to a FLOOR, never to 0, and that floor is load-bearing: the organ
# term only sees *labelled* voxels, so background (air/fat/skin/table) and any
# zero-weighted label have NO other constraint anywhere in the composite loss.
# At lambda_l1=0 they would receive zero gradient for the rest of training and
# could drift into arbitrary artefacts — invisible to every organ-region metric.
# If the floor proves too strong for the organ term to matter, lower it to ~10,
# but not to 0.
USE_L1_DECAY         = False   # per-scenario switch; run_scenarios.sh sets it
L1_DECAY_START_EPOCH =  10     # full LAMBDA_L1 up to here
L1_DECAY_END_EPOCH   =  30     # LAMBDA_L1_FLOOR from here on
LAMBDA_L1_FLOOR      =  25.0
LAMBDA_PERCEPTUAL    =  10.0
LAMBDA_FEATURE_MATCH =  10.0
LAMBDA_SSIM          =  10.0
LAMBDA_GRADIENT      =   5.0
LAMBDA_FREQUENCY     =   1.0
# Raised 5.0 → 20.0 on evidence: at λ=5 the organ term reached only 5% of the L1
# term early and 21% after the decay completed — a minority of the gradient
# throughout. The tiered weighting still produced a significant HU improvement at
# that strength (feature L1 −1.58 HU, t=−4.22 vs l1_only), so give it enough
# weight to actually compete. This is the main knob to sweep next.
LAMBDA_ORGAN         =  20.0
# HU-profile term, sized from the measured error rather than guessed. Its raw
# value is a per-organ mean difference in normalised [0,1] units: the observed
# residual is ~15 HU on the 600 HU window (-200..400), i.e. ~0.025 normalised.
# At lambda=10 the term would contribute ~0.25 against an organ term of ~6.5 —
# under 4% of the gradient, which is precisely the mistake that made
# LAMBDA_ORGAN=5 ineffective (it peaked at 21% and moved almost nothing).
# lambda=50 puts it at ~1.25, a fifth of the organ term: enough to matter,
# not enough to dominate the spatial terms it is meant to complement.
LAMBDA_HU_PROFILE    =  50.0
ORGAN_WEIGHT         =  10.0   # legacy uniform mode, used only when ORGAN_WEIGHTS is None

# ── Per-organ loss weights ───────────────────────────────────────────────────
# Rationale (measured on a sample _seg_full mask, cross-referenced against the
# XGBoost phase model's feature importances in orgFeatXGB_CTPhase/retrain_out_full):
#
#   group                     voxel share   phase importance
#   bone + muscle                  36.6%      ~0
#   GI tract                       27.1%      ~1.3%
#   lungs                          15.5%      ~0
#   solid organs                   13.9%      ~33%   (liver alone 0.290)
#   heart                           2.7%      23.5%
#   phase-critical vessels          1.8%      ~35%   (aorta alone 0.293)
#
# Aorta is the single most informative structure for phase and occupies 0.91% of
# labelled voxels — a ~30:1 mismatch between importance and loss share. Left
# unweighted, the gradient is dominated by bowel (stochastic gas/content, not
# inferable from NCCT) and by bone/lung (no contrast information at all), which
# is exactly what the per-organ metrics show: the vessels have both the worst
# SSIM and the worst HU errors (portal vein 40.3 HU, pulmonary vein 33.3 HU).
#
# Keyed by NAME, not id: ids are resolved through TS_LABEL_MAP_JSON at load time
# so a TotalSegmentator version bump can't silently mis-weight organs (id 54-62,
# for instance, are contrast-carrying vessels that a naive id-range scheme misses).
# Unlisted labels get ORGAN_WEIGHT_DEFAULT.
ORGAN_WEIGHT_DEFAULT    = 1.0
ORGAN_WEIGHT_BACKGROUND = 0.5   # label 0: air/fat/skin/table

_GI_TRACT = ['stomach', 'small_bowel', 'duodenum', 'colon', 'esophagus']

ORGAN_WEIGHT_GROUPS = {
    # Phase-critical vessels — tiny by volume, dominant for phase.
    6.0: ['aorta', 'inferior_vena_cava', 'portal_vein_and_splenic_vein'],
    4.0: ['heart', 'pulmonary_vein', 'superior_vena_cava', 'atrial_appendage_left',
          'iliac_artery_left', 'iliac_artery_right',
          'iliac_vena_left', 'iliac_vena_right',
          'brachiocephalic_trunk', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right',
          'subclavian_artery_left', 'subclavian_artery_right',
          'common_carotid_artery_left', 'common_carotid_artery_right'],
    # Solid organs — real contrast uptake, liver is the 2nd-ranked feature.
    3.0: ['liver'],
    2.0: ['spleen', 'pancreas', 'kidney_left', 'kidney_right', 'gallbladder',
          'adrenal_gland_left', 'adrenal_gland_right', 'urinary_bladder'],
    # Contrast-free bulk — kept in the loss, just de-prioritised.
    0.25: ['lung_upper_lobe_left', 'lung_lower_lobe_left', 'lung_upper_lobe_right',
           'lung_middle_lobe_right', 'lung_lower_lobe_right'],
    # GI tract — 27% of labelled voxels, ~1.3% of phase importance, and its
    # gas/content configuration is genuinely not recoverable from NCCT. Excluded
    # from the organ term entirely; the LAMBDA_L1_FLOOR is what still anchors it.
    0.0: _GI_TRACT,
}
# Everything skeletal/muscular (vertebrae_*, rib_*, hip/femur/scapula/humerus/
# clavicula, gluteus_*, autochthon_*, iliopsoas_*, sacrum, skull, sternum,
# costal_cartilages) is matched by prefix rather than enumerated.
ORGAN_WEIGHT_PREFIXES = {
    0.25: ('vertebrae_', 'rib_', 'hip_', 'femur_', 'scapula_', 'humerus_',
           'clavicula_', 'gluteus_', 'autochthon_', 'iliopsoas_',
           'sacrum', 'skull', 'sternum', 'costal_cartilages'),
}
# Named weight schemes. 'gi_zero' is the CONTROL for the ablation: it changes
# exactly one thing (the GI tract is excluded) and leaves everything else at 1.0.
# If it recovers most of 'tiered's gain, the full tiered vector is unnecessary
# complexity and the simpler intervention is the better result to report.
ORGAN_WEIGHT_PRESETS = {
    'tiered':  (ORGAN_WEIGHT_GROUPS, ORGAN_WEIGHT_PREFIXES),
    'gi_zero': ({0.0: _GI_TRACT}, {}),
}
ORGAN_WEIGHT_PRESET = 'tiered'

USE_PER_ORGAN_WEIGHTS = False   # per-scenario switch; run_scenarios.sh sets it
LAMBDA_SALIENCY      =   5.0
LAMBDA_CYCLE         =  10.0
LAMBDA_SEG           =   2.0

ADV_MODE          = 'lsgan'   # 'lsgan' or 'bce'
ADV_WARMUP_EPOCHS =  5
CYCLE_WARMUP_EPOCHS =  5      # ramp lambda_cycle in, same reasoning as adv warmup

# ── Perceptual / saliency backbone choice ─────────────────────────────────────
# 'dino' loads a frozen ViT backbone lazily (see dino_backbone.py), at most
# once, shared between perceptual + saliency — never loaded if both are 'vgg'/
# 'heuristic'.
PERCEPTUAL_BACKBONE = 'vgg'          # 'vgg' | 'dino'
SALIENCY_MODE       = 'heuristic'    # 'heuristic' | 'dino'
SALIENCY_WEIGHT      =   5.0
SALIENCY_THRESHOLD   =   0.08        # only used by 'heuristic' mode

# ── Validation metrics ────────────────────────────────────────────────────────
# Report MAE / PSNR / SSIM / NCC each epoch, both globally and restricted to the
# organ-mask region. When True, the val/test datasets preload the co-registered
# organ mask per patch (train split is unaffected). Organ-region metrics are the
# clinically meaningful ones — that's where the contrast enhancement lives.
REPORT_ORGAN_METRICS = True

# Metric used to pick `best_model.pth` and drive early stopping.
# 'val_org_ssim' (default) | 'val_ssim' | 'val_loss' (legacy global MAE).
# The legacy 'val_loss' is whole-patch MAE, which structurally selects the
# blurriest epoch — L1-optimal output IS the conditional mean — and is dominated
# by background (global PSNR runs ~3.6 dB above organ-region PSNR on this data).
# Falls back to val_loss automatically when no organ mask is available.
SELECTION_METRIC = 'val_org_ssim'

# Per-ORGAN (not just organ-union) breakdown of the 4 metrics: computed per
# TotalSegmentator label id present in the (now multi-label) mask, saved to
# `organ_metrics.json` each epoch. Needs the multi-label seg masks. The id→name
# map is read from ORGAN_LABEL_MAP_JSON (dumped by CTPhase-XGBoost/retrain_xgb.py
# so organ names match that phase model exactly); if the file is missing, organs
# are reported by raw label id (`label_<id>`). Set to False to skip the per-organ
# breakdown but keep the organ-union metrics above.
REPORT_PER_ORGAN_METRICS = True
# The FULL 117-class TotalSegmentator map, written by
# `orgFeatXGB_CTPhase/dump_ts_label_map.py`. NOT the same file as the XGBoost's
# organ_label_map.json — that one is deliberately restricted to the 16 organs the
# phase model consumes (in its trained feature order), so it names only 16 of the
# ~79 labels present in a _seg_full mask and can't be used to build the weight LUT.
#
# Absolute-ised against this file's location: the previous value was relative and
# silently resolved to nothing whenever the process cwd wasn't the repo root,
# which is why every per-organ metric so far is named `label_<id>`.
TS_LABEL_MAP_JSON = str(_HERE / 'orgFeatXGB_CTPhase' / 'retrain_out_full' / 'ts_label_map_total.json')
ORGAN_LABEL_MAP_JSON = TS_LABEL_MAP_JSON


def resolve_organ_weights(enabled: bool = None, preset: str = None):
    """Build {label_id: weight} from the name-keyed groups above.

    Returns None when per-organ weighting is off (→ legacy uniform ORGAN_WEIGHT).
    Raises on an unknown organ name so a TotalSegmentator version mismatch fails
    loudly here rather than silently weighting the wrong anatomy.

    `enabled`/`preset` override USE_PER_ORGAN_WEIGHTS / ORGAN_WEIGHT_PRESET
    (train.py passes the CLI flags).
    """
    if not (USE_PER_ORGAN_WEIGHTS if enabled is None else enabled):
        return None
    preset = preset or ORGAN_WEIGHT_PRESET
    if preset not in ORGAN_WEIGHT_PRESETS:
        raise KeyError(f"unknown organ weight preset '{preset}' — "
                       f"choose from {sorted(ORGAN_WEIGHT_PRESETS)}")
    groups, prefix_rules = ORGAN_WEIGHT_PRESETS[preset]

    p = Path(TS_LABEL_MAP_JSON)
    if not p.exists():
        raise FileNotFoundError(
            f"per-organ loss weights need the full TS label map, missing at {p}. "
            f"Generate it with: python orgFeatXGB_CTPhase/dump_ts_label_map.py"
        )
    name_to_id = json.loads(p.read_text())

    weights, unknown = {}, []
    for w, names in groups.items():
        for n in names:
            if n not in name_to_id:
                unknown.append(n)
            else:
                weights[int(name_to_id[n])] = float(w)
    if unknown:
        raise KeyError(
            f"organ names in preset '{preset}' absent from {p.name}: {unknown}. "
            f"Your TotalSegmentator version differs from the one the weights were "
            f"written for — check the names with dump_ts_label_map.py."
        )
    # Prefix rules apply only to labels no explicit group already claimed.
    for w, prefixes in prefix_rules.items():
        for n, i in name_to_id.items():
            if int(i) not in weights and n.startswith(tuple(prefixes)):
                weights[int(i)] = float(w)
    return weights


ORGAN_WEIGHTS = resolve_organ_weights()

# ── Misc ─────────────────────────────────────────────────────────────────────
USE_AMP              = True
KEEP_N_CHECKPOINTS   = 3
SAVE_SAMPLES_EVERY   = 1
KEEP_N_SAMPLE_EPOCHS = 5

# Which validation patches the per-epoch sample grid shows.
#   'random' (default) — fresh patches each epoch, spread across as many DISTINCT
#       validation cases as possible. Seeded by epoch, so any given epoch's grid
#       is reproducible. Shows whether the model generalises across patients and
#       anatomy, which one fixed batch cannot.
#   'fixed' — the first n patches every epoch (legacy). Better for watching a
#       single patch sharpen epoch-to-epoch, but it is one slice of one case.
# Trade-off worth knowing: 'random' makes epoch-to-epoch comparison of the SAME
# image impossible. The per-row PSNR/SSIM annotations and curves.png cover
# progress tracking; the grid is for spotting failure modes.
SAMPLE_MODE = 'random'
SAMPLE_N    = 4
# Upper clip of the |error| column's colour scale, in normalised [0,1] units.
# Fixed rather than per-image autoscaled, so rows and epochs stay comparable —
# autoscaling makes every row look equally bad and hides progress.
# 0.15 is chosen against the measured errors on this data (global val MAE ≈0.015,
# organ-region ≈0.031): it keeps typical tissue dark while letting the vessel/edge
# hot spots stand out. Raise it if the map looks uniformly saturated (an untrained
# or diverged model will peg it), lower it to bring out fine differences.
SAMPLE_ERR_VMAX = 0.15
EARLY_STOP_PATIENCE  = 30  # was 12: too tight for adversarial/perceptual runs — early
                           # stopping is keyed to val_loss (an L1 proxy), which is
                           # exactly what those losses trade away for realism, so a
                           # tight patience cuts the run right as the trade-off
                           # starts to develop, before it can show up visually.

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Assembled dict ────────────────────────────────────────────────────────────
train_config: dict = dict(
    # paths
    data_dir             = DATA_DIR,
    labels_csv           = LABELS_CSV,
    output_dir           = OUTPUT_DIR,
    cache_dir            = CACHE_DIR,
    # data glob
    file_tag             = FILE_TAG,
    seg_suffix           = SEG_SUFFIX,
    target_phase         = TARGET_PHASE,
    # HU normalisation
    hu_min               = HU_MIN,
    hu_max               = HU_MAX,
    # patch geometry
    patch_size           = PATCH_SIZE,
    patch_depth          = PATCH_DEPTH,
    overlap              = OVERLAP,
    dims                 = DIMS,
    # validity
    min_patch_std        = MIN_PATCH_STD,
    min_patch_mean       = MIN_PATCH_MEAN,
    min_patch_max        = MIN_PATCH_MAX,
    # organ-focused sampling
    organ_focus_frac           = ORGAN_FOCUS_FRAC,
    organ_focus_labels         = ORGAN_FOCUS_LABELS,
    max_focus_candidates_per_vol = MAX_FOCUS_CAND_PER_VOL,
    # RAM budget
    max_train_patches    = MAX_TRAIN_PATCHES,
    max_val_patches      = MAX_VAL_PATCHES,
    # split
    val_split            = VAL_SPLIT,
    test_split           = TEST_SPLIT,
    seed                 = SEED,
    # training
    batch_size           = BATCH_SIZE,
    epochs               = EPOCHS,
    learning_rate        = LR_GEN,
    lr_disc              = LR_DISC,
    betas                = BETAS,
    weight_decay         = WEIGHT_DECAY,
    use_cosine_schedule  = USE_COSINE,
    cosine_t0            = COSINE_T0,
    cosine_tmult         = COSINE_TMULT,
    cosine_eta_min       = COSINE_ETA,
    disc_update_freq     = DISC_UPDATE_FREQ,
    num_workers          = NUM_WORKERS,
    # architecture
    generator_base_channels = GEN_BASE_CH,
    generator_dropout       = GEN_DROPOUT,
    generator_norm          = GEN_NORM,
    # baseline loss flags
    use_adversarial      = USE_ADVERSARIAL,
    use_perceptual       = USE_PERCEPTUAL,
    use_feature_matching = USE_FEATURE_MATCHING,
    # extra loss flags
    use_ssim             = USE_SSIM,
    use_gradient         = USE_GRADIENT,
    use_frequency        = USE_FREQUENCY,
    use_organ            = USE_ORGAN,
    use_hu_profile       = USE_HU_PROFILE,
    use_saliency         = USE_SALIENCY,
    use_cycle            = USE_CYCLE,
    use_seg_consistency  = USE_SEG_CONSISTENCY,
    # loss weights
    lambda_l1            = LAMBDA_L1,
    lambda_l1_reduced    = LAMBDA_L1_REDUCED,
    use_l1_decay         = USE_L1_DECAY,
    l1_decay_start_epoch = L1_DECAY_START_EPOCH,
    l1_decay_end_epoch   = L1_DECAY_END_EPOCH,
    lambda_l1_floor      = LAMBDA_L1_FLOOR,
    lambda_adv           = LAMBDA_ADV,
    lambda_perceptual    = LAMBDA_PERCEPTUAL,
    lambda_feature_match = LAMBDA_FEATURE_MATCH,
    lambda_ssim          = LAMBDA_SSIM,
    lambda_gradient      = LAMBDA_GRADIENT,
    lambda_frequency     = LAMBDA_FREQUENCY,
    lambda_organ         = LAMBDA_ORGAN,
    lambda_hu_profile    = LAMBDA_HU_PROFILE,
    organ_weight         = ORGAN_WEIGHT,
    organ_weights            = ORGAN_WEIGHTS,          # None → legacy uniform mode
    organ_weight_preset      = ORGAN_WEIGHT_PRESET,
    organ_weight_default     = ORGAN_WEIGHT_DEFAULT,
    organ_weight_background  = ORGAN_WEIGHT_BACKGROUND,
    lambda_saliency      = LAMBDA_SALIENCY,
    lambda_cycle         = LAMBDA_CYCLE,
    lambda_seg           = LAMBDA_SEG,
    adv_mode             = ADV_MODE,
    adv_warmup_epochs    = ADV_WARMUP_EPOCHS,
    cycle_warmup_epochs  = CYCLE_WARMUP_EPOCHS,
    perceptual_backbone  = PERCEPTUAL_BACKBONE,
    saliency_mode        = SALIENCY_MODE,
    saliency_weight      = SALIENCY_WEIGHT,
    saliency_threshold   = SALIENCY_THRESHOLD,
    # misc
    use_mixed_precision     = USE_AMP,
    keep_last_n_checkpoints = KEEP_N_CHECKPOINTS,
    save_samples_interval   = SAVE_SAMPLES_EVERY,
    keep_last_n_sample_epochs = KEEP_N_SAMPLE_EPOCHS,
    sample_mode             = SAMPLE_MODE,
    sample_n                = SAMPLE_N,
    sample_err_vmax         = SAMPLE_ERR_VMAX,
    early_stop_patience     = EARLY_STOP_PATIENCE,
    report_organ_metrics    = REPORT_ORGAN_METRICS,
    selection_metric        = SELECTION_METRIC,
    report_per_organ_metrics = REPORT_PER_ORGAN_METRICS,
    organ_label_map_json    = ORGAN_LABEL_MAP_JSON,
    device                  = DEVICE,
)
