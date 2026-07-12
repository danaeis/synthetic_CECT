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

from pathlib import Path
import torch

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR   = '../../sample_data_reg/ncct_cect/vindr_ds/all_baseline_algorithms/B2_deeds__aligned'
LABELS_CSV = '../../sample_data_reg/ncct_cect/vindr_ds/labels.csv'
OUTPUT_DIR = Path('../../simlified_train/literature_baseline')
# Fixed, independent of OUTPUT_DIR: preloaded patches are cached here keyed by
# data/geometry config (not loss flags), so scenario runs that only toggle
# loss flags reuse the same cache instead of re-preloading from scratch.
CACHE_DIR  = Path('../../simlified_train/patch_cache')

# ── Data / glob ───────────────────────────────────────────────────────────────
FILE_TAG     = '_deeds'           # suffix before .nii.gz (same as autoenc_fresh glob)
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

# ── RAM preload budget ────────────────────────────────────────────────────────
MAX_TRAIN_PATCHES = 20_000
MAX_VAL_PATCHES   =  4_000

# ── Train/val/test split ─────────────────────────────────────────────────────
VAL_SPLIT  = 0.15
TEST_SPLIT = 0.15
SEED       = 42

# ── Training schedule ────────────────────────────────────────────────────────
BATCH_SIZE   = 16
EPOCHS       = 80
LR_GEN       = 2e-4
LR_DISC      = 1e-4
BETAS        = (0.5, 0.999)
WEIGHT_DECAY = 1e-5

USE_COSINE    = True
COSINE_T0     = 15
COSINE_TMULT  = 2
COSINE_ETA    = 5e-7

DISC_UPDATE_FREQ    = 1     # discriminator steps per generator step
NUM_WORKERS         = 0     # 0 = fastest when patches are fully in RAM

# ── Architecture ─────────────────────────────────────────────────────────────
GEN_BASE_CH  = 64
GEN_DROPOUT  = 0.20

# ── Baseline loss flags (pix2pixHD combination) ───────────────────────────────
USE_ADVERSARIAL      = False
USE_PERCEPTUAL       = False
USE_FEATURE_MATCHING = False

# ── Extra loss flags (ablation switches) ─────────────────────────────────────
USE_SSIM             = False
USE_GRADIENT         = False
USE_FREQUENCY        = False
USE_ORGAN            = False
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
LAMBDA_PERCEPTUAL    =  10.0
LAMBDA_FEATURE_MATCH =  10.0
LAMBDA_SSIM          =  10.0
LAMBDA_GRADIENT      =   5.0
LAMBDA_FREQUENCY     =   1.0
LAMBDA_ORGAN         =   5.0
ORGAN_WEIGHT         =  10.0
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

# ── Misc ─────────────────────────────────────────────────────────────────────
USE_AMP              = True
KEEP_N_CHECKPOINTS   = 3
SAVE_SAMPLES_EVERY   = 1
KEEP_N_SAMPLE_EPOCHS = 5
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
    # baseline loss flags
    use_adversarial      = USE_ADVERSARIAL,
    use_perceptual       = USE_PERCEPTUAL,
    use_feature_matching = USE_FEATURE_MATCHING,
    # extra loss flags
    use_ssim             = USE_SSIM,
    use_gradient         = USE_GRADIENT,
    use_frequency        = USE_FREQUENCY,
    use_organ            = USE_ORGAN,
    use_saliency         = USE_SALIENCY,
    use_cycle            = USE_CYCLE,
    use_seg_consistency  = USE_SEG_CONSISTENCY,
    # loss weights
    lambda_l1            = LAMBDA_L1,
    lambda_l1_reduced    = LAMBDA_L1_REDUCED,
    lambda_adv           = LAMBDA_ADV,
    lambda_perceptual    = LAMBDA_PERCEPTUAL,
    lambda_feature_match = LAMBDA_FEATURE_MATCH,
    lambda_ssim          = LAMBDA_SSIM,
    lambda_gradient      = LAMBDA_GRADIENT,
    lambda_frequency     = LAMBDA_FREQUENCY,
    lambda_organ         = LAMBDA_ORGAN,
    organ_weight         = ORGAN_WEIGHT,
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
    early_stop_patience     = EARLY_STOP_PATIENCE,
    device                  = DEVICE,
)
