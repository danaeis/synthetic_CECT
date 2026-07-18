# Weights & environment setup (remote server)

The phase-detection module uses two **frozen, pretrained** encoders. Neither is
allowed to fall back to random weights — `encoders.py` raises if real weights
can't be loaded. Set them up once on the remote box before running
`run_phase_detection.py`.

---

## 1. Python dependencies

```bash
source /media/external20/saeedeh_danaei/ncct_env/bin/activate   # or the phase-detection env
pip install -U torch torchvision timm monai nibabel scikit-learn pandas matplotlib einops
```

Notes:
- `einops` is required by `MedViT/MedViT.py` (import fails without it).
- `monai` + `nibabel` are used by `phase_data.py` for volume loading.
- `timm` provides the DINO backbone (see §3).

---

## 2. MedViT weights

MedViT is a medical-imaging ViT; the backbone must be initialised from the
official ImageNet/medical pretrained checkpoint, then we drop its classification
head and use the pooled features.

**Official source:** MedViT repo — https://github.com/Omid-Nejad/MedViT
(paper: "MedViT: A Robust Vision Transformer for Generalized Medical Image
Classification"). Download the `MedViT-small` checkpoint from that repo's
Releases / the Google-Drive link in its README.

There is a helper in this directory (`download_medvit.py`, kept for reference)
that points at a release URL, but **verify the URL against the repo's current
README before trusting it** — release asset paths move. Fetch manually if unsure:

```bash
cd /media/external20/saeedeh_danaei/synthesis_ct/literature_baseline/phase_detection
# Option A — via the helper (verify URL first):
python -c "from download_medvit import download_medvit_checkpoint; download_medvit_checkpoint('small')"
# Option B — manual (replace with the actual asset URL from the repo README):
# wget -O medvit_small.pth "<URL from https://github.com/Omid-Nejad/MedViT releases>"
```

Then pass the path explicitly:

```bash
--medvit_pretrained_path /abs/path/to/medvit_small.pth
```

`encoders.py` will **raise `FileNotFoundError`** if that path is missing, and
**raise `RuntimeError`** if the checkpoint matches <50% of the backbone's weights
(wrong file) — so a silent random-init run (the prior codebase's bug) can't happen.

---

## 3. DINOv3 weights

`DinoV3VolumeEncoder` loads real DINO weights in priority order, all via `timm`
(no manual download needed — `timm` fetches from HuggingFace on first use):

1. **DINOv3** — first `*dinov3*vits*` model in your installed `timm` registry
   (only if your `timm` version ships DINOv3; may be HF-gated on first download).
2. **DINOv2** — `vit_small_patch14_dinov2.lvd142m` (ungated, works today).

Check what's available before running:

```bash
python -c "import timm; print([m for m in timm.list_models() if 'dinov3' in m])"
python -c "import timm; timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=True, num_classes=0); print('DINOv2 OK')"
```

If DINOv3 is HF-gated, run `huggingface-cli login` and accept the model license
on its HF page first; otherwise the module transparently uses real DINOv2. It
will **never** fall back to a supervised ViT or random weights (that was the old
`dino_encoder.py` bug) — it raises instead.

---

## 4. Run

```bash
cd /media/external20/saeedeh_danaei/synthesis_ct/literature_baseline/phase_detection

# CPU-only logic check first (seconds, no weights/data needed):
python smoke_test_phase.py

# Full run on the FULL dataset (not a subset), both encoders, all 3 heads:
python run_phase_detection.py \
    --data_dir  /path/to/full/volumes \
    --labels_csv /path/to/labels.csv \
    --encoders medvit dinov3 \
    --medvit_pretrained_path /abs/path/to/medvit_small.pth \
    --output_dir phase_results
```

Outputs in `phase_results/`:
- `phase_detection_summary.json` — CV (GroupKFold) + held-out test acc/F1 per
  encoder × head (lda / pca_lda / linear_nn).
- `phase_detection_full.json` — same, plus per-class reports and confusion matrices.
- `phase_head_comparison.png` — CV vs test accuracy bar chart across all combos.
- `feature_cache/*.npz` — extracted features (reused on re-runs via `--use_cache`).

---

## 5. Interpreting the result vs the old ~68%

The old ~68% CV was computed on 2×-duplicated, patient-leaking data (see
`RETRAIN_PLAN.md`). This pipeline removes both issues, so the number it reports is
the **honest** one and may start lower — that is expected and correct. Compare the
three heads: if `lda`/`pca_lda`/`linear_nn` land close together, the frozen-feature
ceiling is the limit (next lever: fine-tune the encoder, `RETRAIN_PLAN.md` item 5);
if `linear_nn` clearly wins, a learned head is worth keeping.
