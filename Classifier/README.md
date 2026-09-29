# Organ Classifier

Thoracic disease classification on NIH ChestX-ray14, used to test whether the
lung images produced by [OrGAN](https://github.com/kawsarahmed93/OrGAN_code)
retain diagnostic information — and whether pairing them with the original
radiograph helps.

Three configurations, trained and evaluated identically under 5-fold
cross-validation:

| config | input | model |
|---|---|---|
| `xray_base` | chest X-ray only | DenseNet-121 |
| `lung_base` | OrGAN lung image only | DenseNet-121 |
| `proposed` | chest X-ray **+** lung image | two-stream gated fusion |

## Layout

```
code/configs.py             dataset paths, label names, the three configurations
code/datasets.py            NIH image- and box-level datasets, transforms
code/models.py              DenseNet-121 and the two-stream FusionNet
code/main.py                cross-validation training entry point
code/trainer.py             training loop
code/trainer_callbacks.py   checkpointing, early stopping, seeding
code/utils.py               helpers
code/test.py                per-class AUC on the held-out test split
code/test_iou.py            Grad-CAM++ localisation against NIH bounding boxes
code/compare_variants.py    paired per-fold comparison of trained variants
```

## Data layout

Paths in `configs.py` are relative to `code/`:

```
../NIH-CXR/images/          NIH ChestX-ray14 images
../NIH-CXR/lungs/           OrGAN lung images, same filenames as images/
../weights_cv/<run>/foldN/  checkpoints, written per fold
code/label_info/            split dictionaries and bbox_data.npy
```

Generate `../NIH-CXR/lungs/` with `getLung.py` from the OrGAN repository. The
fusion configuration reads a lung image for every X-ray by substituting
`images` → `lungs` in the path, so the two directories must correspond
filename-for-filename.

`label_info/` holds the cross-validation split dictionaries and `bbox_data.npy`
(the 984 NIH bounding-box annotations). These are not included here; see
`configs.py` for the expected filenames.

## Training

```bash
cd code
python main.py --run_configs_list proposed
```

## Evaluation

Per-class AUC:

```bash
python test.py --run_configs_list proposed
```

Grad-CAM++ localisation against the NIH boxes:

```bash
python test_iou.py --run_configs_list proposed --n_workers 8
```

`test_iou.py` writes two workbooks to the run directory:

- `final_cross_fold_metrics_<config>.xlsx` — IoU, Dice, precision, recall, F1 per finding
- `final_iou_accuracy_<config>.xlsx` — IoU accuracy per threshold, two sheets:
  - **`cross_fold`** — mean ± s.d. across the five cross-validation models, each
    evaluated on the same held-out boxes. This is the sheet to report: the ±
    is model variability.
  - `pooled` — the same models' predictions pooled. Its ± describes
    image-to-image spread, not uncertainty, and its box count is 5× the number
    of distinct boxes. Reference only.

Both sheets give per-finding accuracy with the box count in each column header,
a macro `accuracy` (findings weighted equally) and a box-weighted
`accuracy_micro`, plus `n_findings` and `n_boxes` so the weighting is explicit.
Findings with no annotated boxes are excluded rather than scored zero.

A third sheet, `per_fold`, records the raw per-fold values in long form
(`fold`, `threshold`, `metric`, `value`), which is what the paired comparison
below consumes.

`--images-root` points the evaluation at a local copy of the images, which is
worth using when the image store is slow — each image is read once per fold.
`--weight-path` overrides the run directory, so one config can be evaluated
against several trained variants without editing `configs.py`.

## Comparing variants

Fold *i* is the same training split in every run, so fold *i* of two variants is
a matched pair. Pairing matters: the fold-to-fold variance that dominates each
variant's own standard deviation is shared by both and cancels in the
difference, so a gap sitting well inside either variant's s.d. can still be
consistent across every fold.

```bash
python test_iou.py --run_configs_list proposed --weight-path ../weights_cv/<variant>/   # per variant
python compare_variants.py --a <variant_a> --b <variant_b> [<variant_c> ...] --metric macro micro
```

This runs a sign test on the paired per-fold differences at each threshold,
with Wilcoxon signed-rank alongside, and writes
`paired_comparison_<a>.xlsx` (the tests, plus the raw per-fold values).

With five folds the sign test bottoms out at p = 1/2^5 = 0.031 one-sided
(0.0625 two-sided), reached only on a clean 5–0 sweep — that is the most
significance five folds can carry. Testing all seven thresholds is seven tests,
so pre-specify a single endpoint (one threshold, or the mean across them) if
you intend to quote a p-value; otherwise report the sweep pattern descriptively
and note that the nested thresholds are strongly correlated.

## License

MIT.
