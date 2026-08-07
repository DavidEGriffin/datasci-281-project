# BDD100K Scene Classification — README

## Overview
This notebook classifies dashcam frames from the BDD100K dataset into six scene
categories (city street, highway, parking lot, residential, tunnel, gas stations)
using a combination of handcrafted and pretrained-CNN/transformer features.

## Data setup (required before running)

This notebook expects the following files in a shared Google Drive folder:
- `BDD100k DATA/bdd100k_images_100k.zip` — original BDD100K images (train/val/test)
- `BDD100k DATA/bdd100k_labels.zip` — original BDD100K scene labels
- `resampled.zip` — class-balanced, cropped training set (sibling folder to `BDD100k DATA`,
  built separately via `resampling.py`; see that script for how it was generated)

**To run this notebook yourself:**
1. Get the shared Drive folder added to your own Drive: go to "Shared with me" →
   right-click the folder → "Add shortcut to Drive" → choose My Drive.
2. Open this notebook in Google Colab.
3. Run cells top to bottom. The first run will unzip all data locally (~10GB+,
   takes several minutes) and extract features from scratch (~4 hours across all
   three splits, GPU required for ResNet/ViT — set Runtime → Change runtime type → GPU).
4. On subsequent runs, feature extraction is cached — the notebook checks for
   `features_train.npz` / `features_val.npz` / `features_test.npz` on Drive and
   skips re-extraction if they already exist.

## Data split logic
- **Train**: sourced from `resampled.zip` — 69,996 images, resampled to exactly
  11,666 per class via random cropping (see `resampling.py`), to address severe
  natural class imbalance.
- **Val / Test**: sourced directly from BDD100K's official val/test folders —
  9,947 / 19,902 images, natural (imbalanced) class distribution, left untouched
  so evaluation reflects real-world conditions.
- Train images are center-cropped to (600, 1000) by the resampling script; val/test
  images are center-cropped to match at load time. All images are then resized to
  (250, 150) before feature extraction for computational tractability.

## Pipeline structure
1. **Data loading** — mount Drive, unzip, build manifests, verify image/label integrity
2. **Feature extraction** — 12 handcrafted features + ResNet50 + ViT-B/16 embeddings,
   extracted per-split and cached to Drive as `.npz` files
3. **Exploratory analysis** — feature illustrations, PCA, t-SNE across all 14 features
4. **Feature selection** — backward elimination on a stratified train subsample,
   removes features that don't improve validation macro-F1 by more than a noise threshold
5. **Feature set construction** — Set A (10 surviving handcrafted features, 956 dims)
   and Set B (Set A + ResNet + ViT, 3,772 dims), standardized (scaler fit on train only)
6. **Hyperparameter search** — RandomizedSearchCV with stratified CV, logistic regression,
   scored by macro-F1; winner selected by validation performance
7. **Final evaluation** — single held-out test pass per winning model, with confusion
   matrices and per-class classification reports

## Key results
| Model | Test Macro-F1 | Test Accuracy | Test Balanced Accuracy |
|---|---|---|---|
| Set A (handcrafted only) | 0.332 | 0.572 | 0.483 |
| Complex only (ResNet+ViT) | 0.456 | 0.645 | 0.655 |
| Set B (handcrafted + complex) | 0.480 | 0.642 | 0.665 |

Two handcrafted features (HOG, region-wise HSV) were removed during ablation —
both improved validation macro-F1 when dropped, likely due to redundancy with
cheaper features already in the set.

## Known limitations
- Cross-validation scores are computed within the resampled, class-balanced
  training set and are not directly comparable to validation/test performance,
  which reflects the natural class distribution.
- Rare classes (gas stations: 6 test images, tunnel: 49) have inherently noisy
  per-class metrics due to small sample size.
- The Set B hyperparameter search used a narrower search space (4 candidates,
  2-fold CV) than Set A (10 candidates, 3-fold CV) due to longer per-fit time
  on higher-dimensional data.
- Near-duplicate crops from the same source image may appear split across
  training CV folds, inflating in-training CV scores relative to true
  generalization performance (see Generalizability discussion in the final report).

## Files produced
- `features_{train,val,test}.npz` — cached extracted features, all 14 types
- `features_{train,val,test}_timing.csv` — per-feature extraction timing
- `X_{train,val,test}_{A,B}_scaled.npy` — final standardized feature sets
- `y_{train,val,test}_full.npy` — corresponding labels
- `tuning_lr_set_{a,b}.joblib` — saved hyperparameter search objects
- `best_model_set_{a,b}.joblib` — final trained models
- `hyperparameter_tuning_results.csv` — full search comparison table
- `ablation_elimination_log.csv` — backward elimination history
