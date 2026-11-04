# Leakage-Controlled Benchmarking of Evidential Multitask Learning for Bitter Gourd Leaf Disease Recognition and Severity Grading

Official code accompanying the paper of the same title.

**Author:** Ranjit Chandro Ray
**Contact:** ranjit.mscs@gmail.com
**Paper:** Manuscript in preparation.
**Code archive (Zenodo):** https://doi.org/[DOI-CODE]
**Dataset and extended results (Zenodo, restricted access):** https://doi.org/[DOI-DATA]

---

## 1. Overview

This repository contains the full experimental pipeline for bitter gourd (*Momordica charantia*) leaf disease recognition and severity grading. The study emphasises **leakage-controlled evaluation** and reports:

- Near-duplicate removal (flip/rotation-invariant perceptual hashing) **before** stratified 5-fold splitting
- An **evidential (Dirichlet) classification head** with uncertainty-based abstention
- A **CORAL ordinal head** for three-level severity grading (Mild / Moderate / Severe)
- Comparison against five CNN baselines and a Swin-T variant, plus two ablations
- Calibration (ECE, Brier), corruption robustness, Grad-CAM faithfulness (Insertion AUC), and statistical testing (95 % CI, Wilcoxon, McNemar)

Proposed model: **BG-EDLNet-ConvNeXt** (ConvNeXt-Tiny backbone).

## 2. Repository structure

```
.
├── first-paper-final-code.ipynb   # complete pipeline (Steps 0–8)
├── config.yaml                    # documented hyper-parameters and seeds
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT
└── README.md
```

The notebook is organised in numbered cells:

| Step | Content |
|------|---------|
| 0 | Configuration (`CFG`), imports, dataset indexing |
| 0A | HSV-based lesion segmentation and severity labels |
| 0B | CLAHE pre-processing (cached offline) and augmentation pipelines |
| 1 | Near-duplicate removal (pHash) and stratified 5-fold split |
| 2 | Backbones and model zoo (timm, ImageNet-pretrained) |
| 3 | Evidential head, EDL loss, abstention rule |
| 4 | CORAL ordinal loss, metrics, training engine |
| 5 | Corruption robustness (lighting, blur, noise, occlusion) |
| 6 | Grad-CAM with Insertion-AUC faithfulness |
| 7 | Full protocol (seeds × folds × models) and statistical tests |
| 8 | Tables and figures |

## 3. Dataset

| Class | Images |
|-------|-------:|
| Anthracnose | 410 |
| Downey mildew | 570 |
| Fresh leaf | 551 |
| Fusarium wilt | 502 |
| Mosaic virus | 600 |
| **Total** | **2,633** |

After near-duplicate removal, 2,439 images are used for the 5-fold evaluation.

The dataset is archived on Zenodo under **restricted access** (DOI above). Access can be requested through the Zenodo record or by e-mail to the corresponding author, for non-commercial academic use. Place the data in the following layout:

```
Bitter Gourd/
├── Anthracnose/
├── Downey mildew/
├── Fresh leaf/
├── Fusarium wilt/
└── Mosaic virus/
```

## 4. Installation

```bash
git clone https://github.com/ranjit-research/bg-edlnet.git
cd bg-edlnet
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version (see https://pytorch.org).

## 5. Usage

1. Obtain the dataset (Section 3).
2. Open `first-paper-final-code.ipynb`.
3. In **Cell 0.0**, set `CFG.DATA_ROOT` to the dataset folder. The notebook ships with the Kaggle path `/kaggle/input/datasets/ranjit2027/bitter-gourd-dataset/Bitter Gourd`.
4. Run all cells in order.

All hyper-parameters live in the `CFG` class of Cell 0.0; `config.yaml` documents the same values. A quick smoke test (1 seed, 1 fold, 2 epochs) is available by setting `CFG.QUICK_TEST = True`.

## 6. Reproducibility

- **Seeds:** split seed 42; training seeds 42, 123, 2024; robustness seed 1234. Each run uses `seed × 100 + fold`.
- **Environment:** Kaggle notebook, Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), timm 1.0.26, albumentations 2.0.8, OpenCV 4.13.0, NumPy 2.0.2, NVIDIA Tesla T4, fp16 mixed precision.
- **Runtime:** about 10.9 hours for the complete protocol (3 seeds × 5 folds × 10 models) on Kaggle.
- **Checkpoints:** none are stored. Models are trained in memory and the best weights are restored in RAM after early stopping. Reported results therefore come directly from a full run of the notebook.
- **Non-determinism:** `cudnn.benchmark` and mixed precision are enabled, so re-runs on different hardware can differ slightly in the last decimals.
- **Pre-trained weights:** ImageNet weights are downloaded automatically by `timm` (internet access required on the first run).

## 7. Outputs

The notebook writes `severity.csv`, the CLAHE cache, and the tables/figures of Step 8 to `working/` (`reports/` for tables and figures). These outputs, together with training logs, are archived on Zenodo (DOI above).

## 8. Citation

```bibtex
@unpublished{ray2026bgedlnet,
  title  = {Leakage-Controlled Benchmarking of Evidential Multitask Learning for Bitter Gourd Leaf Disease Recognition and Severity Grading},
  author = {Ray, Ranjit Chandro},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

If you use the code, please also cite the Zenodo archive: https://doi.org/[DOI-CODE].

## 9. License

The code is released under the [MIT License](LICENSE). The dataset is distributed separately under the terms stated in its Zenodo record.

## 10. Contact

Ranjit Chandro Ray - ranjit.mscs@gmail.com
