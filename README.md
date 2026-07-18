# Segment-anything-4DFlow: SAM & SAM 2 for 4D Flow MRI and DSA Images

[![Paper 1](https://img.shields.io/badge/Paper%201-SPIE%202025-blue?style=flat)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nxq9It8AAAAJ&sortby=pubdate&citation_for_view=nxq9It8AAAAJ:vRqMK49ujn8C)
[![Paper 2](https://img.shields.io/badge/Paper%202-SPIE%202024-blue?style=flat)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=nxq9It8AAAAJ&sortby=pubdate&citation_for_view=nxq9It8AAAAJ:l7t_Zn2s7bgC)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org)

> **Official implementation of two publications** applying the **Segment Anything Model (SAM)**, **SAM 2**, and **MedSAM** to 4D flow MRI and cerebral DSA images for semi-automated segmentation in cardiovascular and intracranial applications.

---

## 📚 Publications

### 1. Semi-Automated Segmentation of 4D Flow MRI with SAM 2 (SPIE 2025)

**Authors:** Amirkhosro Kazemi, A Ghazipour, T Settle, MF Stoddard, Amir A Amini  
**Venue:** Medical Imaging 2025: Clinical and Biomedical Imaging, SPIE  

Semi-automated segmentation of **magnitude images in 4D flow MRI** using the Segment Anything Model 2 (SAM 2), enabling accurate vessel boundary delineation for hemodynamic analysis without manual slice-by-slice annotation.

```bibtex
@inproceedings{kazemi2025semi,
  title     = {Semi-automated segmentation of magnitude images in 4D flow MR
               scans using segment anything model 2 (SAM 2)},
  author    = {Kazemi, Amirkhosro and Ghazipour, A and Settle, T and
               Stoddard, MF and Amini, Amir A},
  booktitle = {Medical Imaging 2025: Clinical and Biomedical Imaging},
  year      = {2025},
  organization = {SPIE}
}
```

---

### 2. SAM and MedSAM for DSA Segmentation in IIH (SPIE 2024)

**Authors:** Amirkhosro Kazemi, Dale Ding, MJ Negahdar, Isaac Josh Abecassis, Amir A Amini  
**Venue:** Medical Imaging 2024: Clinical and Biomedical Imaging (Vol. 12930, pp. 620–628), SPIE  

Evaluation of **SAM and MedSAM** for segmenting cerebral **digital subtraction angiography (DSA)** images in patients with **Idiopathic Intracranial Hypertension (IIH)** and **venous sinus stenosis**, with comparison of zero-shot and prompted inference.

```bibtex
@inproceedings{kazemi2024segmentation,
  title     = {Segmentation of cerebral digital subtraction angiography (DSA)
               images in idiopathic intracranial hypertension and venous sinus
               stenosis: evaluating the efficacy of the segment anything model
               (SAM) and MedSAM},
  author    = {Kazemi, Amirkhosro and Ding, Dale and Negahdar, MJ and
               Abecassis, Isaac Josh and Amini, Amir A},
  booktitle = {Medical Imaging 2024: Clinical and Biomedical Imaging},
  volume    = {12930},
  pages     = {620--628},
  year      = {2024},
  organization = {SPIE}
}
```

---

## 📁 Repository Structure

```
Segment-anything-4DFlow/
├── src/
│   ├── sam_segmenter.py    # SAM / MedSAM wrapper for prompted inference
│   ├── sam2_segmenter.py   # SAM 2 video segmenter for 3D/4D MRI volumes
│   ├── preprocessing.py    # MRI/DSA preprocessing (normalisation, windowing)
│   ├── postprocessing.py   # Mask refinement, hole-filling, largest component
│   └── metrics.py          # Dice, IoU, Hausdorff distance
├── segment_4dflow.py       # Run SAM 2 on a 4D flow MRI magnitude volume
├── segment_dsa.py          # Run SAM / MedSAM on DSA images
├── evaluate.py             # Quantitative evaluation against ground truth masks
├── visualize.py            # Overlay masks on images, save as PNG / GIF
├── configs/
│   └── config.yaml         # Model paths, thresholds, prompts
├── data/
│   └── README.md           # How to organise your input data
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/kazemiakk/Segment-anything-4DFlow.git
cd Segment-anything-4DFlow

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Download Model Checkpoints

| Model | Download |
|-------|----------|
| SAM (ViT-H) | [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| MedSAM | [medsam_vit_b.pth](https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN) |
| SAM 2 (hiera-large) | [sam2_hiera_large.pt](https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt) |

Place downloaded checkpoints in `./checkpoints/`.

---

## 📂 Data Format

### 4D Flow MRI (Paper 1)
```
data/
└── 4dflow/
    ├── subject_001/
    │   ├── magnitude.nii.gz   ← 4D magnitude volume (X, Y, Z, T)
    │   └── mask_gt.nii.gz     ← ground truth mask (optional, for evaluation)
    └── ...
```

### DSA Images (Paper 2)
```
data/
└── dsa/
    ├── patient_001/
    │   ├── frame_001.png
    │   ├── frame_002.png
    │   └── mask_gt.png        ← ground truth mask (optional)
    └── ...
```

---

## 🚀 Usage

### Paper 1: SAM 2 Segmentation of 4D Flow MRI

```bash
python segment_4dflow.py \
  --input    ./data/4dflow/subject_001/magnitude.nii.gz \
  --checkpoint ./checkpoints/sam2_hiera_large.pt \
  --model_cfg sam2_hiera_l \
  --output   ./results/subject_001_mask.nii.gz \
  --prompt_slice 10 \
  --prompt_point 128 128
```

### Paper 2: SAM / MedSAM on DSA Images

```bash
# SAM (with bounding box prompt)
python segment_dsa.py \
  --input_dir  ./data/dsa/patient_001 \
  --checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --model_type sam \
  --output_dir ./results/patient_001 \
  --bbox 50 60 200 220

# MedSAM
python segment_dsa.py \
  --input_dir  ./data/dsa/patient_001 \
  --checkpoint ./checkpoints/medsam_vit_b.pth \
  --model_type medsam \
  --output_dir ./results/patient_001 \
  --bbox 50 60 200 220
```

### Evaluation

```bash
python evaluate.py \
  --pred_dir ./results/subject_001 \
  --gt_dir   ./data/4dflow/subject_001 \
  --output_dir ./results/eval
```

### Visualisation

```bash
python visualize.py \
  --image ./data/dsa/patient_001/frame_001.png \
  --mask  ./results/patient_001/frame_001_mask.png \
  --output ./results/overlay.png
```

---

## 🏛️ Acknowledgements

This work was supported by the **National Institutes of Health** and conducted at the **Medical Imaging Lab, University of Louisville** and the **Robley Rex Veterans Affairs Medical Center, Louisville, KY**.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
