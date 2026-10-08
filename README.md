# Acne Severity Assessment in Facial Images

**AcneFusionNet**: a multi-task deep learning model that counts acne lesions and grades acne severity from facial images.

- **Students:** Maya Metanis, Razan Jiryis
- **Supervisors:** Yair Moshe, Dr. Badea Jiryis (medical advisor)
- **Lab:** Signal and Image Processing Lab (SIPL), Faculty of Electrical & Computer Engineering, Technion, 2026
- **In collaboration with:** Rambam Health Care Campus
- **Project page:** [SIPL project 10366-2-26](https://sipl.ece.technion.ac.il/projects/project-details/?prj_id=11393)
- **Poster:** [docs/poster.pdf](docs/poster.pdf) ([image on the lab site](https://sipl.ece.technion.ac.il/wp-content/uploads/2026/03/Poster10366-scaled.png))

## Overview

Manual acne assessment is slow and varies between clinicians. AcneFusionNet combines lesion counting and severity grading in one network:

- **Shared backbone:** EfficientNet-B2 extracts multi-scale features.
- **Counting branch:** a dilated-context decoder predicts a pixel-wise density map. Its integral gives the predicted lesion count.
- **Feature fusion:** the predicted count is appended to the global feature vector before severity classification, because lesion count is what defines severity under the Hayashi criterion.
- **Severity branch (LDL):** predicts a label distribution over the 4 Hayashi grades (Mild, Moderate, Severe, Very Severe) to model grading uncertainty.
- **Loss:** `L_total = L_count (MSE on density maps) + λ · L_severity (weighted KL divergence)`, with λ = 7.

The model is intended as a decision-support tool, not a replacement for clinical evaluation.

## Results

Evaluated on the ACNE04 dataset (1,457 images) with 5-fold cross-validation:

| Model | Method | Accuracy |
|---|---|---|
| Wu et al. (2019) | LDL + Counting | 84.11% |
| **AcneFusionNet (ours)** | **Fusion + LDL** | **85.86%** |

## Repository structure

```
dataset_checks/                 Dataset integrity audits and their reports
project_1_dataset_processing/   Data preparation pipeline (steps 01-10)
project_2_models/               AcneFusionNet model (backbone, density head, fusion model)
project_3_training/             Dataset loader, multi-task loss, 5-fold training script
docs/                           Project poster
```

## Setup

```bash
pip install -r requirements.txt
```

### Data

The dataset is **not** included in this repository. Download ACNE04 from the [original authors (Wu et al., 2019)](https://github.com/xpwu95/LDL) and arrange it in the repository root as:

```
project_0_datasets/
├── images/            original .jpg images
├── bounding_boxes/    .xml lesion annotations
└── original_splits/   official 5-fold split files
```

Trained model weights are not included either.

## Usage

Run every script from the repository root.

1. **Optional dataset audit:** `python dataset_checks/integrity_audit.py` and `python dataset_checks/full_audit.py`.
2. **Data preparation:** run the scripts in `project_1_dataset_processing/` in numerical order (`01_...` to `10_...`). They compute lesion counts from the XML annotations, check Hayashi consistency, create train/val/test splits, resize images to 512×512, rescale the bounding boxes, generate attention masks and density maps, and write the final training manifests with LDL targets.
   ```bash
   python project_1_dataset_processing/01_get_bbox_count_as_gt.py
   # ... through ...
   python project_1_dataset_processing/10_apply_ldl_distribution.py
   ```
   Each script accepts `--project_root` (default: current directory).
3. **Training:** this trains all 5 folds and prints the mean cross-validation accuracy. The best weights for each fold are saved to `outputs/fold_<k>/best_weights.pt`.
   ```bash
   python project_3_training/03_trainer_ultimate.py
   ```

## References

- Wu et al., *Joint Acne Image Grading and Counting via Label Distribution Learning*, ICCV 2019.
- Alzahrani et al., *Attention Mechanism Guided Deep Regression Model for Acne Severity Grading*, 2022.
- Hayashi et al., *Establishment of grading criteria for acne severity*, 2008.
