# Genomic & Neuroimaging Data Fusion Risk Model

An end-to-end bioinformatics and neuroinformatics pipeline that integrates synthetic genomic biomarkers, structural MRI volumetrics, and clinical outcome metrics. The project utilizes an in-memory SQL feature engineering layer and trains multi-modal machine learning models (Random Forest) to assess conversion risk, demonstrating performance improvements over single-modality baseline models.

---

## 📌 Key Highlights

- **Multi-Modal Data Integration:** Combines genetic risk factors (`APOE-e4` allele counts), structural brain MRI features (hippocampal volume, cortical thickness, eTIV), and patient demographics.
- **In-Memory SQL Feature Engineering:** Uses an in-memory SQLite engine to run automated relational joins (`01_genomic_imaging_join.sql`) across independent raw CSV data streams.
- **Model Fusion & Evaluation:** Trains baseline neuroimaging-only models against multi-modal fused models, automatically generating performance plots and ROC-AUC comparisons.
- **Single-Command Pipeline:** Fully modularized architecture executable via a unified `main.py` entry point.

---

## 📊 Performance Summary

| Model Variant | Input Modalities | Metric (ROC-AUC) | Lift vs Baseline |
| :--- | :--- | :---: | :---: |
| **Baseline** | Neuroimaging Only (MRI) | 0.9755 | — |
| **Fused Model** | Genomic + Neuroimaging + Clinical | **0.9847** | **+0.92%** |

*Note: Visual evaluation graphs are automatically saved to `assets/model_performance_comparison.png` upon execution.*

---

## 📁 Repository Structure

```text
genomic-neuro-fusion-risk/
├── assets/
│   └── model_performance_comparison.png  # Generated evaluation plots
├── data/
│   ├── clinical_outcomes.csv              # Synthetic clinical target metadata
│   ├── genomic_biomarkers.csv             # Synthetic APOE-e4 & genetic markers
│   └── mri_volumetrics.csv                # Synthetic brain structure volumes
├── queries/
│   └── 01_genomic_imaging_join.sql        # SQL feature engineering join script
├── src/
│   ├── data_loader.py                     # SQLite engine & data preprocessing
│   ├── generate_data.py                   # Realistic synthetic data generator
│   └── model_trainer.py                   # Random Forest training & evaluation
├── main.py                                # End-to-end pipeline launcher
└── README.md                              # Project documentation