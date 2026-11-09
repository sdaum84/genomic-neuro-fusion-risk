import os
import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic subjects
num_subjects = 500

# 1. Demographic & Genomic Data
subject_ids = [f"SUBJ_{i:04d}" for i in range(1, num_subjects + 1)]
ages = np.random.normal(loc=72, scale=6.5, size=num_subjects).clip(55, 90).round(1)
sexes = np.random.choice(["M", "F"], size=num_subjects, p=[0.48, 0.52])

# APOE-e4 allele count: 0 (60%), 1 (30%), 2 (10%)
apoe_e4_counts = np.random.choice([0, 1, 2], size=num_subjects, p=[0.60, 0.30, 0.10])
genomic_df = pd.DataFrame({
    "subject_id": subject_ids,
    "age": ages,
    "sex": sexes,
    "apoe_e4_count": apoe_e4_counts
})


# 2. Structural MRI Features (Volumes in cm^3, Cortical Thickness in mm)
# Higher APOE-e4 count & older age accelerate volume loss
eTIV = np.random.normal(loc=1500, scale=120, size=num_subjects) # Estimated Total Intracranial Volume

# Hippocampal Volume (atrophy correlated with APOE status + age)
base_hippocampus = 7.5 - (ages - 55) * 0.04 - (apoe_e4_counts * 0.4)
hippocampal_vol = base_hippocampus + np.random.normal(loc=0, scale=0.4, size=num_subjects)

# Ventricular Volume (expands with neurodegeneration)
base_ventricles = 25.0 + (ages - 55) * 0.5 + (apoe_e4_counts * 3.0)
ventricles_vol = base_ventricles + np.random.normal(loc=0, scale=4.0, size=num_subjects)

# Entorhinal Cortical Thickness
base_entorhinal = 3.6 - (ages - 55) * 0.015 - (apoe_e4_counts * 0.15)
entorhinal_thickness = base_entorhinal + np.random.normal(loc=0, scale=0.2, size=num_subjects)

imaging_df = pd.DataFrame({
    "subject_id": subject_ids,
    "eTIV_cm3": eTIV.round(1),
    "hippocampus_vol_cm3": hippocampal_vol.round(2),
    "ventricles_vol_cm3": ventricles_vol.round(2),
    "entorhinal_thickness_mm": entorhinal_thickness.round(2)
})

# 3. Clinical Target Variable (Risk Conversion: 0 = Resilient, 1 = High Risk/Converter)
# Risk score combines age, genetic risk, and MRI atrophy
risk_score = (
    0.05 * (ages - 55)
    + 0.8 * apoe_e4_counts
    - 1.2 * (hippocampal_vol - 7.0)
    + 0.05 * (ventricles_vol - 30)
    - 1.5 * (entorhinal_thickness - 3.2)
)
probabilities = 1 / (1 + np.exp(-risk_score)) # Sigmoid function
clinical_status = (probabilities > 0.5).astype(int)

clinical_df = pd.DataFrame({
    "subject_id": subject_ids,
    "conversion_status": clinical_status
})

# Save CSV datasets into data/ folder
os.makedirs("data", exist_ok=True)
genomic_df.to_csv("data/genomic_biomarkers.csv", index=False)
imaging_df.to_csv("data/mri_volumetrics.csv", index=False)
clinical_df.to_csv("data/clinical_outcomes.csv", index=False)

print("✅ Data Generation Complete!")
print(f"Generated {num_subjects} patient records across genomic, MRI, and clinical targets.")