-- Feature Engineering: Normalizing imaging volumes by Total Intracranial Volume (eTIV)
-- and joining multi-modal features with clinical conversion target outcomes.

SELECT
    g.subject_id,
    g.age,
    g.sex,
    g.apoe_e4_count,

    -- Absolute MRI measurements
    m.eTIV_cm3,
    m.hippocampus_vol_cm3,
    m.ventricles_vol_cm3,
    m.entorhinal_thickness_mm,

    -- Normalized Volumetric Metrics (Volume / eTIV ratio)
    ROUND((m.hippocampus_vol_cm3 / m.eTIV_cm3) * 100, 4) AS norm_hippocampus_ratio,
    ROUND((m.ventricles_vol_cm3 / m.eTIV_cm3) * 100, 4) AS norm_ventricles_ratio,

    -- Clinical Outcome Target (0 = Resilient, 1 = High-Risk Converter)
    c.conversion_status

FROM genomic_biomarkers g
JOIN mri_volumetrics m ON g.subject_id = m.subject_id
JOIN clinical_outcomes c ON g.subject_id = c.subject_id;
