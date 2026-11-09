import pandas as pd
import sqlite3


def load_and_preprocess_data():
    """
    Loads raw CSV files into an in-memory SQLite database, executes the SQL
    feature engineering query, and returns the unified multi-modal DataFrame.
    """
    # 1. Load CSVs
    genomic_df = pd.read_csv("data/genomic_biomarkers.csv")
    imaging_df = pd.read_csv("data/mri_volumetrics.csv")
    clinical_df = pd.read_csv("data/clinical_outcomes.csv")

    # 2. In-memory SQLite Database connection
    conn = sqlite3.connect(":memory:")

    genomic_df.to_sql("genomic_biomarkers", conn, index=False, if_exists="replace")
    imaging_df.to_sql("mri_volumetrics", conn, index=False, if_exists="replace")
    clinical_df.to_sql("clinical_outcomes", conn, index=False, if_exists="replace")

    # 3. Read and execute the SQL feature engineering query
    with open("queries/01_genomic_imaging_join.sql", "r") as file:
        sql_query = file.read()

    fused_df = pd.read_sql_query(sql_query, conn)
    conn.close()

    # Encode categorical features (Sex: M -> 1, F -> 0)
    fused_df["sex_encoded"] = fused_df["sex"].map({"M": 1, "F": 0})

    return fused_df


if __name__ == "__main__":
    df = load_and_preprocess_data()
    print("✅ SQL Feature Engineering Complete!")
    print(f"Dataset shape: {df.shape[0]} subjects, {df.shape[1]} features.")
    print("\nFirst 5 rows of engineered features:")
    print(df[["subject_id", "apoe_e4_count", "norm_hippocampus_ratio", "conversion_status"]].head())