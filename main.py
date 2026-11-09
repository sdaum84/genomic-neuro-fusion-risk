import sys
from pathlib import Path

# Add project root to system path
sys.path.append(str(Path(__file__).parent))

import src.generate_data
from src.data_loader import load_and_preprocess_data
from src.model_trainer import train_and_evaluate


def run_pipeline():
    print("--- Starting Genomic & Neuroimaging Pipeline ---")

    # 1. Run SQL Feature Engineering & Join
    print("\n[Step 1/2] Loading & joining multi-modal data...")
    df = load_and_preprocess_data()

    # 2. Train Models & Save Performance Artifacts
    print("\n[Step 2/2] Training model and evaluating...")
    train_and_evaluate()

    print("\n--- Pipeline Completed Successfully ---")


if __name__ == "__main__":
    run_pipeline()

