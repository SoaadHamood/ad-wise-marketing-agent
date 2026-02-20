import os
import pandas as pd

DATA_DIR = "data"

# Main CSV datasets
DATASETS = {
    "marketing_campaign_performance": "marketing_campaign_dataset.csv",
    "social_media_advertising": "Social_Media_Advertising.csv",
}

# Relational CSV datasets (from the 3rd Kaggle dataset)
RELATIONAL_CSV_DATASETS = {
    "ad_events": "ad_events.csv",
    "ads": "ads.csv",
    "campaigns": "campaigns.csv",
    "users": "users.csv",
}

def safe_read_csv(path: str, nrows: int = 50) -> pd.DataFrame:
    """
    Read only a small sample to avoid loading huge files.
    Tries common encodings automatically.
    """
    encodings = ["utf-8", "utf-8-sig", "latin1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, nrows=nrows, encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_err}")

def inspect_csv(name: str, filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    print("\n" + "=" * 90)
    print(f"DATASET (CSV): {name}")
    print(f"FILE: {path}")

    if not os.path.exists(path):
        print("❌ File not found. Check the filename and the ./data folder.")
        return

    df = safe_read_csv(path, nrows=50)

    print("\n✅ Columns:")
    print(list(df.columns))

    print("\n✅ Dtypes (sample-based):")
    print(df.dtypes)

    print("\n✅ Sample rows:")
    print(df.head(5))

    print("\n✅ Missing values count (sample-based):")
    print(df.isna().sum())

def main():
    print("Files detected under ./data:")
    if not os.path.exists(DATA_DIR):
        print("❌ data/ folder not found. Create it and put files inside.")
        return

    for f in os.listdir(DATA_DIR):
        print(" -", f)

    # Main datasets
    for name, filename in DATASETS.items():
        inspect_csv(name, filename)

    # Relational datasets
    for name, filename in RELATIONAL_CSV_DATASETS.items():
        inspect_csv(name, filename)

if __name__ == "__main__":
    main()