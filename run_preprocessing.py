from modules.preprocessing import load_and_prepare, build_signals, save_signals, save_group_csvs

DATASETS = [
    ("marketing_campaign_dataset", "data/marketing_campaign_dataset.csv"),
    ("social_media_advertising", "data/Social_Media_Advertising.csv"),
]

def main():
    for name, path in DATASETS:
        print("\n" + "=" * 90)
        print(f"Building signals (Way 1) for: {name}")

        df = load_and_prepare(path)
        signals = build_signals(df, dataset_name=name)

        out_json = save_signals(signals, out_dir="outputs", base_name=name)
        out_csvs = save_group_csvs(df, out_dir="outputs", base_name=name)

        print(f"✅ Saved JSON signals to: {out_json}")
        if out_csvs:
            print("✅ Saved group summary CSVs:")
            for p in out_csvs:
                print(" -", p)

        # quick peek (so you can paste results here)
        print("\nQuick peek: overall_kpis")
        print(signals["overall_kpis"])

        print("\nQuick peek: rule_based_insights")
        for ins in signals.get("rule_based_insights", []):
            print("-", ins["message"])

if __name__ == "__main__":
    main()
