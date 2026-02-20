from modules.relational_preprocessing import (
    load_relational_data,
    build_fact_table,
    build_relational_signals,
    save_json,
    save_group_csvs,
)

def main():
    dfs = load_relational_data(
        events_path="data/ad_events.csv",
        ads_path="data/ads.csv",
        campaigns_path="data/campaigns.csv",
        users_path="data/users.csv",
    )

    fact = build_fact_table(dfs)

    signals = build_relational_signals(fact, dataset_name="relational_ads")
    out_json = save_json(signals, "outputs/relational_ads_signals.json")
    out_csvs = save_group_csvs(fact, out_dir="outputs", base_name="relational_ads")

    print("\n" + "=" * 90)
    print("✅ Saved relational signals JSON to:", out_json)
    if out_csvs:
        print("✅ Saved relational summary CSVs:")
        for p in out_csvs:
            print(" -", p)

    print("\nQuick peek: overall_kpis")
    print(signals["overall_kpis"])

    print("\nQuick peek: rule_based_insights")
    for ins in signals.get("rule_based_insights", []):
        print("-", ins["message"])

if __name__ == "__main__":
    main()
