import json
import pandas as pd


def main():
    with open("results_mc/mc_results.json") as f:
        data = json.load(f)

    rows = []
    for sc, sc_data in data["results"].items():
        for m, mdata in sc_data["methods"].items():
            row = {"scenario": sc, "method": m}
            for k, v in mdata["test_agg"].items():
                row[f"{k}_median"] = v["p50"]
                row[f"{k}_p95"] = v["p95"]
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("tables_mc_summary.csv", index=False)
    print("[OK] Exported tables_mc_summary.csv")


if __name__ == "__main__":
    main()
