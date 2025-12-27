import json
import glob


def main():
    files = glob.glob("results_raw/**/*.json", recursive=True)
    assert files, "No result files found"

    for f in files:
        with open(f) as h:
            d = json.load(h)
        assert "metrics" in d, f"Missing metrics in {f}"
        assert "RMSE" in d["metrics"], f"Missing RMSE in {f}"

    print(f"[OK] Verified {len(files)} result files")


if __name__ == "__main__":
    main()
