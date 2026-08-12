"""Official ESA-ADB reproduction via TimeEval + Docker, LIGHTWEIGHT SUBSET only.
Mirrors mission{1,2}_experiments.py's non-deep block exactly, but:
  - subset_channels only (the lightweight subset)
  - the main split (M1=84_months, M2=21_months)
  - skip_pull=True (images built locally)
  - resource caps so it never overstresses the laptop (1 container at a time, CPU/mem limits)
"""
import os, sys, argparse
from pathlib import Path
from timeeval import TimeEval, DatasetManager, ResourceConstraints
from timeeval.metrics import ESAScores
from timeeval.params import FullParameterGrid
from timeeval_experiments.algorithms import pcc, hbos, iforest, subsequence_if, knn
from durations import Duration

REPO = "/Users/Shreyas/Downloads/Capstone/ESA/ESA-ADB"
# channel-trimmed copies (lightweight subset only) so CSVs fit the laptop's Docker VM;
# results are identical because the algorithms only ever read target_channels.
DATA_PROCESSED = "/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/data_light"
GB = 1024 ** 3

MISS = {
    "M1": dict(collection="ESA-Mission1", raw="/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission1",
               subset=[f"channel_{i}" for i in range(41, 47)], split="84_months"),
    "M2": dict(collection="ESA-Mission2", raw="/Users/Shreyas/Downloads/Capstone/ESA/ESA-Mission2",
               subset=[f"channel_{i}" for i in range(18, 29)], split="21_months"),
}


def build_algos(subset, only=None):
    sk = dict(skip_pull=True)
    algos = {
        "subsequence_if": subsequence_if(params=FullParameterGrid({"target_channels": [subset], "n_trees": [200], "window_size": [17]}), **sk),
        "iforest": iforest(params=FullParameterGrid({"target_channels": [subset]}), **sk),
        "hbos": hbos(params=FullParameterGrid({"target_channels": [subset], "n_bins": [50]}), **sk),
        "pcc": pcc(params=FullParameterGrid({"target_channels": [subset]}), **sk),
        "knn": knn(params=FullParameterGrid({"target_channels": [subset]}), **sk),
    }
    if only:
        return [algos[a] for a in only]
    return list(algos.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1", choices=["M1", "M2"])
    ap.add_argument("--algos", nargs="*", default=None, help="subset of algo names; default all")
    ap.add_argument("--cpu", type=float, default=4.0)
    ap.add_argument("--mem_gb", type=float, default=6.0)
    args = ap.parse_args()
    m = MISS[args.mission]

    dm = DatasetManager(DATA_PROCESSED)
    datasets = dm.select(collection=m["collection"], dataset=m["split"])
    print(f"mission={args.mission} collection={m['collection']} split={m['split']} "
          f"| {len(m['subset'])} subset channels | datasets={datasets}", flush=True)

    limits = ResourceConstraints(
        tasks_per_host=1,                     # one container at a time -> laptop-safe
        task_cpu_limit=args.cpu,
        task_memory_limit=int(args.mem_gb * GB),
        train_timeout=Duration("1h"),
        execute_timeout=Duration("1h"),
    )
    beta = 0.5
    metrics = [ESAScores(betas=beta, select_labels={"Category": ["Rare Event", "Anomaly"]}),
               ESAScores(betas=beta, select_labels={"Category": ["Anomaly"]})]
    labels_csv = Path(os.path.join(m["raw"], "labels.csv"))
    test_path = Path(os.path.join(DATA_PROCESSED, "multivariate",
                                  f"{m['collection']}-semi-supervised", f"{m['split']}.test.csv"))

    algorithms = build_algos(m["subset"], only=args.algos)
    te = TimeEval(dm, datasets, algorithms,
                  metrics=metrics, ranking_metrics=[],
                  resource_constraints=limits,
                  labels_csv_path=labels_csv,
                  test_dataset_path=test_path)
    te.run()
    res = te.get_results(aggregated=False)
    import pandas as pd
    pd.set_option("display.max_columns", None, "display.width", 250)
    print("\n================ RESULTS ================", flush=True)
    print(res, flush=True)
    out = f"/Users/Shreyas/Downloads/Capstone/ESA/FederatedGDN/results/official_{args.mission}.csv"
    res.to_csv(out)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
