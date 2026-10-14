"""Run ESA-ADB's official non-deep baselines through the official TimeEval + Docker harness,
on the lightweight subsets. This is how we confirmed our setup matches the paper:
Windowed iForest on Mission 2 scores 0.9487 event-wise F0.5 (published: 0.949).

Same algorithms and parameters as ESA's mission{1,2}_experiments.py, restricted to the
lightweight channels and the main split, one container at a time with CPU/memory caps.

    python preprocessing/official_baselines/run_official_baselines.py --mission M2 --algos subsequence_if
Needs: built images (build_official_images.sh), trimmed CSVs (make_light_csvs.py), and the
GPU patch applied to ESA-ADB's TimeEval (timeeval-docker-nogpu.patch) on machines without a GPU.
Output: results/official_baselines_<mission>.csv (TimeEval's result table); the raw
anomaly scores land in ESA-ADB/results/<timestamp>/.
"""
import os, sys, argparse
from pathlib import Path
import pandas as pd
from timeeval import TimeEval, DatasetManager, ResourceConstraints
from timeeval.metrics import ESAScores
from timeeval.params import FullParameterGrid
from timeeval_experiments.algorithms import pcc, hbos, iforest, subsequence_if, knn
from durations import Duration

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from config import ESA, MISSIONS, RESULTS                         # noqa: E402

DATA_LIGHT = f"{ESA}/data_light"            # from make_light_csvs.py
GB = 1024 ** 3


def build_algos(subset, only=None):
    """TimeEval algorithm objects with ESA's parameters. subset = channels to use."""
    sk = dict(skip_pull=True)               # use the locally built images
    algos = {
        "subsequence_if": subsequence_if(params=FullParameterGrid(
            {"target_channels": [subset], "n_trees": [200], "window_size": [17]}), **sk),
        "iforest": iforest(params=FullParameterGrid({"target_channels": [subset]}), **sk),
        "hbos": hbos(params=FullParameterGrid({"target_channels": [subset], "n_bins": [50]}), **sk),
        "pcc": pcc(params=FullParameterGrid({"target_channels": [subset]}), **sk),
        "knn": knn(params=FullParameterGrid({"target_channels": [subset]}), **sk),
    }
    return [algos[a] for a in only] if only else list(algos.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", default="M1", choices=["M1", "M2"])
    ap.add_argument("--algos", nargs="*", default=None, help="subset of algorithm names; default all")
    ap.add_argument("--cpu", type=float, default=4.0, help="CPU cap per container")
    ap.add_argument("--mem_gb", type=float, default=6.0, help="memory cap per container")
    args = ap.parse_args()
    m = MISSIONS[args.mission]
    collection, split = m["coll"].replace("-semi-supervised", ""), m["test"].replace(".test.csv", "")

    dm = DatasetManager(DATA_LIGHT)                                   # reads datasets.csv
    datasets = dm.select(collection=collection, dataset=split)
    limits = ResourceConstraints(tasks_per_host=1, task_cpu_limit=args.cpu,
                                 task_memory_limit=int(args.mem_gb * GB),
                                 train_timeout=Duration("1h"), execute_timeout=Duration("1h"))
    # ESA's two label selections: anomalies + rare events (headline), anomalies only
    metrics = [ESAScores(betas=0.5, select_labels={"Category": ["Rare Event", "Anomaly"]}),
               ESAScores(betas=0.5, select_labels={"Category": ["Anomaly"]})]
    # TimeEval API: trains + executes every algorithm in Docker, then scores with `metrics`
    te = TimeEval(dm, datasets, build_algos(m["chs"], args.algos),
                  metrics=metrics, ranking_metrics=[], resource_constraints=limits,
                  labels_csv_path=Path(f"{m['raw']}/labels.csv"),
                  test_dataset_path=Path(f"{DATA_LIGHT}/multivariate/{m['coll']}/{m['test']}"))
    te.run()
    res = te.get_results(aggregated=False)                            # one row per algorithm
    pd.set_option("display.max_columns", None, "display.width", 250)
    print(res, flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(f"{RESULTS}/official_baselines_{args.mission}.csv")


if __name__ == "__main__":
    main()
