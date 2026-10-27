#!/usr/bin/env python
"""Compare the timings of a DDStore method-1 and a method-2 DeepHyper search.

Each run directory (job-omnistat-deephyper.sh with COMPARE_TIMINGS=1) holds
  deephyper/results.csv     DeepHyper results (hyperparameters, objective,
                            m:trial_start / m:trial_end)
  timings/timings_*.json    one per trial (gfm_mlip_all_mpnn.py), each stage
                            as mean/min/max across the trial's ranks
  timings/search.json       search start/end times and node counts
  timings/core_timing.json  method 2 only: the DDStore core's load

Trials are compared only when their hyperparameters are identical in both
runs. Stage tables report the mean over matched trials of the rank mean and
of the rank max (the slowest rank, which every rank waits for).

  python compare-timings.py runs/m1_timing_<job> runs/m2_timing_<job> --out runs/timing_compare_<job>
"""

import argparse
import csv
import glob
import json
import os

import numpy as np

## Stages in reading order; other keys follow alphabetically
STAGE_ORDER = [
    "trial_wall",
    "startup_launch_to_python",
    "startup_imports",
    "setup_ddp",
    "load_data",
    "data_metadata",
    "data_adios_load",
    "data_ddstore_publish",
    "data_wait_ready",
    "data_reader_join",
    "create_dataloaders",
    "update_config",
    "nccl_init",
    "create_model",
    "startup_launch_to_train",
    "train_validate_test",
    "epoch0_total",
    "epoch0_train",
    "epoch0_validate",
    "epoch0_test",
    "epochs_total",
    "epochs_train",
    "epochs_validate",
    "epochs_test",
    "trace_dataload",
    "trace_dataload_sync",
    "trace_h2d",
    "trace_forward",
    "trace_forward_sync",
    "trace_backward",
    "trace_backward_sync",
    "trace_opt_step",
    "save_model",
]


def trial_key(row):
    return tuple(sorted((k, v) for k, v in row.items() if k.startswith("p:")))


def load_run(path):
    with open(os.path.join(path, "timings", "search.json")) as f:
        search = json.load(f)
    rows = list(csv.DictReader(open(os.path.join(path, "deephyper", "results.csv"))))
    timings = dict()
    for fname in glob.glob(os.path.join(path, "timings", "timings_*.json")):
        with open(fname) as f:
            payload = json.load(f)
        if payload.get("trial_id") is not None:
            timings[payload["trial_id"].split(".")[-1]] = payload
    trials = dict()
    for row in rows:
        stages = dict()
        payload = timings.get(row["job_id"])
        if payload is not None:
            stages = {
                k: (v["mean"], v["max"]) for k, v in payload["timings_seconds"].items()
            }
        wall = float(row["m:trial_end"]) - float(row["m:trial_start"])
        stages["trial_wall"] = (wall, wall)
        trials[trial_key(row)] = {
            "job_id": row["job_id"],
            "objective": row["objective"],
            "trial_end": float(row["m:trial_end"]),
            "has_timing": payload is not None,
            "stages": stages,
        }
    core = None
    core_path = os.path.join(path, "timings", "core_timing.json")
    if os.path.exists(core_path):
        with open(core_path) as f:
            core = json.load(f)
    return {"path": path, "search": search, "trials": trials, "core": core}


def ordered(keys):
    first = [k for k in STAGE_ORDER if k in keys]
    return first + sorted(k for k in keys if k not in STAGE_ORDER)


def search_summary(run, n_results):
    s = run["search"]
    ends = sorted(t["trial_end"] - s["t_search_start"] for t in run["trials"].values())
    job = s["t_end"] - s["t_start"]
    nodes = s["trial_nodes"] + s["core_nodes"]
    out = {
        "trial_nodes": s["trial_nodes"],
        "core_nodes": s["core_nodes"],
        "core_startup": s["t_search_start"] - s["t_start"],
        "search_span": s["t_search_end"] - s["t_search_start"],
        "run_span": job,
        "trials_completed": len(ends),
        "trials_per_hour": 3600.0 * len(ends) / job,
        "node_hours": nodes * job / 3600.0,
        "node_hours_per_trial": nodes * job / 3600.0 / max(len(ends), 1),
    }
    ## Time (from the search start) until the k-th result, k = n/4 .. n
    for q in [0.25, 0.5, 0.75, 1.0]:
        k = max(int(round(q * n_results)), 1)
        out[f"time_to_result_{k}"] = ends[k - 1] if len(ends) >= k else None
    return out


def fmt(x, width=12, prec=3):
    if x is None:
        return f"{'-':>{width}}"
    return f"{x:{width}.{prec}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("m1", help="run directory of the method-1 search")
    parser.add_argument("m2", help="run directory of the method-2 search")
    parser.add_argument("--out", help="prefix for <out>.csv and <out>.json")
    args = parser.parse_args()

    m1, m2 = load_run(args.m1), load_run(args.m2)
    matched = [
        k
        for k in m1["trials"]
        if k in m2["trials"]
        and m1["trials"][k]["has_timing"]
        and m2["trials"][k]["has_timing"]
    ]
    matched.sort(key=lambda k: int(m1["trials"][k]["job_id"]))

    for name, run in [("Method 1", m1), ("Method 2", m2)]:
        n = len(run["trials"])
        nt = sum(t["has_timing"] for t in run["trials"].values())
        print(f"{name}: {run['path']}")
        print(f"  trials: {n}, with timing JSON: {nt}")
    print(f"Matched trials (identical hyperparameters): {len(matched)}")
    print()

    ## Per-stage table over matched trials
    keys = set()
    for k in matched:
        keys |= set(m1["trials"][k]["stages"]) | set(m2["trials"][k]["stages"])
    stages = dict()
    header = (
        f"{'stage (s)':<28} {'m1_mean':>10} {'m2_mean':>10} "
        f"{'m1_max':>10} {'m2_max':>10} {'delta_max':>10} {'speedup':>8}"
    )
    print("Per trial, mean over matched trials (mean / max across ranks)")
    print(header)
    print("-" * len(header))
    for key in ordered(keys):
        v1 = [m1["trials"][k]["stages"].get(key) for k in matched]
        v2 = [m2["trials"][k]["stages"].get(key) for k in matched]
        v1 = [v for v in v1 if v is not None]
        v2 = [v for v in v2 if v is not None]
        a1 = np.mean(v1, axis=0) if v1 else (None, None)
        a2 = np.mean(v2, axis=0) if v2 else (None, None)
        delta = a2[1] - a1[1] if v1 and v2 else None
        speedup = a1[1] / a2[1] if v1 and v2 and a2[1] > 0 else None
        stages[key] = {
            "m1_mean": a1[0],
            "m2_mean": a2[0],
            "m1_max": a1[1],
            "m2_max": a2[1],
            "n1": len(v1),
            "n2": len(v2),
        }
        print(
            f"{key:<28} {fmt(a1[0], 10)} {fmt(a2[0], 10)} {fmt(a1[1], 10)} "
            f"{fmt(a2[1], 10)} {fmt(delta, 10)} {fmt(speedup, 8, 2)}"
        )
    print()

    ## Per-search table
    n_results = min(len(m1["trials"]), len(m2["trials"]))
    s1, s2 = search_summary(m1, n_results), search_summary(m2, n_results)
    print("Per search (node-hours count the core's nodes and its start-up)")
    print(f"{'metric':<28} {'method 1':>12} {'method 2':>12} {'m1/m2':>8}")
    print("-" * 64)
    for key in s1:
        a, b = s1[key], s2[key]
        ratio = a / b if a is not None and b else None
        print(f"{key:<28} {fmt(a)} {fmt(b)} {fmt(ratio, 8, 2)}")
    print()

    if m2["core"] is not None:
        print("DDStore core (method 2), mean / max across core ranks")
        for key, v in sorted(m2["core"]["timings_seconds"].items()):
            print(f"  {key:<26} {v['mean']:10.3f} {v['max']:10.3f}")
        print()

    if args.out:
        with open(args.out + ".csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                ["job_id_m1", "job_id_m2", "stage", "m1_mean", "m1_max", "m2_mean", "m2_max"]
            )
            for k in matched:
                t1, t2 = m1["trials"][k], m2["trials"][k]
                for key in ordered(set(t1["stages"]) | set(t2["stages"])):
                    a = t1["stages"].get(key, (None, None))
                    b = t2["stages"].get(key, (None, None))
                    w.writerow([t1["job_id"], t2["job_id"], key, a[0], a[1], b[0], b[1]])
        summary = {
            "m1": args.m1,
            "m2": args.m2,
            "matched_trials": len(matched),
            "stages": stages,
            "search": {"m1": s1, "m2": s2},
            "core": m2["core"]["timings_seconds"] if m2["core"] else None,
        }
        with open(args.out + ".json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {args.out}.csv and {args.out}.json")


if __name__ == "__main__":
    main()
