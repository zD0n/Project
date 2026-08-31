"""Recover per-model confusion matrices from Run8 job logs.

Run8 writes confusion_matrix.csv unqualified, so every model in a sweep
overwrites the previous one and only the last survives on disk. The job log
still prints all of them, so parse them back out into one file per model:

    <RESULT_DIR>/confusion_matrix_<model>.csv

Usage:
    python parse_run_logs.py logs/*.log [--root .] [--dry-run]
"""
import argparse
import csv
import glob
import os
import re
import sys

# srun prefixes every line with the task id ("0: "); strip it if present.
PREFIX = re.compile(r"^\s*\d+:\s?")
HEADER = re.compile(r"==\s*Run8\.py on (\S+)\s*\|(.*?)==")
SAVED = re.compile(r"Saved to (.+?)\s*\(results appended")
CM_START = "Confusion Matrix (test):"
INDEX_ROW = "True \\ Pred"


def clean(line):
    return PREFIX.sub("", line.rstrip("\n"))


def parse(path):
    """Yield one dict per training run found in the log."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = [clean(l) for l in fh]

    run = None
    i = 0
    while i < len(lines):
        line = lines[i]

        m = HEADER.search(line)
        if m:
            # a new run block starts; anything half-parsed before it is dropped
            env = dict(re.findall(r"(\w+)=(\S+)", m.group(2)))
            run = {"dataset": m.group(1).lower(),
                   "frontend": env.get("FRONTEND", "").lower(),
                   "model": env.get("MODEL", "").lower(),
                   "labels": None, "matrix": None, "dir": None,
                   "log": os.path.basename(path)}

        elif run is not None and line.strip() == CM_START:
            # header row of class names, then the index-name row, then the data
            labels = lines[i + 1].split()
            rows, j = [], i + 2
            if lines[j].strip().startswith(INDEX_ROW):
                j += 1
            while j < len(lines) and len(rows) < len(labels):
                parts = lines[j].split()
                if len(parts) == len(labels) + 1:
                    rows.append((parts[0], [int(v) for v in parts[1:]]))
                    j += 1
                else:
                    break
            if len(rows) == len(labels):
                run["labels"] = labels
                run["matrix"] = rows
            i = j
            continue

        elif run is not None and SAVED.search(line):
            run["dir"] = SAVED.search(line).group(1).strip()
            if run["matrix"] and run["model"]:
                yield run
            run = None

        i += 1


def write_cm(run, root, dry_run=False):
    out_dir = os.path.join(root, run["dir"])
    if not os.path.isdir(out_dir):
        return None, "no such result dir: %s" % out_dir
    path = os.path.join(out_dir, "confusion_matrix_%s.csv" % run["model"])
    if dry_run:
        return path, "would write"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([INDEX_ROW] + run["labels"])
        for name, counts in run["matrix"]:
            w.writerow([name] + counts)
    return path, "wrote"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", help="job log files (globs allowed)")
    ap.add_argument("--root", default=os.path.dirname(os.path.abspath(__file__)),
                    help="project root the 'Saved to' paths are relative to")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    paths = []
    for pattern in args.logs:
        paths.extend(sorted(glob.glob(pattern)) or [pattern])

    found = 0
    for path in paths:
        if not os.path.isfile(path):
            print("skip (not a file):", path)
            continue
        for run in parse(path):
            found += 1
            total = sum(sum(c) for _, c in run["matrix"])
            diag = [run["matrix"][k][1][k] for k in range(len(run["matrix"]))]
            recalls = [d / sum(c) * 100 for d, (_, c) in zip(diag, run["matrix"]) if sum(c)]
            dest, action = write_cm(run, args.root, args.dry_run)
            print("%-8s %-8s %-9s  n=%-5d UAR=%.2f%%  %s %s"
                  % (run["frontend"], run["model"], run["dataset"], total,
                     sum(recalls) / len(recalls) if recalls else 0.0,
                     action, dest or ""))

    if not found:
        sys.exit("no confusion matrices found in %d log(s)" % len(paths))
    print("\n%d run(s) recovered" % found)


if __name__ == "__main__":
    main()
