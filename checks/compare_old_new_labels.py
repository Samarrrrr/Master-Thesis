import os, sys, csv
sys.path.insert(0, os.path.expanduser("~/thesis-final/analysis")); import config as c
for ds in ("cast2019", "cast2020"):
    old = {}
    with open(c.HOLE_LABELS[ds]) as fh:
        for r in csv.DictReader(fh): old[(r["qid"], r["docid"])] = int(r["llm_grade"])
    new = {}
    with open(os.path.join(c.RESULTS_DIR, f"hole_labels_pinned_{ds}.csv")) as fh:
        for r in csv.DictReader(fh): new[(r["qid"], r["docid"])] = int(r["llm_grade"])
    shared = set(old) & set(new)
    exact = sum(1 for k in shared if old[k] == new[k])
    bin_agree = sum(1 for k in shared if (old[k] >= c.REL_THRESHOLD) == (new[k] >= c.REL_THRESHOLD))
    print(f"\n=== {ds} ===")
    print(f"  shared holes: {len(shared)}")
    print(f"  exact grade match: {exact}/{len(shared)} ({100*exact/max(len(shared),1):.1f}%)")
    print(f"  raw binary (>=2) agreement: {bin_agree}/{len(shared)} ({100*bin_agree/max(len(shared),1):.1f}%)")
    diffs = [old[k]-new[k] for k in shared if old[k] != new[k]]
    if diffs:
        from collections import Counter
        print(f"  grade diffs (old-new): {dict(sorted(Counter(diffs).items()))}")
