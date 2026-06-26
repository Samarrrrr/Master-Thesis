"""
Count the UNION of holes under both pools (mixed = official + my other systems
leave-one-out; official-only = official runs only), at depth 10 / k 10.
Writes the full union (qid,docid,systems) so the labelling script can consume it.
Read-only on the data; writes union_holes_<ds>.csv to results.
"""
import glob, os, csv, sys
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/thesis-final/analysis")); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec

# build the official-only pool once per ds; mixed pool reuses it + sibling runs
for ds in ("cast2019", "cast2020"):
    official = {}
    for p in glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*")):
        if p.endswith(".gz") or p.endswith(".html"): continue
        try: official[os.path.basename(p).replace("input.", "")] = io_trec.load_run(p)
        except Exception: pass
    mine = {s: io_trec.load_run(os.path.join(c.RUNS_DIR, f"run_{ds}_{s}.trec"))
            for s in c.ALL_SYSTEMS}

    off_pool = defaultdict(set)
    for r in official.values():
        for qid, ranked in r.items():
            for d, _ in ranked[:c.POOLING_DEPTH]: off_pool[qid].add(d)

    # precompute the "all my systems to depth" coverage so mixed pool is fast
    mine_cov = defaultdict(lambda: defaultdict(set))  # system -> qid -> docs(depth)
    for s, run in mine.items():
        for qid, ranked in run.items():
            for d, _ in ranked[:c.POOLING_DEPTH]:
                mine_cov[s][qid].add(d)

    union = {}  # (qid,docid) -> set(systems for which it's a hole under either pool)
    for s, run in mine.items():
        for qid, ranked in run.items():
            # mixed pool for s = official ∪ (all my systems except s), at this qid
            mixed_q = set(off_pool.get(qid, set()))
            for s2 in mine:
                if s2 != s:
                    mixed_q |= mine_cov[s2].get(qid, set())
            offq = off_pool.get(qid, set())
            for d, _ in ranked[:c.K]:
                if d not in offq or d not in mixed_q:   # hole under either pool
                    union.setdefault((qid, d), set()).add(s)

    labelled = set()
    with open(c.HOLE_LABELS[ds]) as fh:
        for row in csv.DictReader(fh): labelled.add((row["qid"], row["docid"]))
    already = sum(1 for k in union if k in labelled)
    need = sum(1 for k in union if k not in labelled)

    print(f"\n=== {ds} ===")
    print(f"  unique union holes (mixed OR official-only): {len(union)}")
    print(f"  already labelled (old floating gpt-4o):       {already}")
    print(f"  need fresh label:                             {need}")

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    out = os.path.join(c.RESULTS_DIR, f"union_holes_{ds}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["qid", "docid", "systems"])
        for (qid, d), syss in sorted(union.items()):
            w.writerow([qid, d, "|".join(sorted(syss))])
    print(f"  -> wrote {len(union)} union holes to {out}")
