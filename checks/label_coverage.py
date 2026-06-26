"""
SANITY CHECK: do existing LLM hole-labels cover the official-only-pool holes?

judge_holes.py defined a hole as 'not in qrels' (unjudged only). Under option(a)
+ official-only pool, a hole = 'top-k doc no official run retrieves to depth',
which INCLUDES judged-but-out-of-pool docs never sent to the assessor. This
counts how many official-only holes lack an LLM label, per system. Read-only.
"""
import glob, os, csv, sys
sys.path.insert(0, os.path.expanduser("~/thesis-final/analysis"))
import config as c
sys.path.insert(0, c.IO_TREC_SRC)
import io_trec

def official_only_pool(ds):
    pool, n = {}, 0
    for p in glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*")):
        if p.endswith(".gz") or p.endswith(".html"): continue
        try: r = io_trec.load_run(p)
        except Exception: continue
        n += 1
        for qid, ranked in r.items():
            pool.setdefault(qid, set()).update(d for d,_ in ranked[:c.POOLING_DEPTH])
    return pool, n

def labelled_set(ds):
    s = set()
    with open(c.HOLE_LABELS[ds]) as fh:
        for row in csv.DictReader(fh): s.add((row["qid"], row["docid"]))
    return s

for ds in ("cast2019","cast2020"):
    pool, n_off = official_only_pool(ds)
    labelled = labelled_set(ds)
    qrels = io_trec.load_qrels(c.QRELS[ds])
    missing = total = missing_judged = 0
    per_sys = {}
    for p in glob.glob(os.path.join(c.RUNS_DIR, f"run_{ds}_*.trec")):
        sysname = os.path.basename(p).replace(f"run_{ds}_","").replace(".trec","")
        r = io_trec.load_run(p); sm = st = 0
        for qid, ranked in r.items():
            for d,_ in ranked[:c.K]:
                if d not in pool.get(qid, set()):
                    st += 1; total += 1
                    if (qid, d) not in labelled:
                        sm += 1; missing += 1
                        if d in qrels.get(qid, {}):
                            missing_judged += 1
        per_sys[sysname] = (sm, st)
    pct = 100*missing/max(total,1)
    print(f"\n=== {ds}: {n_off} official runs, depth={c.POOLING_DEPTH}, k={c.K} ===")
    print(f"official-only holes (sys-hole pairs): {total}")
    print(f"MISSING label: {missing} ({pct:.1f}%)  | judged-but-out-of-pool: {missing_judged}")
    for s,(sm,st) in sorted(per_sys.items()):
        print(f"    {s:<26} {sm:>5}/{st:<6}{'  <-- gaps' if sm else ''}")
