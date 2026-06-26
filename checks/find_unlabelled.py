"""Which cast2020 holes lack a pinned label, and are they carve-out-induced?"""
import os, sys, glob, csv
from collections import defaultdict
sys.path.insert(0, os.path.expanduser("~/thesis-final/analysis")); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec

ds = "cast2020"
official = {}
for p in glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*")):
    if p.endswith(".gz") or p.endswith(".html"): continue
    try: official[os.path.basename(p).replace("input.","")] = io_trec.load_run(p)
    except Exception: pass
mine = {s: io_trec.load_run(os.path.join(c.RUNS_DIR, f"run_{ds}_{s}.trec")) for s in c.ALL_SYSTEMS}
qrels = io_trec.load_qrels(c.QRELS[ds])
labels = set()
with open(os.path.join(c.RESULTS_DIR, f"hole_labels_pinned_{ds}.csv")) as fh:
    for r in csv.DictReader(fh): labels.add((r["qid"], r["docid"]))
jt = sorted(set(qrels) & set().union(*[set(r) for r in mine.values()]))

def carve(s): return set(c.CARVEOUT_CAST2020.get(s, []))
def pool(system, kind):
    p = defaultdict(set)
    for n, run in official.items():
        if n in carve(system): continue
        for qid, rk in run.items():
            for d,_ in rk[:c.POOLING_DEPTH]: p[qid].add(d)
    if kind=="mixed":
        for n, run in mine.items():
            if n==system: continue
            for qid, rk in run.items():
                for d,_ in rk[:c.POOLING_DEPTH]: p[qid].add(d)
    return p

missing = defaultdict(list)
for kind in ("mixed","officialonly"):
    for s in c.ALL_SYSTEMS:
        pl = pool(s, kind)
        for qid in jt:
            if qid not in mine[s]: continue
            for d,_ in mine[s][qid][:c.K]:
                if d in pl.get(qid,set()): continue
                if (qid,d) not in labels:
                    missing[(kind,s)].append((qid,d))

print("Unlabelled holes by (pool, system):")
for (kind,s), v in sorted(missing.items()):
    if v: print(f"  {kind:13} {s:26} {len(v)} missing")
# are they concentrated in t5/quretec (carve-out) ?
allmiss = set(d for v in missing.values() for d in v)
print(f"\nUnique missing (qid,docid): {len(allmiss)}")
# write them for a top-up labelling pass
out = os.path.join(c.RESULTS_DIR, "union_holes_cast2020_TOPUP.csv")
with open(out,"w",newline="") as fh:
    w=csv.writer(fh); w.writerow(["qid","docid","systems"])
    for qid,d in sorted(allmiss): w.writerow([qid,d,""])
print(f"-> wrote {len(allmiss)} to {out}")
