"""
Why are 6000 official-only holes unlabelled on CAsT-19 when judge_holes.py
labelled all 'not-in-qrels' top-10 holes? Break the missing set into:
  A: missing & in qrels        (judged-but-out-of-pool -> need LLM topup for opt-a)
  B: missing & NOT in qrels & retrieved by some official run beyond depth-10
  C: missing & NOT in qrels & truly never retrieved by any official run
This tells us whether the old labelling used a different k/depth or pool.
"""
import glob, os, csv, sys
sys.path.insert(0, os.path.expanduser("~/thesis-final/analysis")); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec

for ds in ("cast2019","cast2020"):
    # official pool at depth 10 AND a deeper "ever retrieved" set (all ranks)
    pool10, poolALL = {}, {}
    for p in glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*")):
        if p.endswith(".gz") or p.endswith(".html"): continue
        try: r = io_trec.load_run(p)
        except Exception: continue
        for qid, ranked in r.items():
            pool10.setdefault(qid,set()).update(d for d,_ in ranked[:c.POOLING_DEPTH])
            poolALL.setdefault(qid,set()).update(d for d,_ in ranked)
    qrels = io_trec.load_qrels(c.QRELS[ds])
    labelled=set()
    with open(c.HOLE_LABELS[ds]) as fh:
        for row in csv.DictReader(fh): labelled.add((row["qid"],row["docid"]))
    A=B=C=0
    for p in glob.glob(os.path.join(c.RUNS_DIR, f"run_{ds}_*.trec")):
        r=io_trec.load_run(p)
        for qid,ranked in r.items():
            for d,_ in ranked[:c.K]:
                if d in pool10.get(qid,set()):  continue   # not a hole
                if (qid,d) in labelled:          continue   # already labelled
                if d in qrels.get(qid,{}):       A+=1
                elif d in poolALL.get(qid,set()):B+=1       # official has it, but deeper than 10
                else:                            C+=1       # truly nobody
    print(f"\n{ds}: missing breakdown")
    print(f"  A judged-but-out-of-pool (need LLM topup):        {A}")
    print(f"  B unjudged, official has it deeper than depth-10: {B}")
    print(f"  C unjudged, truly never retrieved by official:    {C}")
    print(f"  -> C is the real surprise if large: these SHOULD have been labelled by judge_holes.py")
