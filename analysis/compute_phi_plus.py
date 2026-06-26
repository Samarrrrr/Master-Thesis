"""
compute_phi_plus.py -- SRQ1: per-system missing judgments, LLM vs traditional.

For each of the eight systems, computes Unjudged@10 (relevance-agnostic) and phi+
(relevant holes, using the pinned gpt-4o labels) per turn, under leave-one-model-
out. Two pools are reported: the mixed pool (official runs plus the other seven
systems) and official-only. On CAsT-2020, T5- and QuReTeC-based official runs are
carved out of the pool when scoring the t5/quretec systems, keeping the
traditional arm non-contributing.

Also reports the matched-pair phi+ differences (LLM minus traditional of
comparable effectiveness), the headline being llama-3.3-70b vs t5, where the LLM
system is the less effective yet creates more relevant holes.

Sanity checks per run: phi+ <= phi, zero unlabelled holes, and raw ranking above
human on Unjudged@10.

Outputs: table3_phiplus_mixed_<ds>.csv, table3_phiplus_officialonly_<ds>.csv
"""
import os, sys, glob, csv
from collections import defaultdict
sys.path.insert(0, os.path.dirname(__file__)); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec

PAIRS = [
    ("llama-3.3-70b-versatile", "t5"),
    ("gpt-4o-mini",             "t5"),
    ("gpt-4o",                  "t5"),
    ("llama-3.3-70b-versatile", "quretec"),
    ("gpt-4o",                  "quretec"),
]

def load_official(ds):
    runs = {}
    for p in sorted(glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*"))):
        if p.endswith(".gz") or p.endswith(".html"): continue
        try: runs[os.path.basename(p).replace("input.", "")] = io_trec.load_run(p)
        except Exception: pass
    return runs

def load_mine(ds):
    return {s: io_trec.load_run(os.path.join(c.RUNS_DIR, f"run_{ds}_{s}.trec"))
            for s in c.ALL_SYSTEMS}

def load_pinned_labels(ds):
    lab = {}
    with open(os.path.join(c.RESULTS_DIR, f"hole_labels_pinned_{ds}.csv")) as fh:
        for r in csv.DictReader(fh): lab[(r["qid"], r["docid"])] = int(r["llm_grade"])
    return lab

def carveout_for(ds, system):
    return set(c.CARVEOUT_CAST2020.get(system, [])) if ds == "cast2020" else set()

def build_pool(ds, system, official, mine, kind):
    carve = carveout_for(ds, system)
    pool = defaultdict(set)
    for name, run in official.items():
        if name in carve: continue
        for qid, ranked in run.items():
            for d, _ in ranked[:c.POOLING_DEPTH]: pool[qid].add(d)
    if kind == "mixed":
        for name, run in mine.items():
            if name == system: continue
            for qid, ranked in run.items():
                for d, _ in ranked[:c.POOLING_DEPTH]: pool[qid].add(d)
    return pool

def arm_of(s):
    if s == "raw": return "floor"
    if s == "human": return "ceiling"
    if s in c.TRADITIONAL: return "traditional"
    return "llm"

def per_system(ds, official, mine, qrels, labels, jt, kind):
    rows = {}; unlabelled = 0
    for s in c.ALL_SYSTEMS:
        run = mine[s]; pool = build_pool(ds, s, official, mine, kind)
        phi = phiplus = 0; unj = 0.0; turns = 0
        for qid in jt:
            if qid not in run: continue
            turns += 1
            top10 = [d for d, _ in run[qid][:c.K]]
            unj += sum(1 for d in top10 if d not in qrels.get(qid, {})) / max(len(top10), 1)
            for d in top10:
                if d in pool.get(qid, set()): continue
                phi += 1
                g = labels.get((qid, d))
                if g is None: unlabelled += 1
                elif g >= c.REL_THRESHOLD: phiplus += 1
        rows[s] = {"system": s, "arm": arm_of(s),
                   "unjudged@10": round(unj / max(turns, 1), 4),
                   "phi": phi, "phi_plus_total": phiplus,
                   "phi_plus_per_turn": round(phiplus / max(len(jt), 1), 4), "turns": turns}
    return rows, unlabelled

def judged_turns_of(qrels, mine):
    retrieved = set()
    for run in mine.values(): retrieved |= set(run.keys())
    return sorted(set(qrels.keys()) & retrieved)

def main():
    for ds in ("cast2019", "cast2020"):
        official = load_official(ds); mine = load_mine(ds)
        qrels = io_trec.load_qrels(c.QRELS[ds]); labels = load_pinned_labels(ds)
        jt = judged_turns_of(qrels, mine)
        print(f"\n{'='*70}\n{ds}: {len(official)} official, {len(mine)} mine, "
              f"{len(jt)} judged turns, {len(labels)} pinned labels\n{'='*70}")
        results = {}
        for kind in ("mixed", "officialonly"):
            rows, unlab = per_system(ds, official, mine, qrels, labels, jt, kind)
            results[kind] = rows
            print(f"\n--- {ds} pool={kind} ---")
            print(f"{'system':<26}{'arm':<12}{'Unj@10':>8}{'phi':>7}{'phi+':>7}{'phi+/turn':>11}")
            for s in sorted(rows, key=lambda x: rows[x]["unjudged@10"]):
                r = rows[s]
                print(f"{r['system']:<26}{r['arm']:<12}{r['unjudged@10']:>8}{r['phi']:>7}"
                      f"{r['phi_plus_total']:>7}{r['phi_plus_per_turn']:>11}")
            bad = [s for s in rows if rows[s]["phi_plus_total"] > rows[s]["phi"]]
            print(f"  SANITY: phi+<=phi {'OK' if not bad else '!! '+str(bad)} | "
                  f"unlabelled holes {unlab} {'OK' if unlab==0 else '!! INCOMPLETE'}")
            by = sorted(rows, key=lambda x: rows[x]["unjudged@10"])
            print(f"          raw idx {by.index('raw')} > human idx {by.index('human')}: "
                  f"{'OK' if by.index('raw')>by.index('human') else 'CHECK'}")
        print(f"\n--- {ds} MIXED vs OFFICIALONLY phi+/turn ---")
        print(f"{'system':<26}{'mixed':>8}{'offonly':>9}{'ratio':>7}")
        for s in c.ALL_SYSTEMS:
            m = results["mixed"][s]["phi_plus_per_turn"]; o = results["officialonly"][s]["phi_plus_per_turn"]
            print(f"{s:<26}{m:>8}{o:>9}{(round(o/m,2) if m>0 else '-'):>7}")
        print(f"\n--- {ds} TABLE 6: matched-pair phi+/turn diff (LLM - traditional) ---")
        print(f"{'pair':<36}{'mixed':>9}{'offonly':>10}")
        for llm, trad in PAIRS:
            md = results["mixed"][llm]["phi_plus_per_turn"] - results["mixed"][trad]["phi_plus_per_turn"]
            od = results["officialonly"][llm]["phi_plus_per_turn"] - results["officialonly"][trad]["phi_plus_per_turn"]
            tag = "  <-- HEADLINE" if (llm,trad)==PAIRS[0] else ""
            print(f"{llm+' - '+trad:<36}{round(md,3):>9}{round(od,3):>10}{tag}")
        os.makedirs(c.RESULTS_DIR, exist_ok=True)
        for kind in ("mixed", "officialonly"):
            out = os.path.join(c.RESULTS_DIR, f"table3_phiplus_{kind}_{ds}.csv")
            with open(out, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["system","arm","unjudged@10","phi",
                                   "phi_plus_total","phi_plus_per_turn","turns"])
                w.writeheader()
                for s in sorted(results[kind], key=lambda x: results[kind][x]["unjudged@10"]):
                    w.writerow(results[kind][s])
            print(f"-> {out}")

if __name__ == "__main__":
    main()
