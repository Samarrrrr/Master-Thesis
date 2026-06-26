"""
srq3_depth.py -- depth analysis (SRQ2 in the thesis): does hole creation (phi)
                 rise with conversational depth, and does adding the LLM team to
                 the pool increase it?

Computed under leave-one-team-out, in two conditions:
  (1) baseline: each official team is held out in turn and its holes counted
      against the remaining official teams -- the collection's own per-depth
      reusability.
  (2) with the LLM team: the four LLM runs are held out together as one team and
      their holes counted against the official pool.
Line (2) lying above line (1) means the LLM team adds holes, reducing
reusability.

A hole is a document in the held-out unit's top-k that no run outside it
retrieves to pooling depth. Uses plain phi only, with no relevance labels. Paths
are read from config.py.

Outputs: the per-depth figure (two panels, mean +/- SEM), the per-depth CSVs,
and the depth-collapsed totals.
"""
import os, sys, glob, csv, json
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__)); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec

# team grouping: reuse the LOTO team_of (reproduces the 28/21 team counts)
_src = open(os.path.expanduser("~/thesis26/scripts/leave_one_team_out.py")).read()
_ns = {}; exec(_src[_src.index("def team_of"):_src.index("def load_official")], _ns)
team_of = _ns["team_of"]


def load_official(ds):
    runs = {}
    for p in sorted(glob.glob(os.path.join(c.OFFICIAL_DIR[ds], "input.*"))):
        if p.endswith(".gz") or p.endswith(".html"): continue
        try: runs[os.path.basename(p).replace("input.", "")] = io_trec.load_run(p)
        except Exception: pass
    return runs

def load_llm(ds):
    return {s: io_trec.load_run(os.path.join(c.RUNS_DIR, f"run_{ds}_{s}.trec"))
            for s in c.LLM_SYSTEMS}

def load_depths(ds):
    d = {}
    with open(os.path.join(c.RUNS_DIR.replace("runs", "queries"), f"queries_{ds}.jsonl")) as fh:
        for line in fh:
            o = json.loads(line); d[o["query_id"]] = o["depth"]
    return d


def holes_held_out(held_runs, pool_run_dict, all_runs, depths, k, pooling_depth):
    """
    phi for one held-out unit. held_runs: list of run names forming the unit.
    pool_run_dict: the runs that form the pool (the held unit is excluded from it).
    Returns {depth: [phi per turn]} -- one value per qid the unit retrieved.
    """
    held = set(held_runs)
    other = defaultdict(set)
    for name, run in pool_run_dict.items():
        if name in held: continue
        for qid, ranked in run.items():
            for d, _ in ranked[:pooling_depth]:
                other[qid].add(d)
    htk = defaultdict(set)
    for name in held_runs:
        for qid, ranked in all_runs[name].items():
            for d, _ in ranked[:k]:
                htk[qid].add(d)
    bd = defaultdict(list)
    for qid, docs in htk.items():
        if qid in depths:
            bd[depths[qid]].append(len(docs - other.get(qid, set())))
    return bd


def baseline_official(official, depths, k, pd):
    """Line (1): each official team held out vs the other official teams."""
    teams = defaultdict(list)
    for n in official: teams[team_of(n)].append(n)
    agg = defaultdict(list)
    for t, members in teams.items():
        bd = holes_held_out(members, official, official, depths, k, pd)
        for dpt, vals in bd.items(): agg[dpt] += vals
    return agg, len(teams)


def llm_team(official, llm, depths, k, pd):
    """Line (2): the LLM team (4 runs) held out vs all official teams."""
    all_runs = {**official, **llm}
    # pool = official only (the LLM team is the new unit, never in its own pool)
    return holes_held_out(list(c.LLM_SYSTEMS), official, all_runs, depths, k, pd)


def summarise(bd):
    ds = sorted(bd)
    return (np.array(ds),
            np.array([np.mean(bd[d]) for d in ds]),
            np.array([np.std(bd[d]) for d in ds]),
            np.array([len(bd[d]) for d in ds]))


def run_collection(ds):
    official, llm, depths = load_official(ds), load_llm(ds), load_depths(ds)
    bd1, nteams = baseline_official(official, depths, c.K, c.POOLING_DEPTH)
    bd2 = llm_team(official, llm, depths, c.K, c.POOLING_DEPTH)

    d1, m1, s1, n1 = summarise(bd1)
    d2, m2, s2, n2 = summarise(bd2)

    # --- SANITY ------------------------------------------------------------
    print(f"\n=== {ds} SANITY ===")
    print(f"  official teams: {nteams}  (expect 28 / 21)")
    print(f"  LLM-team total phi held out: {int(sum(v for vv in bd2.values() for v in vv))} (>0)")
    shared = sorted(set(d1) & set(d2))
    above = [d for d in shared if dict(zip(d2, m2))[d] >= dict(zip(d1, m1))[d]]
    print(f"  depths where LLM line >= baseline: {len(above)}/{len(shared)} "
          f"(expect most; tail may dip on few turns)")
    thin = [int(d) for d, n in zip(d2, n2) if n < 10]
    print(f"  thin LLM depths (<10 turns, read cautiously): {thin}")

    os.makedirs(c.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(c.RESULTS_DIR, f"srq3_phi_by_depth_{ds}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["depth", "baseline_mean", "baseline_std", "baseline_n",
                    "llm_mean", "llm_std", "llm_n"])
        for d in sorted(set(d1) | set(d2)):
            i1 = dict(zip(d1, range(len(d1)))).get(d)
            i2 = dict(zip(d2, range(len(d2)))).get(d)
            w.writerow([d,
                        m1[i1] if i1 is not None else "", s1[i1] if i1 is not None else "",
                        n1[i1] if i1 is not None else "",
                        m2[i2] if i2 is not None else "", s2[i2] if i2 is not None else "",
                        n2[i2] if i2 is not None else ""])
    return {"ds": ds, "d1": d1, "m1": m1, "s1": s1, "d2": d2, "m2": m2, "s2": s2,
            "bd1": bd1, "bd2": bd2}


def make_figure(results):
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for ax, r in zip(axes, results):
        ax.plot(r["d1"], r["m1"], "-o", color="#1f77b4", zorder=3,
                label="(1) official teams (baseline)")
        ax.fill_between(r["d1"], r["m1"]-r["s1"], r["m1"]+r["s1"], color="#1f77b4", alpha=0.15)
        ax.plot(r["d2"], r["m2"], "-s", color="#d62728", zorder=4,
                label="(2) + LLM team (new runs)")
        ax.fill_between(r["d2"], r["m2"]-r["s2"], r["m2"]+r["s2"], color="#d62728", alpha=0.15)
        ax.set_title(f"{r['ds'].upper()}: hole creation (\u03c6) vs conversation depth")
        ax.set_xlabel("conversation depth (turn)"); ax.set_ylabel("\u03c6 (holes per turn)")
        ax.legend(frameon=False); ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(c.FIGURES_DIR, exist_ok=True)
    out = os.path.join(c.FIGURES_DIR, "srq3_depth_phi.pdf")
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"\n-> figure: {out}")


def make_table5(results):
    out = os.path.join(c.RESULTS_DIR, "table5_total_phi.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "line", "phi_per_turn_mean", "phi_per_turn_std", "n"])
        for r in results:
            for label, bd in (("official_baseline", r["bd1"]), ("llm_team", r["bd2"])):
                v = [x for vv in bd.values() for x in vv]
                w.writerow([r["ds"], label, round(np.mean(v), 3), round(np.std(v), 3), len(v)])
    print(f"-> Table 5: {out}\n\n=== Table 5 (phi per turn, depth-collapsed) ===")
    for r in results:
        v1 = [x for vv in r["bd1"].values() for x in vv]
        v2 = [x for vv in r["bd2"].values() for x in vv]
        print(f"  {r['ds']}: baseline={np.mean(v1):.2f}  LLM-team={np.mean(v2):.2f}  "
              f"increase={100*(np.mean(v2)-np.mean(v1))/np.mean(v1):+.1f}%")


if __name__ == "__main__":
    results = [run_collection(ds) for ds in ("cast2019", "cast2020")]
    make_figure(results)
    make_table5(results)
    print("\nDONE. Check SANITY lines above.")
