"""
srq3_figure.py -- render the depth figure (SRQ2 in the thesis) from the
per-depth CSVs written by srq3_depth.py. Kept separate from the computation so
the figure can be restyled without recomputing. Produces:
  - srq3_depth_phi_SEM.pdf/.png   main figure: SEM bands, capped at the deepest
                                  well-populated depth

Reads srq3_phi_by_depth.csv
"""
import os, sys, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(__file__)); import config as c

MIN_TURNS = 40   # cap the main figure at the deepest depth with >= this many turns

def load(ds):
    rows = list(csv.DictReader(open(os.path.join(c.RESULTS_DIR, f"srq3_phi_by_depth_{ds}.csv"))))
    def col(name):
        return np.array([float(r[name]) if r[name] != "" else np.nan for r in rows])
    return {"depth": col("depth"),
            "bm": col("baseline_mean"), "bs": col("baseline_std"), "bn": col("baseline_n"),
            "lm": col("llm_mean"), "ls": col("llm_std"), "ln": col("llm_n")}

def band(mean, std, n, kind):
    if kind == "SEM":
        spread = std / np.sqrt(np.maximum(n, 1))
    else:  # STD
        spread = std
    return np.maximum(mean - spread, 0.0), mean + spread   # floor at 0 (phi is a count)

def draw(ax, d, data, kind, cap_depth=None):
    m = np.ones_like(data["depth"], dtype=bool)
    if cap_depth is not None:
        m = data["depth"] <= cap_depth
    dep = data["depth"][m]
    blo, bhi = band(data["bm"][m], data["bs"][m], data["bn"][m], kind)
    llo, lhi = band(data["lm"][m], data["ls"][m], data["ln"][m], kind)
    ax.plot(dep, data["bm"][m], "-o", color="#1f77b4", zorder=3, label="(1) official teams (baseline)")
    ax.fill_between(dep, blo, bhi, color="#1f77b4", alpha=0.15)
    ax.plot(dep, data["lm"][m], "-s", color="#d62728", zorder=4, label="(2) + LLM team (new runs)")
    ax.fill_between(dep, llo, lhi, color="#d62728", alpha=0.15)
    ax.set_title(f"{d.upper()}: hole creation (\u03c6) vs conversation depth")
    ax.set_xlabel("conversation depth (turn)"); ax.set_ylabel("\u03c6 (holes per turn)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.3)

def cap_for(data):
    ok = data["depth"][data["bn"] >= MIN_TURNS]
    return int(ok.max()) if len(ok) else int(data["depth"].max())

def make(kind, suffix, capped):
    data = {ds: load(ds) for ds in ("cast2019", "cast2020")}
    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for ax, ds in zip(axes, ("cast2019", "cast2020")):
        cap = cap_for(data[ds]) if capped else None
        draw(ax, ds, data[ds], kind, cap)
    fig.tight_layout()
    out = os.path.join(c.FIGURES_DIR, f"srq3_depth_phi_{suffix}.pdf")
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"-> {out}")
    if capped:
        for ds in ("cast2019", "cast2020"):
            print(f"   {ds}: capped at depth {cap_for(data[ds])} (>= {MIN_TURNS} turns)")

if __name__ == "__main__":
    make("SEM", "SEM", capped=True)    # main thesis figure
    make("STD", "STD", capped=True)    # comparison
    make("SEM", "FULL", capped=False)  # appendix: all depths
    print("\nCompare SEM vs STD; SEM is recommended (shows precision of the mean).")
