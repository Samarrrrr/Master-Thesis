"""
eda_plots.py  --  exploratory data analysis figures the guidelines ask for.

Produces four EDA plots per dataset (saved to results/figures/):
  1. relevance grade distribution (qrels)               eda_grades_<ds>.png
  2. conversation depth distribution (judged turns)     eda_depth_<ds>.png
  3. judged vs unjudged at the studied systems' top-10  eda_judged_<ds>.png
  4. corpus composition of relevant docs (MARCO vs CAR) eda_corpus_<ds>.png
Also prints the basic statistics behind each, for the Data subsection text.

Corpus split is inferred from docid prefixes: CAR ids start with non-digit
(e.g. they contain letters / 'CAR_'), MARCO ids are integer-like. Adjust the
is_car() heuristic if your ids differ.
"""
import argparse, csv, glob, json, os, sys
from collections import Counter, defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import io_trec

def is_car(docid):
    # CAsT docids are prefixed: CAR_xxx (TREC CAR) and MARCO_xxx (MS MARCO).
    return docid.startswith("CAR_")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",required=True)
    ap.add_argument("--qrels",required=True)
    ap.add_argument("--queries",required=True)
    ap.add_argument("--runs_dir",required=True)
    ap.add_argument("--outdir",default="results/figures")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    qrels=io_trec.load_qrels(args.qrels)

    # 1. relevance grade distribution
    grades=Counter()
    for q,docs in qrels.items():
        for d,g in docs.items(): grades[g]+=1
    fig,ax=plt.subplots(figsize=(5,3.2))
    gs=sorted(grades); ax.bar([str(g) for g in gs],[grades[g] for g in gs],color="#4878a8")
    ax.set_xlabel("relevance grade"); ax.set_ylabel("# judgments")
    ax.set_title(f"{args.dataset}: relevance grade distribution")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/eda_grades_{args.dataset}.png",dpi=150)
    print(f"grades {args.dataset}: "+", ".join(f"{g}:{grades[g]}" for g in gs))

    # 2. conversation depth distribution
    depths={}
    for line in open(args.queries):
        o=json.loads(line); depths[o["query_id"]]=o["depth"]
    djudged=Counter(depths[q] for q in qrels if q in depths)
    fig,ax=plt.subplots(figsize=(5,3.2))
    ds=sorted(djudged); ax.bar([str(d) for d in ds],[djudged[d] for d in ds],color="#d09a3c")
    ax.set_xlabel("conversation depth (turn)"); ax.set_ylabel("# judged turns")
    ax.set_title(f"{args.dataset}: judged turns by depth")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/eda_depth_{args.dataset}.png",dpi=150)
    print(f"depth {args.dataset}: "+", ".join(f"{d}:{djudged[d]}" for d in ds))

    # 3. judged vs unjudged at our systems' top-10
    judged_n,unjudged_n=0,0
    for p in sorted(glob.glob(os.path.join(args.runs_dir,f"run_{args.dataset}_*.trec"))):
        run=io_trec.load_run(p)
        for qid,ranked in run.items():
            if qid not in qrels: continue
            jd=set(qrels[qid].keys())
            for d,_ in ranked[:10]:
                if d in jd: judged_n+=1
                else: unjudged_n+=1
    fig,ax=plt.subplots(figsize=(4,3.2))
    ax.bar(["judged","unjudged"],[judged_n,unjudged_n],color=["#4878a8","#a83232"])
    ax.set_ylabel("# top-10 documents (all systems)")
    ax.set_title(f"{args.dataset}: judged vs unjudged @10")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/eda_judged_{args.dataset}.png",dpi=150)
    print(f"judged/unjudged@10 {args.dataset}: {judged_n} / {unjudged_n} "
          f"({100*unjudged_n/(judged_n+unjudged_n):.1f}% unjudged)")

    # 4. corpus composition of RELEVANT docs (MARCO vs CAR)
    car,marco=0,0
    for q,docs in qrels.items():
        for d,g in docs.items():
            if g>=2:
                if is_car(d): car+=1
                else: marco+=1
    fig,ax=plt.subplots(figsize=(4,3.2))
    ax.bar(["MS MARCO","TREC CAR"],[marco,car],color=["#4878a8","#5a9e5a"])
    ax.set_ylabel("# relevant documents")
    ax.set_title(f"{args.dataset}: relevant-doc corpus split")
    fig.tight_layout(); fig.savefig(f"{args.outdir}/eda_corpus_{args.dataset}.png",dpi=150)
    print(f"relevant corpus split {args.dataset}: MARCO {marco}, CAR {car} "
          f"({100*car/(car+marco):.1f}% CAR)")
    print(f"-> 4 EDA figures in {args.outdir}/")

if __name__=="__main__":
    main()
