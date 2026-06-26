import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__)); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher

lab = open(os.path.join(os.path.dirname(__file__), "label_holes.py")).read()
exec(lab.split("def main")[0])

ds = "cast2020"
client = OpenAI(); searcher = LuceneSearcher(INDEX)
qrels = io_trec.load_qrels(c.QRELS[ds])
queries = load_queries(os.path.join(c.RUNS_DIR.replace("runs","queries"), f"queries_{ds}.jsonl"))
one_shot = {}
for qid, docs in qrels.items():
    best = [d for d, g in docs.items() if g >= 3]
    if best: one_shot[qid] = get_passage_text(searcher, best[0])

topup_path = os.path.join(c.RESULTS_DIR, "union_holes_cast2020_TOPUP.csv")
topup = list(csv.DictReader(open(topup_path)))
out_path = os.path.join(c.RESULTS_DIR, f"hole_labels_pinned_{ds}.csv")
done = set()
for r in csv.DictReader(open(out_path)): done.add((r["qid"], r["docid"]))

fh = open(out_path, "a", newline="")
w = csv.DictWriter(fh, fieldnames=["qid","docid","llm_grade","systems","model"])
n = empty = 0
grades = []
for row in topup:
    qid, docid = row["qid"], row["docid"]
    if (qid,docid) in done or qid not in queries: continue
    p = get_passage_text(searcher, docid)
    if not p.strip(): empty += 1; continue
    g = judge_one(client, queries[qid]["resolved"], queries[qid]["context"], p, one_shot.get(qid))
    if g is None: continue
    w.writerow({"qid":qid,"docid":docid,"llm_grade":g,"systems":"","model":MODEL})
    fh.flush(); n += 1; grades.append(g)
fh.close()
from collections import Counter
print(f"top-up done: {n} labelled, {empty} empty-skipped")
print(f"grade distribution of top-up holes: {dict(sorted(Counter(grades).items()))}")
rel = sum(1 for g in grades if g >= c.REL_THRESHOLD)
print(f"  of {n}: {rel} relevant (>=2), {n-rel} not-relevant")
print(f"-> appended to {out_path}")
