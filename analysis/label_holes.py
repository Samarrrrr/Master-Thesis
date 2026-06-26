"""
label_holes.py -- uniform LLM relevance labelling of the union hole set.

Labels every hole in union_holes_<ds>.csv with a pinned gpt-4o snapshot, so that
phi+ uses one consistent assessor across all holes and both pools. The procedure
matches the validated assessor: same system prompt, one-shot canonical-positive
construction, passage fetch, temperature 0, and 0-4 scale. Runs are resumable:
already-written (qid, docid) pairs are skipped on restart.

Output: hole_labels_pinned_<ds>.csv, including a model-provenance column.

Run (per dataset):
    python label_holes.py --dataset cast2019
    python label_holes.py --dataset cast2020
"""
import os, sys, csv, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__)); import config as c
sys.path.insert(0, c.IO_TREC_SRC); import io_trec
from openai import OpenAI
from pyserini.search.lucene import LuceneSearcher

MODEL = "gpt-4o-2024-08-06"          # PINNED snapshot (not floating 'gpt-4o')
INDEX = "/scratch-shared/sjamshaid/pyserini_cache/indexes/index-cast2019.36e604d7f5a4e08ade54e446be2f6345"

# ---- exact prompt construction from judge_holes.py (do not alter) ----------
SYSTEM_PROMPT = (
    "You are a relevance assessor for a conversational search task. "
    "Given the user's information need (a resolved, self-contained query), the "
    "conversation context, and a passage, judge how relevant the passage is to "
    "the information need. Use this graded scale:\n"
    "0 = not relevant\n1 = marginally/topically related but not useful\n"
    "2 = relevant\n3 = highly relevant\n4 = perfectly relevant\n"
    "Answer with ONLY a single integer 0-4."
)

def build_user_prompt(resolved_q, context, passage, one_shot=None):
    parts = []
    if context:
        ctx = " ".join(context) if isinstance(context, list) else str(context)
        parts.append(f"Conversation context: {ctx}")
    parts.append(f"Information need (resolved query): {resolved_q}")
    if one_shot is not None:
        parts.append("Example of a relevant passage for a similar need: "
                      f"\"{one_shot[:600]}\"")
    parts.append(f"Passage to judge: {passage}")
    parts.append("Relevance (0-4):")
    return "\n".join(parts)

def get_passage_text(searcher, docid):
    try:
        d = searcher.doc(docid)
        if d is None:
            return ""
        raw = d.raw()
        try:
            return json.loads(raw)["contents"]
        except Exception:
            return raw or ""
    except Exception:
        return ""

def judge_one(client, resolved_q, context, passage, one_shot):
    prompt = build_user_prompt(resolved_q, context, passage, one_shot)
    try:
        r = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": prompt}],
            temperature=0, timeout=30,
        )
        txt = r.choices[0].message.content.strip()
        for ch in txt:
            if ch in "01234":
                return int(ch)
        return 0
    except Exception as e:
        print(f"   judge error: {e}")
        return None

def load_queries(path):
    out = {}
    for line in open(path):
        o = json.loads(line)
        out[o["query_id"]] = {"resolved": o.get("manual_rewrite") or o.get("raw_utterance"),
                              "context": o.get("context", [])}
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["cast2019", "cast2020"])
    ap.add_argument("--sleep", type=float, default=0.0, help="optional pause between calls")
    args = ap.parse_args()
    ds = args.dataset

    client = OpenAI()
    searcher = LuceneSearcher(INDEX)
    qrels = io_trec.load_qrels(c.QRELS[ds])
    queries = load_queries(os.path.join(c.RUNS_DIR.replace("runs", "queries"),
                                        f"queries_{ds}.jsonl"))

    # one-shot positive per qid: a known grade>=3 doc from human qrels (same as validated run)
    one_shot = {}
    for qid, docs in qrels.items():
        best = [d for d, g in docs.items() if g >= 3]
        if best:
            one_shot[qid] = get_passage_text(searcher, best[0])

    # the union hole set to label
    union_path = os.path.join(c.RESULTS_DIR, f"union_holes_{ds}.csv")
    holes = list(csv.DictReader(open(union_path)))
    print(f"{ds}: {len(holes)} union holes to label with {MODEL}")

    out_path = os.path.join(c.RESULTS_DIR, f"hole_labels_pinned_{ds}.csv")
    done = set()
    if os.path.exists(out_path):                      # resume
        for r in csv.DictReader(open(out_path)):
            done.add((r["qid"], r["docid"]))
        print(f"  resuming: {len(done)} already labelled, skipping those")

    new = os.path.exists(out_path) is False
    fh = open(out_path, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=["qid", "docid", "llm_grade", "systems", "model"])
    if new:
        w.writeheader()

    t0 = time.time(); n = 0; skipped_empty = 0
    for i, row in enumerate(holes):
        qid, docid = row["qid"], row["docid"]
        if (qid, docid) in done:
            continue
        if qid not in queries:
            continue
        passage = get_passage_text(searcher, docid)
        if not passage.strip():                       # never judge an empty passage
            skipped_empty += 1
            continue
        q = queries[qid]
        g = judge_one(client, q["resolved"], q["context"], passage, one_shot.get(qid))
        if g is None:
            continue
        w.writerow({"qid": qid, "docid": docid, "llm_grade": g,
                    "systems": row.get("systems", ""), "model": MODEL})
        fh.flush()
        n += 1
        if n % 50 == 0:
            rate = n / (time.time() - t0)
            print(f"  {n} labelled ({i+1}/{len(holes)} scanned), {rate:.1f}/s, "
                  f"{skipped_empty} empty-skipped")
        if args.sleep:
            time.sleep(args.sleep)

    fh.close()
    print(f"\nDONE {ds}: {n} newly labelled, {skipped_empty} skipped (empty passage)")
    print(f"-> {out_path}")
    if skipped_empty:
        print(f"  NOTE: {skipped_empty} holes had empty passage text and were not judged. "
              f"Check these aren't a systematic gap.")

if __name__ == "__main__":
    main()
