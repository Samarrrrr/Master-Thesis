"""
rewrite_quretec.py -- Stage 2 (traditional control arm): QuReTeC rewriter.

Produces QuReTeC query rewrites for CAsT-2019 and CAsT-2020 in the same JSONL
format as the other systems, so QuReTeC slots in as one of the eight.

QuReTeC (Voskarides et al., 2020) is a non-LLM, BERT-based term-classification
resolver: it appends selected terms from the conversation history to the current
utterance. Following the reference work, the rewrites are not recomputed here but
taken from the released precomputed files (2019 uses question context, 2020
question-and-answer context), so this arm matches the literature.

Source files (columns: conversation_id, turn_id, id, query, original):
    2019: rewrites/2019/5_QuReTeC_Q.tsv
    2020: rewrites/2020/5_QuReTeC_QnA.tsv

Output: rewrites_<dataset>_quretec.jsonl
        each line: {query_id, dataset, model, depth, raw_utterance, rewrite}
"""

import argparse
import json
import os
import urllib.request


QURETEC_URLS = {
    "cast2019": "https://raw.githubusercontent.com/svakulenk0/cast_evaluation/main/rewrites/2019/5_QuReTeC_Q.tsv",
    "cast2020": "https://raw.githubusercontent.com/svakulenk0/cast_evaluation/main/rewrites/2020/5_QuReTeC_QnA.tsv",
}


def load_tsv_rewrites(path):
    """
    Read Vakulenko's QuReTeC tsv into {qid: rewrite}.

    Columns: conversation_id  turn_id  id  query  original
      id    = the query id we use elsewhere ("31_2")
      query = the QuReTeC-resolved query (terms appended to the utterance)
    We locate columns BY NAME (robust to reordering), with a positional fallback.
    """
    out = {}
    with open(path) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            id_idx = header.index("id")
            q_idx = header.index("query")
        except ValueError:
            id_idx, q_idx = 2, 3   # known fixed positions if header names differ
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) > max(id_idx, q_idx):
                out[parts[id_idx].strip()] = parts[q_idx].strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["cast2019", "cast2020"], required=True)
    ap.add_argument("--queries", required=True, help="queries_<dataset>.jsonl (for depth+raw+alignment)")
    ap.add_argument("--raw_dir", required=True, help="where to download/keep the source tsv")
    ap.add_argument("--out", required=True, help="output rewrites_<dataset>_quretec.jsonl")
    args = ap.parse_args()

    # 1. download the precomputed tsv if we don't already have it
    os.makedirs(args.raw_dir, exist_ok=True)
    tsv_path = os.path.join(args.raw_dir, f"quretec_{args.dataset}.tsv")
    if not os.path.exists(tsv_path):
        url = QURETEC_URLS[args.dataset]
        print(f"downloading QuReTeC rewrites: {url}")
        urllib.request.urlretrieve(url, tsv_path)
    rewrites = load_tsv_rewrites(tsv_path)
    print(f"loaded {len(rewrites)} QuReTeC rewrites")

    # 2. attach the matching QuReTeC rewrite to each of our turns by qid, so the
    #    output is aligned with every other system (same qids, same order).
    turns = [json.loads(l) for l in open(args.queries)]
    written, missing = 0, 0
    with open(args.out, "w") as out:
        for t in turns:
            qid = t["query_id"]
            rw = rewrites.get(qid)
            if rw is None:
                # Fallback so the system still has a query. VERIFIED to never fire
                # on our data (tsv covers all turns); kept as a guard. A large
                # `missing` would mean the qid scheme drifted -- investigate.
                missing += 1
                rw = t["raw_utterance"]
            out.write(json.dumps({
                "query_id": qid,
                "dataset": args.dataset,
                "model": "quretec",
                "depth": t["depth"],
                "raw_utterance": t["raw_utterance"],
                "rewrite": rw,
            }) + "\n")
            written += 1
    print(f"wrote {written} -> {args.out}  ({missing} qids had no QuReTeC match, used raw fallback)")
    if missing > 0:
        print("  NOTE: missing>0 means the qid scheme differs -- inspect the tsv header "
              "and our query_ids before trusting this system. (Verified 0 on current data.)")


if __name__ == "__main__":
    main()
