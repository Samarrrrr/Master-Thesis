"""
============================================================================
 rewrite_quretec.py  --  Stage 2 (traditional control arm): QuReTeC rewriter
============================================================================

WHAT THIS DOES
--------------
Produces QuReTeC query rewrites for CAsT-19 and CAsT-20 in the same jsonl format
as every other system, so QuReTeC slots in as one of the eight systems.

WHAT QuReTeC IS
---------------
QuReTeC (Voskarides et al., 2020) is a NON-LLM, BERT-based term-classification
resolver: it decides which terms from the conversation history to APPEND to the
current utterance. It is a "traditional" rewriter (pre-LLM era) and the second
half of the CONTROL ARM (with t5). Note its behaviour differs from generative
rewriters: it only appends terms when its classifier fires, so it is CONSERVATIVE
-- on some turns it adds nothing and the rewrite equals the raw utterance. That
is genuine QuReTeC behaviour (verified, see below), not a bug, and it is part of
why QuReTeC sits lower in effectiveness than the generative rewriters -- useful
spread for the control arm.

WHY WE DON'T RUN IT (we use precomputed rewrites)
-------------------------------------------------
Following QPP4CS (Meng et al.) and Abbasiantaeb et al., we do NOT run QuReTeC
ourselves. We use the PRECOMPUTED rewrites released by Vakulenko et al. -- the
same ones the reference papers use -- so this arm matches the literature exactly.
Per-year files differ: 2019 = Q (question context), 2020 = QnA (question+answer
context). We take both as-is; this is the one documented exception to our
otherwise questions-only context standardisation.

VERIFIED (against real data): the qid join is perfect -- the tsv covers all of
our turns (479/479 on 2019) and the raw-fallback below NEVER fired (0 turns).
Every QuReTeC turn got its real Vakulenko rewrite. The turns where the rewrite
equals raw are genuine QuReTeC no-ops (mostly turn-1 + conservative deeper turns),
not join failures.

SOURCE (Vakulenko et al., cast_evaluation repo):
  2019: rewrites/2019/5_QuReTeC_Q.tsv      2020: rewrites/2020/5_QuReTeC_QnA.tsv
  (tsv columns: conversation_id  turn_id  id  query  original)

OUTPUT: rewrites_<dataset>_quretec.jsonl
  each line: {query_id, dataset, model, depth, raw_utterance, rewrite}
============================================================================
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
