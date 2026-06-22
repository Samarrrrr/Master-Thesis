"""
============================================================================
 retrieve_rerank.py  --  Stage 2 of the pipeline: BM25 + cross-encoder rerank
============================================================================

WHAT THIS DOES
--------------
Takes ONE system's queries and produces ONE TREC run file, using the FIXED
retrieval pipeline that is byte-for-byte identical for all 8 systems:

    queries  ->  BM25 batch_search (Pyserini, k1=0.9, b=0.4), top-100 per query
             ->  cross-encoder rerank (ms-marco-MiniLM-L-6-v2) reorders the 100
             ->  TREC run file:  qid Q0 docid rank score tag   (100 deep)

WHY THE PIPELINE IS FIXED
-------------------------
Only the QUERY differs between systems (it came from a different rewriter).
Index, BM25 params, reranker, and depth are constant. So any difference in the
documents retrieved -- and therefore in the holes a system creates -- traces
solely to the rewriter. That isolation is the experimental design.

WHERE EACH SYSTEM'S QUERY COMES FROM (set by --query_field):
  raw    : raw_utterance   field of queries_<ds>.jsonl
  human  : manual_rewrite  field of queries_<ds>.jsonl
  others : rewrite         field of rewrites_<ds>_<model>.jsonl

NOTE ON PER-YEAR HANDLING: this script is year-agnostic on purpose. It retrieves
with whatever query text it is given. The per-year t5 context handling
(CAsT-19 = question only; CAsT-20 = prepend canonical response) lives UPSTREAM
in rewrite_t5.py, not here.

CONFIG MATCHES MQ4CS: k1=0.9, b=0.4, top-100, ms-marco-MiniLM-L-6-v2, max_len 512.
Validated: t5/human runs reproduce the published nDCG/Recall within a few points
(small gap attributable to different rewriter models than MQ4CS).
============================================================================
"""

import argparse
import json
import torch
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder

# Expected judged-turn counts per dataset, used as a silent-drop guard.
EXPECTED_TURNS = {"cast2019": 173, "cast2020": 208}


def load_queries(path, query_field):
    """
    Read a jsonl of turns -> ([(qid, query_string)], skipped_count).

    `query_field` selects which text is the query for THIS system:
    raw_utterance / manual_rewrite / rewrite. A turn whose chosen field is
    empty or missing is skipped and counted (so a silent drop can't hide).
    """
    out, skipped = [], 0
    with open(path) as fh:
        for line in fh:
            obj = json.loads(line)
            q = obj.get(query_field)
            if not q or not str(q).strip():
                skipped += 1
                continue
            out.append((obj["query_id"], str(q).strip()))
    return out, skipped


def extract_text(raw):
    """
    Docs are stored in the index as json {"id":..., "contents":...}.
    Pull out `contents` (the passage text the reranker scores). Falls back to
    the raw string if it is not json, and to "" if there is nothing.
    """
    try:
        return json.loads(raw)["contents"]
    except Exception:
        return raw if raw else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="query/rewrite jsonl for this system")
    ap.add_argument("--query_field", required=True,
                    choices=["raw_utterance", "manual_rewrite", "rewrite"])
    ap.add_argument("--system", required=True, help="system name (run tag + filename)")
    ap.add_argument("--index", required=True, help="Pyserini index (MARCO+CAR union)")
    ap.add_argument("--out", required=True, help="output run_<dataset>_<system>.trec")
    ap.add_argument("--dataset", default=None, choices=[None, "cast2019", "cast2020"],
                    help="optional: enables the expected-turn-count check")
    ap.add_argument("--k1", type=float, default=0.9)
    ap.add_argument("--b", type=float, default=0.4)
    ap.add_argument("--retrieve_depth", type=int, default=100,
                    help="BM25 candidates retrieved + reranked per query (top-100)")
    ap.add_argument("--threads", type=int, default=16, help="BM25 batch_search threads")
    ap.add_argument("--reranker", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=128, help="reranker GPU batch size")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  system: {args.system}")

    # --- load this system's queries (only the query text differs between systems) ---
    queries, skipped = load_queries(args.queries, args.query_field)
    print(f"{len(queries)} queries ({skipped} skipped: empty {args.query_field})")

    # SILENT-DROP GUARD: if a dataset was named, warn loudly when the loaded
    # query count does not match the expected judged-turn count. A wrong count
    # here means turns went missing before retrieval -- exactly the kind of
    # subtle issue that otherwise hides until the final numbers look off.
    if args.dataset and len(queries) != EXPECTED_TURNS[args.dataset]:
        print(f"  WARNING: expected {EXPECTED_TURNS[args.dataset]} turns for "
              f"{args.dataset}, loaded {len(queries)}. Investigate before trusting runs.")

    qids = [qid for qid, _ in queries]
    qtexts = [q for _, q in queries]
    qid_to_text = {qid: q for qid, q in queries}   # O(1) lookup during rerank

    # --- BM25 FIRST STAGE: search every query at once (fast, multi-threaded) ---
    # batch_search returns {qid: [hit, hit, ...]} with up to retrieve_depth hits.
    searcher = LuceneSearcher(args.index)
    searcher.set_bm25(k1=args.k1, b=args.b)
    print("running BM25 batch_search...")
    results = searcher.batch_search(qtexts, qids, k=args.retrieve_depth, threads=args.threads)
    print("BM25 done; reranking...")

    # --- CROSS-ENCODER SECOND STAGE: rerank each query's 100 candidates on GPU ---
    reranker = CrossEncoder(args.reranker, max_length=args.max_len, device=device)

    n_written = 0
    n_empty_passages = 0   # guard: count passages that came back with no text
    with open(args.out, "w") as out:
        # iterate in the original query order so the run file is reproducible
        for qid in tqdm(qids, desc=args.system, unit="q"):
            hits = results.get(qid, [])
            if not hits:
                continue

            # pull docids + passage text for the 100 candidates.
            # h.raw is the stored doc (fast path); fall back to a doc lookup if absent.
            docids, passages = [], []
            for h in hits:
                raw = getattr(h, "raw", None)
                if raw is None:
                    d = searcher.doc(h.docid)
                    raw = d.raw() if d is not None else ""
                text = extract_text(raw)
                if not text:
                    n_empty_passages += 1   # empty passage -> reranker can't score it well
                docids.append(h.docid)
                passages.append(text)

            # rerank: cross-encoder scores each (query, passage) pair, sort descending.
            # stable sort => ties keep BM25 order (a sane, deterministic tie-break).
            qtext = qid_to_text[qid]
            pairs = [(qtext, p) for p in passages]
            scores = reranker.predict(pairs, batch_size=args.batch_size, show_progress_bar=False)
            ranked = sorted(zip(docids, scores), key=lambda x: -float(x[1]))

            # write standard 6-column TREC run lines: qid Q0 docid rank score tag
            for rank, (docid, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {docid} {rank} {float(score):.6f} {args.system}\n")
                n_written += 1
            out.flush()   # progressive write so progress is visible on disk

    # EMPTY-PASSAGE GUARD: if many passages had no text, the reranker was scoring
    # blanks and the ranking is suspect. Surface it instead of letting it hide.
    if n_empty_passages:
        print(f"  WARNING: {n_empty_passages} candidate passages had empty text "
              f"(reranker scored blanks for these). Check index 'raw' storage.")

    print(f"done -> {args.out}  ({n_written} lines)")


if __name__ == "__main__":
    main()
