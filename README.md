# Reusability of Conversational Search Test Collections for LLM-Based Systems

Retrieval pipeline for an MSc thesis (UvA) studying how LLM-based query
rewriters affect the reusability of the TREC CAsT 2019 and 2020 test
collections, compared to traditional rewriters.

## What this is

Eight query-resolution systems feed ONE fixed BM25 -> MiniLM retrieval pipeline.
Only the query rewriting varies, so any downstream difference traces solely to
the rewriter. The eight systems span an effectiveness ladder with a non-LLM
control arm, which lets the study separate "LLM-ness" from "effectiveness".

| System | Arm | Query source |
|---|---|---|
| raw | floor | unresolved utterance |
| human | ceiling | human gold rewrite |
| t5 | traditional control | T5-CANARD rewrite |
| quretec | traditional control | QuReTeC rewrite (precomputed, Vakulenko et al.) |
| gpt-4o, gpt-4o-mini | LLM | OpenAI rewrite |
| llama-3.1-8b-instant, llama-3.3-70b-versatile | LLM | Groq rewrite |

## Pipeline
Retrieval config follows Abbasiantaeb et al. (MQ4CS). Context is standardised to
prior user utterances across all systems and both years (single-varying-factor
design). Validated against published nDCG/Recall within a few points.

## Running

See `docs/PIPELINE.md` for exact commands. In short, for each dataset:
1. `python scripts/build_queries.py --raw <topics> --out_dir queries`
2. `python scripts/rewrite_<system>.py ...` for each rewriter
3. `python scripts/retrieve_rerank.py --queries ... --index ... --out ...`

## Requirements

`pip install -r requirements.txt`. Retrieval needs a Pyserini index of the
MS MARCO + TREC CAR collections and a GPU for the cross-encoder reranker.

## Data

Data (indexes, qrels, runs) is not in this repo. TREC CAsT topics and qrels are
available from the TREC CAsT organisers; the MS MARCO and TREC CAR collections
from their respective sources.

## Status

Retrieval pipeline, certified and documented. Analysis (missing-judgment
metrics) is maintained separately and added once finalised.
