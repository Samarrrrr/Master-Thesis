# Running the pipeline

Environment (Snellius example):
    module load 2024 && module load Java/17.0.6
    source <your_env>/bin/activate
    set -a; source .env; set +a   # OPENAI_API_KEY, GROQ_API_KEY

## Stage 1 — build per-turn query records
    python scripts/build_queries.py --raw <cast_topics_dir> --out_dir queries --dataset both

## Stage 2 — query rewrites (one per system)
    python scripts/rewrite_t5.py      --queries queries/queries_cast2019.jsonl --out rewrites/rewrites_cast2019_t5.jsonl
    python scripts/rewrite_llm.py     --queries queries/queries_cast2019.jsonl --model gpt-4o --out rewrites/rewrites_cast2019_gpt-4o.jsonl
    python scripts/rewrite_quretec.py --dataset cast2019 --queries queries/queries_cast2019.jsonl --raw_dir raw --out rewrites/rewrites_cast2019_quretec.jsonl
    # (repeat per model and per year)

## Stage 3 — retrieve + rerank (identical for all systems)
    python scripts/retrieve_rerank.py --queries <query_or_rewrite_jsonl> --query_field <raw_utterance|manual_rewrite|rewrite> \
        --system <name> --index <pyserini_index> --out runs/run_cast2019_<name>.trec --dataset cast2019

Config: BM25 k1=0.9 b=0.4, top-100; reranker ms-marco-MiniLM-L-6-v2, max_len 512.
