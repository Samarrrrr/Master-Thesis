# Beyond Effectiveness: The LLM Effect on Conversational Test-Collection Reusability

A controlled comparison on **TREC CAsT 2019 and 2020**.

---

## What this project investigates

Information retrieval evaluation depends on *reusable* test collections: fixed
queries, documents, and human relevance judgments against which any system can be
scored. Because judging every document is infeasible, collections are built by
**pooling** — only the top-ranked documents of the contributing systems are judged,
and everything unjudged is treated as non-relevant. A later system that retrieves a
relevant document no contributor returned hits a **hole**: it is scored as
non-relevant and the system is penalised for it.

As conversational search shifts to large language models (LLMs), it is unclear
whether LLM-based systems create more holes because they are *more effective* or
because LLM-based query resolution *diverges* from earlier systems. Prior work
leaves these explanations entangled, because it includes no non-LLM system of
comparable effectiveness.

This project addresses that with a **controlled, single-variable experiment**: eight
query-resolution systems share one fixed retrieval pipeline and differ only in how
they turn a conversational utterance into a stand-alone query. By placing traditional
(non-LLM) and LLM-based systems at overlapping effectiveness levels, the effect of
LLM-based resolution can be separated from the effect of effectiveness.

## Research questions

**RQ.** To what extent do LLM-based conversational search systems affect the
reusability of existing conversational test collections (TREC CAsT 2019 and 2020),
compared to traditional systems, as measured by missing judgments (φ, φ⁺)?

- **SRQ1.** How do missing judgments differ between LLM-based and traditional
  systems, both in the proportion of unjudged top-ranked documents (Unjudged@10) and
  in the number of relevant missing judgments (φ⁺) per turn?
- **SRQ2.** How does hole creation (φ) vary with conversational depth, and does
  adding an LLM-based contributor to the pool increase it?

## The pipeline

All eight systems share one fixed two-stage retrieval pipeline; **only the query
resolution stage varies**, so any difference in the holes a system creates is
attributable to query resolution alone.

```
conversational turn + context
        │
        ▼
  [1] QUERY REWRITER     ← the ONLY thing that varies (8 systems)
        │  one standalone query string
        ▼
  [2] BM25 (Pyserini, k1=0.9, b=0.4, top-100)   ← fixed, shared index
        │
        ▼
  [3] cross-encoder rerank (ms-marco-MiniLM-L-6-v2)  ← fixed
        │
        ▼
  [4] TREC run file  →  reusability analysis
         φ, φ⁺ (holes labelled by the pinned gpt-4o assessor), Unjudged@10
         leave-one-model-out (SRQ1) · leave-one-team-out (SRQ2)
```

- First stage: **BM25** (k1 = 0.9, b = 0.4), top 100 passages (Pyserini).
- Reranker: **ms-marco-MiniLM-L-6-v2** cross-encoder (max seq length 512).
- Relevance threshold: graded label ≥ 2 (CAsT convention).
- Pooling depth and hole cut-off **k = 10**.
- LLM relevance assessor: **gpt-4o**, pinned to snapshot **gpt-4o-2024-08-06**,
  temperature 0, one-shot prompt. Validated against human judgments before use.

## The 8 systems (query rewriters), by effectiveness band

Effectiveness deliberately varies along a graded scale and is **not** aligned with
whether a system is LLM-based — that is the core of the controlled design.

| System          | Arm          | Role                                            |
|-----------------|--------------|-------------------------------------------------|
| `raw`           | reference    | unresolved utterance — deliberately weak floor  |
| `llama-3.1-8b`  | LLM          | instruction-tuned resolver                      |
| `gpt-4o-mini`   | LLM          | instruction-tuned resolver                      |
| `llama-3.3-70b` | LLM          | instruction-tuned resolver                      |
| `t5`            | traditional  | seq2seq rewriter fine-tuned on CANARD           |
| `qretec`        | traditional  | binary term-classification resolver (QuReTeC)   |
| `gpt-4o`        | LLM          | instruction-tuned resolver (strongest LLM)      |
| `human`         | reference    | manual track rewrite — strong ceiling           |

The four LLM resolvers use a single fixed rewrite prompt (from MQ4CS) at temperature
0, so differences within the LLM arm reflect model capability, not prompt design.

## Dataset

- **TREC CAsT 2019 and 2020.** Both index the same corpus: the union of the
  **MS MARCO** passage collection and the **TREC CAR** paragraph collection
  (~38.4M passages).
- After intersecting conversational turns with relevance judgments:
  **173 judged turns (CAsT-2019)** and **208 (CAsT-2020)**.
- Official pools: **64 and 55 runs** (one CAsT-2019 run discarded for malformed
  scores), grouped into **28 and 21 TREC teams**.

> **The corpus, indexes, and run files are NOT included in this repository** (size
> and licensing). TREC CAsT topics and qrels are available from the TREC CAsT
> organisers; the MS MARCO and TREC CAR collections from their respective sources.

## Repository structure

```
.
├── scripts/                    # retrieval + query resolution (the 8 systems)
│   ├── build_queries.py        # assemble per-system query strings
│   ├── retrieve_rerank.py      # BM25 (top 100) → MiniLM cross-encoder rerank
│   ├── rewrite_t5.py           # t5 (CANARD seq2seq) resolver
│   ├── rewrite_quretec.py      # qretec (QuReTeC) resolver
│   └── rewrite_llm.py          # the four LLM resolvers (fixed MQ4CS prompt, temp 0)
├── analysis/                   # the trusted, rebuilt reusability analysis
│   ├── config.py               # paths and constants (POOLING_DEPTH, K, system lists)
│   ├── label_holes.py          # pinned gpt-4o assessor labels the union of holes
│   ├── label_topup.py          # carve-out top-up labelling (CAsT-2020 t5/qretec)
│   ├── compute_phi_plus.py     # SRQ1: per-system φ⁺ / Unjudged@10 + matched pairs (Table 3/4)
│   ├── srq3_depth.py           # SRQ2: per-depth φ under leave-one-team-out (Table 5)
│   ├── srq3_figure.py          # SRQ2: renders the depth figure from the per-depth CSVs
│   └── table4_diagnostic.py    # per-system assessor positive-rate diagnostic
├── checks/                     # sanity checks that validated the pipeline
│   ├── union_holes.py          # build the union-of-holes set per collection
│   ├── label_coverage.py       # assert every hole carries exactly one label
│   ├── find_unlabelled.py      # guard: zero unlabelled holes
│   ├── compare_old_new_labels.py
│   └── diagnose_gap.py
├── results/                    # small derived outputs (CSVs) + figures   (committed)
│   └── figures/
├── docs/
├── requirements.txt
├── .env.example                # template; copy to .env and add your key (gitignored)
└── README.md
```

## How to run

> Tested on the Snellius HPC (SLURM). Java 17 is required for Pyserini/Lucene.

```bash
# 1. Environment
module load 2024
module load Java/17.0.6
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))
python -m venv thesis_env && source thesis_env/bin/activate
pip install -r requirements.txt

# 2. Secrets — the LLM assessor needs an OpenAI key (NEVER commit this)
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...

# 3. Obtain data: a Pyserini index of MS MARCO + TREC CAR, plus CAsT topics/qrels
#    (not redistributed here — see Dataset section above)

# 4. Reproduce the analysis  (config lives in analysis/config.py)
#    Retrieval + resolution (the 8 systems' run files) — see docs/PIPELINE.md
#    for the exact per-stage commands. In short:
#      scripts/build_queries.py  →  scripts/rewrite_<system>.py  →  scripts/retrieve_rerank.py
#
#    Label the holes with the pinned gpt-4o assessor
python analysis/label_holes.py
python analysis/label_topup.py          # CAsT-2020 t5/qretec carve-out top-up
#    SRQ1: per-system φ⁺ / Unjudged@10 + matched pairs (Tables 3 and 4)
python analysis/compute_phi_plus.py
python analysis/table4_diagnostic.py
#    SRQ2: per-depth φ + the depth figure (Table 5, Figure)
python analysis/srq3_depth.py           # writes results/srq3_phi_by_depth_<ds>.csv
python analysis/srq3_figure.py          # writes results/figures/srq3_depth_phi_*.png
#    Sanity checks (optional but recommended)
python checks/label_coverage.py
python checks/find_unlabelled.py
```

See [`docs/PIPELINE.md`](docs/PIPELINE.md) for the exact retrieval and resolution
commands (per system and per collection).

## Key methodological notes

- **φ⁺ ≤ φ always holds** — relevant holes are a subset of all holes; this
  is an invariant the analysis asserts as a sanity check.
- **SRQ1 uses leave-one-model-out** against a mixed pool (official runs + the other
  seven systems). **SRQ2 uses leave-one-team-out**, with the four LLM runs grouped
  as a single team so that runs covering one another's documents do not mask the
  team's true divergence.
- **φ⁺ uses uniform LLM labels**: every hole is labelled by the pinned gpt-4o
  assessor; human and LLM labels are not mixed.
- **Depth has two meanings** that are easy to confuse: the EDA figures count *judged
  turns* per depth, while the SRQ2 depth figure counts φ over *all* turns (φ needs no
  relevance labels), so their maximum depths differ by design.
- On CAsT-2020, official runs that are themselves T5- or QuReTeC-based are excluded
  from the pool when scoring the `t5`/`qretec` systems, so the traditional arm counts
  as genuinely non-contributing (`baselineQR` and `humanQR` are retained).
