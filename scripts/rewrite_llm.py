"""
============================================================================
 rewrite_llm.py  --  Stage 2 (LLM arm): single-rewrite query resolver
============================================================================

WHAT THIS DOES
--------------
For ONE LLM, turn every conversational turn into a single standalone search
query. This is the ONLY place an LLM acts in the whole experiment -- it rewrites
the query; it does NOT retrieve or rank.

    input:   raw="Is it treatable?"  context=["What is throat cancer?"]
    output:  "Is throat cancer treatable?"

These rewrites become the query BM25 searches with (Stage 3). Because every
system shares the same BM25->reranker pipeline, the rewrite is the ONLY thing
that differs -- so any difference in holes/ranking traces to the rewriter.

MODELS (one script, selected by --model):
    OpenAI:  gpt-4o, gpt-4o-mini
    Groq:    llama-3.1-8b-instant, llama-3.3-70b-versatile

PROMPT (verified): MQ4CS Table 6 single-rewrite prompt, VERBATIM
----------------------------------------------------------------
Used exactly as published (Abbasiantaeb et al., EACL 2026) so our rewrites are
comparable to that paper's GPT4QR/LlamaQR baselines. The iKAT-only persona line
is omitted (CAsT has no persona). Context = previous raw utterances joined by
newlines -- the SAME questions-only context the t5 arm gets (single-varying-
factor design). Temperature 0 for determinism/reproducibility.

OUTPUT (verified clean -- no preamble/chatter, used as-is): the API returns a
bare standalone query; we .strip() it and feed it straight to retrieval. (We
spot-checked all four models' outputs -- they are clean queries, e.g.
"Is throat cancer treatable?", with no "Sure, here is..." wrapping.)

RESUMABLE: re-running skips query_ids already written, so a crash never re-pays
an API call; every line is flushed immediately. This effectively freezes the
rewrites as reproducible cached artifacts.

OUTPUT: rewrites_<dataset>_<model>.jsonl
  each line: {query_id, dataset, model, depth, raw_utterance, rewrite}
============================================================================
"""

import argparse
import json
import os
import time


# --- MQ4CS Table 6 single-rewrite prompt, verbatim (persona line omitted for CAsT) ---
PROMPT_TEMPLATE = (
    "# Instruction: I will give you a conversation between a user and a system. "
    "You should rewrite the last question of the user into a self-contained query.\n"
    "# Context: {context}\n"
    "# Please rewrite the following user question: {question}\n"
    "# Re-written query:"
)


def build_prompt(raw_utterance: str, context: list) -> str:
    # context = previous raw utterances (questions only), one per line.
    # Turn 1 -> empty context block; the model then returns the question cleaned.
    context_block = "\n".join(context) if context else ""
    return PROMPT_TEMPLATE.format(context=context_block, question=raw_utterance)


# --- two thin API clients; key read from env, never hardcoded ---
def call_openai(model: str, prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,      # deterministic
        timeout=30,         # don't let one stalled call hang the run
    )
    return resp.choices[0].message.content.strip()


def call_groq(model: str, prompt: str) -> str:
    from groq import Groq
    client = Groq()  # reads GROQ_API_KEY
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        timeout=30,
    )
    return resp.choices[0].message.content.strip()


OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini"}
GROQ_MODELS = {"llama-3.1-8b-instant", "llama-3.3-70b-versatile"}


def rewrite_one(model: str, prompt: str) -> str:
    if model in OPENAI_MODELS:
        return call_openai(model, prompt)
    elif model in GROQ_MODELS:
        return call_groq(model, prompt)
    else:
        raise ValueError(f"Unknown model: {model}")


def load_done(out_path: str) -> set:
    # resumability: query_ids already written, so we can skip them on a re-run.
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["query_id"])
                except Exception:
                    pass
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="queries_<dataset>.jsonl from Stage 1")
    ap.add_argument("--model", required=True,
                    help="gpt-4o | gpt-4o-mini | llama-3.1-8b-instant | llama-3.3-70b-versatile")
    ap.add_argument("--out", required=True, help="output rewrites_<dataset>_<model>.jsonl")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="optional pause between calls (rate-limit friendliness)")
    args = ap.parse_args()

    turns = [json.loads(l) for l in open(args.queries)]
    done = load_done(args.out)
    print(f"{len(turns)} turns; {len(done)} already done; "
          f"{len(turns) - len(done)} to do with {args.model}")

    from tqdm import tqdm
    todo = [t for t in turns if t["query_id"] not in done]
    with open(args.out, "a") as out:   # append so resuming adds to what's there
        for t in tqdm(todo, desc=args.model, unit="turn"):
            prompt = build_prompt(t["raw_utterance"], t["context"])
            rewrite = rewrite_one(args.model, prompt)
            out.write(json.dumps({
                "query_id": t["query_id"],
                "dataset": t["dataset"],
                "model": args.model,
                "depth": t["depth"],
                "raw_utterance": t["raw_utterance"],
                "rewrite": rewrite,
            }) + "\n")
            out.flush()                # crash loses nothing
            if args.sleep:
                time.sleep(args.sleep)
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
