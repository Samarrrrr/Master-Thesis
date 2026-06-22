"""
============================================================================
 rewrite_t5.py  --  Stage 2 (traditional control arm): T5-CANARD rewriter
============================================================================

WHAT THIS DOES
--------------
Produces T5 query rewrites for CAsT-19 and CAsT-20, in the same jsonl format as
every other system, so t5 slots into the pipeline as one of the eight systems.

WHAT T5-CANARD IS
-----------------
A T5-base seq2seq model fine-tuned on CANARD to rewrite a context-dependent
question into a self-contained one (castorini/t5-base-canard -- the SAME model
Abbasiantaeb et al. and Meng et al. use). It is a "traditional" (non-LLM)
rewriter and forms half of the CONTROL ARM (with QuReTeC). Comparing the LLMs
against t5/QuReTeC at matched effectiveness is what isolates a genuinely
LLM-specific reusability effect from a mere effectiveness effect.

CONTEXT (verified): QUESTIONS-ONLY, same as every other rewriter
----------------------------------------------------------------
build_input joins the `context` field (previous RAW UTTERANCES only -- see
build_queries.py) with the current utterance, CANARD-style with " ||| ". This is
the same questions-only context the LLM arm receives, so all automatic rewriters
see identical context. This is the deliberate single-varying-factor design: only
the rewriter differs, not the context it is given. (It differs from QPP4CS, which
feeds CAsT-20 t5 the canonical responses; we standardise instead -- documented.)

DETERMINISM: greedy decoding (num_beams=1), matching the temp-0 LLM arm. A small
deviation from any beam-search setup in the reference papers; stated in methods.

OUTPUT: rewrites_<dataset>_t5.jsonl
  each line: {query_id, dataset, model, depth, raw_utterance, rewrite}
============================================================================
"""

import argparse
import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

MODEL_NAME = "castorini/t5-base-canard"
SEP = " ||| "   # CANARD-style turn separator


def build_input(raw_utterance: str, context: list) -> str:
    # Join previous raw utterances + the current one with the CANARD separator.
    # context holds questions only (from build_queries.py). Turn 1 -> just the
    # current utterance (empty context).
    parts = list(context) + [raw_utterance]
    return SEP.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True, help="queries_<dataset>.jsonl from Stage 1")
    ap.add_argument("--out", required=True, help="output rewrites_<dataset>_t5.jsonl")
    ap.add_argument("--max_input", type=int, default=512)
    ap.add_argument("--max_output", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # load model + tokenizer (t5-base ~220M params; a small GPU is plenty)
    tok = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()

    turns = [json.loads(l) for l in open(args.queries)]
    print(f"{len(turns)} turns to rewrite with T5-CANARD")

    with open(args.out, "w") as out:
        for t in tqdm(turns, desc="t5", unit="turn"):
            # build the CANARD input (history ||| current question)
            text = build_input(t["raw_utterance"], t["context"])
            enc = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_input).to(device)
            with torch.no_grad():
                # greedy (deterministic) decode -> the resolved query
                gen = model.generate(**enc, max_length=args.max_output, num_beams=1)
            rewrite = tok.decode(gen[0], skip_special_tokens=True).strip()
            out.write(json.dumps({
                "query_id": t["query_id"],
                "dataset": t["dataset"],
                "model": "t5",
                "depth": t["depth"],
                "raw_utterance": t["raw_utterance"],
                "rewrite": rewrite,
            }) + "\n")
    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
