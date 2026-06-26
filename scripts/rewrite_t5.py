"""
rewrite_t5.py -- Stage 2 (traditional control arm): T5-CANARD rewriter.

Produces T5 query rewrites for CAsT-2019 and CAsT-2020 in the same JSONL format
as the other systems, so t5 slots in as one of the eight.

T5-CANARD (castorini/t5-base-canard) is a T5-base seq2seq model fine-tuned on
CANARD to rewrite a context-dependent question into a self-contained one, the
same model used by the reference work. It is a non-LLM rewriter and, with
QuReTeC, forms the traditional control arm. Context is the previous raw
utterances joined with the current utterance (CANARD-style " ||| "), matching the
context the LLM arm receives, so only the rewriter differs. Decoding is greedy
(num_beams=1), matching the temperature-0 LLM arm.

Output: rewrites_<dataset>_t5.jsonl
        each line: {query_id, dataset, model, depth, raw_utterance, rewrite}
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
