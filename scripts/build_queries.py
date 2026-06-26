"""
build_queries.py -- Stage 1: assemble per-turn query records.

Converts the raw CAsT topic files into one per-turn JSONL record read by every
downstream stage. Each line describes one conversational turn:
    query_id        e.g. "31_2" (topic 31, turn 2)
    depth           turn position in the conversation (SRQ2 axis)
    raw_utterance   the user's literal turn        (the "raw" system)
    context         previous raw user utterances   (input to the rewriters)
    manual_rewrite  human-resolved query, if provided (the "human" system)

"raw" and "human" are produced directly from this file with no model; the other
six systems read raw_utterance and context and rewrite in Stage 2. Context holds
only previous user utterances (not system responses), uniformly across systems
and both collections, so the comparison varies in the rewriter alone.

Input  (under --raw): the CAsT-2019 and CAsT-2020 topic JSONs, plus the 2019
       resolved-rewrite TSV (which carries the 2019 human rewrites).
Output: queries_cast2019.jsonl, queries_cast2020.jsonl
"""

import argparse
import json
import os


def build_cast2019(raw_dir: str, out_path: str):
    # --- load the raw topics (structure + raw utterances) ---
    with open(os.path.join(raw_dir, "evaluation_topics_v1.0.json")) as fh:
        topics = json.load(fh)

    # --- load human-resolved rewrites from the SEPARATE tsv (the 2019 fix) ---
    # format per line:  "31_1 \t What is throat cancer?"  -> key by query_id.
    manual = {}
    tsv_path = os.path.join(raw_dir, "evaluation_topics_annotated_resolved_v1.0.tsv")
    if os.path.exists(tsv_path):
        with open(tsv_path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    qid, text = parts
                    manual[qid] = text
    else:
        # Loud warning: without this, the "human" ceiling system is missing for 2019.
        print(f"WARNING: {tsv_path} not found -> CAsT-2019 manual_rewrite will be null!")

    # --- walk every topic/turn, emit one record per turn ---
    n = 0
    with open(out_path, "w") as out:
        for topic in topics:
            topic_number = topic["number"]
            context = []                            # accumulates raw utterances (questions only)
            for turn in topic["turn"]:
                turn_number = turn["number"]
                qid = f"{topic_number}_{turn_number}"
                raw = turn["raw_utterance"]
                record = {
                    "dataset": "cast2019",
                    "query_id": qid,
                    "topic_number": topic_number,
                    "turn_number": turn_number,
                    "depth": turn_number,           # in CAsT, turn number == conversational depth
                    "raw_utterance": raw,
                    "context": list(context),       # COPY: previous raw utterances only
                    "manual_rewrite": manual.get(qid),   # None if absent
                }
                out.write(json.dumps(record) + "\n")
                context.append(raw)                 # this turn becomes context for the next
                n += 1
    print(f"CAsT-2019: wrote {n} turns -> {out_path}")
    have_manual = sum(1 for _ in open(out_path) if json.loads(_)["manual_rewrite"])
    print(f"           of which {have_manual} have a human rewrite (human-system coverage)")


def build_cast2020(raw_dir: str, out_path: str):
    # CAsT-2020 ships TWO aligned json files: automatic (raw) and manual (human).
    # They line up topic-by-topic and turn-by-turn, so we zip them.
    with open(os.path.join(raw_dir, "2020_automatic_evaluation_topics_v1.0.json")) as fh:
        auto = json.load(fh)
    with open(os.path.join(raw_dir, "2020_manual_evaluation_topics_v1.0.json")) as fh:
        manual_topics = json.load(fh)

    n = 0
    with open(out_path, "w") as out:
        for auto_topic, manual_topic in zip(auto, manual_topics):
            topic_number = auto_topic["number"]
            # safety: the two files must describe the same topic in the same order
            assert topic_number == manual_topic["number"], "2020 auto/manual misalignment"
            context = []
            for auto_turn, manual_turn in zip(auto_topic["turn"], manual_topic["turn"]):
                turn_number = auto_turn["number"]
                qid = f"{topic_number}_{turn_number}"
                raw = auto_turn["raw_utterance"]
                manual_rw = manual_turn["manual_rewritten_utterance"]
                record = {
                    "dataset": "cast2020",
                    "query_id": qid,
                    "topic_number": topic_number,
                    "turn_number": turn_number,
                    "depth": turn_number,
                    "raw_utterance": raw,
                    "context": list(context),       # previous raw utterances only
                    "manual_rewrite": manual_rw,
                }
                out.write(json.dumps(record) + "\n")
                context.append(raw)
                n += 1
    print(f"CAsT-2020: wrote {n} turns -> {out_path}")
    have_manual = sum(1 for _ in open(out_path) if json.loads(_)["manual_rewrite"])
    print(f"           of which {have_manual} have a human rewrite (human-system coverage)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="dir containing the CAsT topic files")
    ap.add_argument("--out_dir", required=True, help="where to write queries_*.jsonl")
    ap.add_argument("--dataset", choices=["cast2019", "cast2020", "both"], default="both")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.dataset in ("cast2019", "both"):
        build_cast2019(args.raw, os.path.join(args.out_dir, "queries_cast2019.jsonl"))
    if args.dataset in ("cast2020", "both"):
        build_cast2020(args.raw, os.path.join(args.out_dir, "queries_cast2020.jsonl"))
