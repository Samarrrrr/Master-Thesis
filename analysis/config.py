import os
USER = os.environ.get("USER", "sjamshaid")

SCRATCH      = f"/scratch-shared/{USER}/thesis26"
RUNS_DIR     = f"{SCRATCH}/runs"
OFFICIAL_DIR = {
    "cast2019": f"/scratch-shared/{USER}/trec_runs/cast2019",
    "cast2020": f"/scratch-shared/{USER}/trec_runs/cast2020",
}
QRELS = {
    "cast2019": f"/scratch-shared/{USER}/ir_datasets/trec-cast/2019/2019qrels.txt",
    "cast2020": f"/scratch-shared/{USER}/ir_datasets/trec-cast/2020/2020qrels.txt",
}
HOLE_LABELS = {
    "cast2019": os.path.expanduser("~/thesis26/results/hole_labels_cast2019.csv"),
    "cast2020": os.path.expanduser("~/thesis26/results/hole_labels_cast2020.csv"),
}
IO_TREC_SRC = os.path.expanduser("~/thesis26/src")
RESULTS_DIR = os.path.expanduser("~/thesis-final/results")
FIGURES_DIR = os.path.expanduser("~/thesis-final/results/figures")

K             = 10
POOLING_DEPTH = 10
REL_THRESHOLD = 2

LLM_SYSTEMS = ["gpt-4o", "gpt-4o-mini", "llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
TRADITIONAL = ["t5", "quretec"]
REFERENCE   = ["raw", "human"]
ALL_SYSTEMS = REFERENCE + TRADITIONAL + LLM_SYSTEMS

CARVEOUT_CAST2020 = {
    "t5":      ["AUTO_T5_RRF","HBKU_t5_1v1","HBKU_t5_1v1_mnl","HBKU_t5_1v2",
                "HBKU_t5_1v2_mnl","ielab-bm25T5QLM","T5_BERT100"],
    "quretec": ["quretecNoRerank","quretecQR"],
}
