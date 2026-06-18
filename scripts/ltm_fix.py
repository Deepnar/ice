import json

PROBES_LABELED = "data/labeled/probes_labeled.jsonl"
PROBES_FIXED   = "data/labeled/probes_labeled_ltm.jsonl"

with open(PROBES_LABELED, "r") as fin, open(PROBES_FIXED, "w") as fout:
    for line in fin:
        item = json.loads(line)
        item["label"]["context_reliance"] = "Long_Term_Memory"
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
print("Fixed.")