import json, os, glob

CURATION_DIR = "experiments/curation_files"
OUTPUT = "data/labeled/probes_unlabeled.jsonl"

probes = []
for fpath in glob.glob(os.path.join(CURATION_DIR, "*.json")):
    with open(fpath, "r") as f:
        cdata = json.load(f)
    for probe in cdata.get("evaluation_probes", []):
        prompt = probe.get("user_injected_prompt", "").strip()
        if prompt and prompt != "ENTER_PROBE_HERE":
            probes.append({
                "id": f"{cdata['evaluation_checkpoint_id']}_{probe['probe_id']}",
                "source": "personal",
                "prompt": prompt,
                "label": None
            })

with open(OUTPUT, "w", encoding="utf-8") as f:
    for p in probes:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
print(f"Wrote {len(probes)} probes to {OUTPUT}")