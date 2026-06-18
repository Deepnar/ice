#!/usr/bin/env python3
"""
Scan curation files for missing ground truth (placeholders).
Outputs a list of files and probes that need 'Surgical Oracle' generation.
"""

import json
import os
import glob

CURATION_DIR = "experiments/curation_files"
PLACEHOLDER = "ENTER_EXPECTED_ANSWER_OR_BLANK"

def scan_placeholders():
    files = glob.glob(os.path.join(CURATION_DIR, "*.json"))
    missing_probes = []
    total_probes = 0
    
    print(f"Scanning {len(files)} curation files...")
    
    for fpath in sorted(files):
        with open(fpath, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue
                
            checkpoint_id = data.get("evaluation_checkpoint_id", os.path.basename(fpath))
            split_turn = data.get("split_turn_index", "UNKNOWN")
            probes = data.get("evaluation_probes", [])
            
            for p in probes:
                total_probes += 1
                if p.get("expected_answer") == PLACEHOLDER:
                    # Skip empty/placeholder injected prompts too
                    if p.get("user_injected_prompt") == "ENTER_PROBE_HERE":
                        continue
                        
                    missing_probes.append({
                        "file": os.path.basename(fpath),
                        "checkpoint_id": checkpoint_id,
                        "turn": split_turn,
                        "probe_id": p.get("probe_id"),
                        "prompt": p.get("user_injected_prompt")
                    })

    print("-" * 50)
    print(f"SCAN COMPLETE")
    print(f"Total Probes found: {total_probes}")
    print(f"Probes with missing Truth: {len(missing_probes)}")
    print("-" * 50)
    
    # Group by file for easier fixing
    current_file = ""
    for item in missing_probes:
        if item["file"] != current_file:
            print(f"\n📁 FILE: {item['file']} (Turn {item['turn']})")
            current_file = item["file"]
        print(f"   └─ {item['probe_id']}: {item['prompt'][:100]}...")

if __name__ == "__main__":
    scan_placeholders()