"""Renumber synthetic prompt IDs in a JSONL file.

Reads JSON lines from an input file, replaces the first matching string field
that starts with "synth_" to a new sequential series beginning at a given
start index (default 3792), and writes the results to an output JSONL file.

Usage:
	python synth_promt_renumber.py \
		--input /path/to/synthetic_prompts_labeled.jsonl \
		--output /path/to/synthetic_prompts_renumbered_labeled.jsonl \
		--start 3792
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import IO


def find_and_replace_id(obj: dict, new_label: str) -> bool:
	"""Find a string field starting with 'synth_' and replace it.

	Returns True if a replacement was made, False otherwise.
	"""
	for k, v in obj.items():
		if isinstance(v, str) and v.startswith("synth_"):
			obj[k] = new_label
			return True
	# no existing synth_ string field found; if there's an 'id' that's numeric,
	# set it to the new label. If 'id' exists and is not numeric, don't overwrite.
	if "id" in obj and not isinstance(obj["id"], str):
		obj["id"] = new_label
		return True
	return False


def renumber_jsonl(input_f: IO[str], output_f: IO[str], start: int) -> int:
	"""Read JSONL from input_f, renumber, write to output_f.

	Returns the count of processed records.
	"""
	idx = start
	count = 0
	for line in input_f:
		line = line.strip()
		if not line:
			continue
		obj = json.loads(line)
		label = f"synth_{idx}"
		replaced = find_and_replace_id(obj, label)
		if not replaced:
			# if nothing was replaced, add an 'id' field
			obj["id"] = label
		output_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
		idx += 1
		count += 1
	return count


def main() -> None:
	p = argparse.ArgumentParser(description="Renumber synthetic prompt IDs in a JSONL file")
	p.add_argument("--input", required=True, help="Path to input JSONL file")
	p.add_argument("--output", required=True, help="Path to output JSONL file")
	p.add_argument("--start", type=int, default=3792, help="Starting index for synth_ labels")
	args = p.parse_args()

	input_path = Path(args.input).expanduser()
	output_path = Path(args.output).expanduser()

	if not input_path.exists():
		raise SystemExit(f"Input file not found: {input_path}")

	with input_path.open("r", encoding="utf-8") as inf, output_path.open("w", encoding="utf-8") as outf:
		count = renumber_jsonl(inf, outf, args.start)

	print(f"Renumbered {count} records and wrote to {output_path}")


if __name__ == "__main__":
	main()

