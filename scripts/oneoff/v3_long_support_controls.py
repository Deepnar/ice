"""Frozen synthetic long-source checks for a candidate; no production activation."""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.memory.source import single_provenance
from src.workers.conversation_summary import _original_source

MODEL = 'MoritzLaurer/bge-m3-zeroshot-v2.0'
REVISION = '9abf1c8aaeb82a2447809c20753ed0b106b76652'
CASES = [
 ('late_correction', 'user', 'Atlas initially chose PostgreSQL.',
  'Final correction: Atlas abandoned PostgreSQL and now uses SQLite instead.',
  'The user says Atlas now uses SQLite instead of PostgreSQL.',
  'The user says Atlas now uses PostgreSQL.'),
 ('late_condition', 'user', 'If approval arrives, Atlas may deploy Redis.',
  'Approval was denied. Atlas will not deploy Redis.',
  'The user says Redis was conditional on approval, which was denied.',
  'The user committed to deploy Redis after approval.'),
 ('speaker', 'assistant', 'I recommend Redis for Atlas.',
  'The user has still not agreed to use Redis.',
  'The assistant recommended Redis, but the user has not agreed.',
  'The user decided to use Redis for Atlas.'),
 ('hindi_negation', 'user', 'मीरा दिल्ली में नहीं रहती है।',
  'मीरा दिल्ली में नहीं रहती है।',
  'मीरा दिल्ली में नहीं रहती है।', 'मीरा दिल्ली में रहती है।'),
]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, revision=REVISION,
        local_files_only=True, dtype=torch.float32).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rows=[]
    target=Path('experiments/v3_repair/results/nli_bge_long_candidate.json')
    start=time.monotonic()
    if device == 'cuda': torch.cuda.reset_peak_memory_stats()
    try:
        model.to(device)
        for n in [50, 160, 340]:
            filler=' '.join(f'Unrelated documentation entry {i} describes the spare indexing notes.' for i in range(n))
            for name, role, beginning, ending, positive, negative in CASES:
                raw=f'{beginning}\n\n{filler}\n\n{ending}'
                row=SimpleNamespace(raw_text=raw, source_spans=single_provenance(raw,role),
                    timestamp=datetime(2026,9,20,tzinfo=timezone.utc),ts_provenance='original')
                source,_=_original_source(row)
                for expected,candidate in [(True,positive),(False,negative)]:
                    encoded=tokenizer(source,candidate,truncation=False,return_tensors='pt')
                    length=encoded['input_ids'].shape[-1]
                    assert 512 < length <= 8192, length
                    with torch.inference_mode():
                        probs=model(**{k:v.to(device) for k,v in encoded.items()}).logits.softmax(-1)[0].cpu().tolist()
                    scores={model.config.id2label[i]:p for i,p in enumerate(probs)}
                    admitted=scores['entailment']>=.95
                    rows.append(dict(name=name,filler_entries=n,pair_tokens=length,
                        source=source,candidate=candidate,expected_supported=expected,
                        admitted=admitted,scores=scores,passed=admitted==expected))
                    print(name,n,expected,length,round(scores['entailment'],5),flush=True)
                    target.write_text(json.dumps(dict(version='ICE v3', model=MODEL,revision=REVISION,
                        threshold=.95,dtype='float32',device=device,completed=len(rows),expected_total=24,
                        seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None,
                        scope='Synthetic complete long-source qualification, not natural-corpus summary quality.',
                        passed=sum(r['passed'] for r in rows),rows=rows),indent=2,ensure_ascii=False)+'\n')
    finally:
        model.cpu()
        if device=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':main()
