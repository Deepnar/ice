"""Bounded v3 source-support controls, not a benchmark or threshold fit."""
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = 'MoritzLaurer/mDeBERTa-v3-base-mnli-xnli'
REVISION = '8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c'
CASES = [
    ('direction', 'Mira teaches Niko.', 'Mira teaches Niko.', True),
    ('reversal', 'Mira teaches Niko.', 'Niko teaches Mira.', False),
    ('negation', 'Mira does not use Redis.', 'Mira uses Redis.', False),
    ('negative_supported', 'Mira does not use Redis.', 'Mira does not use Redis.', True),
    ('proposal', 'Mira is considering moving to Oslo.', 'Mira lives in Oslo.', False),
    ('hypothetical', 'If Mira moved to Oslo, she would cycle to work.', 'Mira cycles to work in Oslo.', False),
    ('question', 'Does Mira use Redis?', 'Mira uses Redis.', False),
    ('correction', 'We considered SQLite but chose PostgreSQL.', 'We chose PostgreSQL.', True),
    ('discarded_choice', 'We considered SQLite but chose PostgreSQL.', 'We chose SQLite.', False),
    ('numeric', 'The cache limit is 8 GB.', 'The cache limit is 4 GB.', False),
    ('conjunction', 'Mira uses PostgreSQL.', 'Mira uses PostgreSQL and Redis.', False),
    ('paraphrase', 'Mira relocated from Paris to Oslo.', 'Mira moved to Oslo.', True),
    ('attribution', 'Niko suggested that Mira use Redis.', 'Mira uses Redis.', False),
    ('quoted_denial', 'Mira said the claim "I use Redis" is false.', 'Mira uses Redis.', False),
    ('later_reversal', 'Initially we chose PostgreSQL. Later we abandoned it for SQLite.', 'We currently use PostgreSQL.', False),
    ('span_context', 'Mira did not say that Niko uses Redis.', 'Niko uses Redis.', False),
    ('hindi_support', 'मीरा दिल्ली में रहती है।', 'मीरा दिल्ली में रहती है।', True),
    ('hindi_negative', 'मीरा दिल्ली में नहीं रहती है।', 'मीरा दिल्ली में रहती है।', False),
    ('german_negative', 'Mira wohnt nicht in Berlin.', 'Mira wohnt in Berlin.', False),
    ('style', 'ok so like we thought sqlite but nah chose postgres instead', 'We chose PostgreSQL.', True),
]

ATTRIBUTED_CASES = [
    ('conditional_kept', 'If Mira moved to Oslo, she would cycle to work.', 'Mira would cycle to work if she moved to Oslo.', True),
    ('suggestion_kept', 'Niko suggested that Mira use Redis.', 'Niko suggested Redis to Mira.', True),
    ('question_kept', 'Mira asked whether Niko uses Redis.', 'Mira asked about Niko using Redis.', True),
    ('quote_kept', 'Mira said the claim "I use Redis" is false.', 'Mira denied using Redis.', True),
    ('reporting_kept', 'Mira did not say that Niko uses Redis.', 'Mira did not state that Niko uses Redis.', True),
    ('assistant_not_user', 'User: I am considering SQLite. Assistant: I suggest PostgreSQL.', 'The user chose PostgreSQL.', False),
    ('assistant_attributed', 'User: I am considering SQLite. Assistant: I suggest PostgreSQL.', 'The assistant suggested PostgreSQL.', True),
    ('tentative_not_final', 'User: Maybe PostgreSQL, but I have not decided.', 'The user decided on PostgreSQL.', False),
    ('attributed_decision', 'User: We chose PostgreSQL. Assistant: You could also try SQLite.', 'The user chose PostgreSQL.', True),
    ('later_denial', 'User: I previously used Redis. I no longer use it.', 'The user currently uses Redis.', False),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--output', default='experiments/v3_repair/results/nli_qualification.json')
    parser.add_argument('--attribution', action='store_true')
    args = parser.parse_args()
    model_name, revision = args.model, args.revision
    cases = CASES + (ATTRIBUTED_CASES if args.attribution else [])
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision, local_files_only=True, dtype=torch.float32).eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    rows = []
    try:
        model.to(device)
        for name, premise, hypothesis, supported in cases:
            inputs = tokenizer(premise, hypothesis, return_tensors='pt', truncation=False)
            assert inputs['input_ids'].shape[-1] <= model.config.max_position_embeddings
            with torch.inference_mode():
                logits = model(**{k: v.to(device) for k, v in inputs.items()}).logits
            probabilities = logits.softmax(-1)[0].cpu().tolist()
            scores = {model.config.id2label[i]: p for i, p in enumerate(probabilities)}
            rows.append(dict(name=name, premise=premise, hypothesis=hypothesis, supported=supported, scores=scores, predicted=max(scores,key=scores.get)))
        result = dict(version='ICE v3', model=model_name, revision=revision, dtype='float32', device=device, seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None, controls=rows)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
        for row in rows:
            print(row['name'], row['supported'], row['predicted'], round(row['scores']['entailment'], 4))
        print('seconds',result['seconds'],'peak',result['peak_allocated_bytes'])
    finally:
        model.cpu()
        if device == 'cuda':
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
