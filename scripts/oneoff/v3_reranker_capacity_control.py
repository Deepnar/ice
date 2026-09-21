"""Actual v3 reranker: an oversized source cannot disable smaller evidence."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.api.config import settings
from src.memory.tokens import count
from src.retrieval.reranker import rerank
from src.retrieval.orchestrator import ContextFragment, HybridRetrievalOrchestrator

query='Which port does Atlas use?'
answer='Atlas listens on port 8391.'
texts=['Unrelated background material. ' * 1000,
       'Boreal listens on port 4422.', answer]
fragments=[ContextFragment(t,'episodic',3-i,count(t),source_batch_id=str(i),
                            covers_entire_source=(i==0)) for i,t in enumerate(texts)]
old=settings.retrieval_rerank_max_tokens
try:
    settings.retrieval_rerank_max_tokens=512
    ranked,ok=rerank(query,fragments)
    packed=HybridRetrievalOrchestrator(None,None)._enforce_token_budget(
        ranked,max_tokens=count(answer),relevance_order=ok)
    controls={'ranking_succeeded':ok,'answer_first':ranked[0].text==answer,
              'oversized_source_preserved':ranked[-1].text==texts[0],
              'packing_selects_answer':[f.text for f in packed]==[answer]}
    Path('experiments/v3_repair/results/reranker_capacity.json').write_text(json.dumps(
        dict(version='ICE v3',model=settings.retrieval_rerank_model,
             revision=settings.retrieval_rerank_revision,max_pair_tokens=512,
             synthetic_control=True,controls=controls,
             scored_order=[f.text for f in ranked if f.text!=texts[0]]),indent=2)+'\n')
    print(controls)
    assert all(controls.values())
finally:
    settings.retrieval_rerank_max_tokens=old
