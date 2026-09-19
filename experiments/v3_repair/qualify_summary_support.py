"""v3 bounded summary-support controls, actual canonical source + NLI model."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.memory.representation import choose_representation, representation_source
from src.memory.source import chat_provenance
from src.memory.support import verify_support

CASES=[
 ('decision','We chose PostgreSQL instead of SQLite.','Acknowledged.','The user chose PostgreSQL rather than SQLite.',True),
 ('reversed_decision','We chose PostgreSQL instead of SQLite.','Acknowledged.','The user chose SQLite rather than PostgreSQL.',False),
 ('suggestion','We have not selected a cache.','I suggest Redis.','The assistant suggested Redis.',True),
 ('speaker_swap','We have not selected a cache.','I suggest Redis.','The user selected Redis.',False),
 ('negation','Keep persistence disabled.','Understood.','The user requested that persistence remain disabled.',True),
 ('negation_flip','Keep persistence disabled.','Understood.','The user requested that persistence be enabled.',False),
 ('quoted_denial','The claim "we use Redis" is false.','Understood.','The user denied using Redis.',True),
 ('quoted_adoption','The claim "we use Redis" is false.','Understood.','The user uses Redis.',False),
 ('quantity','The deployment has 3 replicas on port 8391.','Noted.','The user described 3 replicas on port 8391.',True),
 ('wrong_quantity','The deployment has 3 replicas on port 8391.','Noted.','The user described 8 replicas on port 8391.',False),
 ('conditional','If approval arrives we may use Redis. Approval was denied.','Understood.','The user said Redis was conditional on approval, which was denied.',True),
 ('invented_commitment','If approval arrives we may use Redis. Approval was denied.','Understood.','The user committed to Redis after approval.',False),
]

def main():
 rows=[]
 for name,user,assistant,summary,expected in CASES:
  row=SimpleNamespace(raw_text=f'User: {user}\n\nAssistant: {assistant}',source_spans=chat_provenance(user,assistant))
  source=representation_source(row)
  verdict=verify_support(source,summary)
  row.summary_text=summary;row.summary_coverage=1.0;row.abstract_text=None;row.inject_raw=False
  row.representation_verification={'summary':asdict(verdict)}
  selected=choose_representation(row)[0]
  passed=((verdict.status=='supported')==expected and (selected==summary)==expected)
  rows.append(dict(name=name,source=source,summary=summary,expected_supported=expected,passed=passed,verdict=asdict(verdict)))
  print(name,verdict.status,passed,flush=True)
 Path('experiments/v3_repair/results/summary_support.json').write_text(json.dumps(dict(version='v3',
  limitation='Synthetic qualification; not summary completeness or long-source validation.',
  passed=sum(r['passed'] for r in rows),total=len(rows),rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
