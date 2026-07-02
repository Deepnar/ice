"""MERA – Meta‑Enumeration Retrieval Agent for category‑based entity retrieval."""

from typing import List, Optional
from sqlalchemy.orm import Session
from src.memory.models import CodexEntity, CodexEdge, EpisodicMemory
from src.workers.bg_client_factory import get_bg_client, get_bg_model_name
from datetime import datetime, timezone

CATEGORY_TRIGGERS = [
    "character", "characters", "function", "functions", "class", "classes",
    "library", "libraries", "dependency", "dependencies", "module", "modules",
    "role", "roles", "species", "entity", "entities", "person", "people",
    "tool", "tools", "framework", "frameworks", "component", "components",
    "service", "services", "endpoint", "endpoints", "model", "models",
    "algorithm", "algorithms", "method", "methods", "variable", "variables",
    "project", "projects", "team", "teams", "member", "members", "friend",
    "friends", "enemy", "enemies", "item", "items", "weapon", "weapons",
]

def is_mera_candidate(prompt: str) -> bool:
    prompt_lower = prompt.lower()
    if not any(trigger in prompt_lower for trigger in CATEGORY_TRIGGERS):
        return False
    enum_hints = ["list", "all", "who are", "what are", "every", "each",
                  "tell me about", "what is", "which", "name"]
    return any(hint in prompt_lower for hint in enum_hints)

def map_category_to_filters(db: Session, prompt: str) -> dict:
    bg_client = get_bg_client()
    try:
        completion = bg_client.chat.completions.create(
            model=get_bg_model_name(),
            messages=[
                {"role": "system", "content": (
                    "You are a category mapper. Given a user's request for a list of entities "
                    "(e.g., 'who are the characters?'), output a JSON object with two keys: "
                    "'tags' (an array of Codex entity tags to filter by) and 'relations' "
                    "(an array of edge relation types to search for). "
                    "Use the following known tags and relations when applicable:\n"
                    "Tags: character, person, tool, library, project, concept, location, event, "
                    "organization, software, hardware, algorithm, model, framework, language\n"
                    "Relations: name, type, uses, imports, extends, friend, enemy, member_of, "
                    "works_on, created, part_of\n"
                    "If no mapping is clear, return {'tags':[], 'relations':[]}."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150,
            timeout=15.0
        )
        import json
        result = json.loads(completion.choices[0].message.content.strip())
        return {"tags": result.get("tags", []), "relations": result.get("relations", [])}
    except Exception:
        return {"tags": [], "relations": []}

def enumerate_entities(
    db: Session,
    tags: List[str],
    relations: List[str],
    limit: int = 15
) -> List[CodexEntity]:
    """Return entities matching tags, involved in specified relations, or
    having properties whose key matches a relation filter, ranked by recency."""
    entities = []
    seen = set()

    # By tag
    for tag in tags:
        for ent in db.query(CodexEntity).filter(CodexEntity.tags.any(tag)).all():
            if ent.id not in seen:
                entities.append(ent)
                seen.add(ent.id)

    # By relation (source of property edges)
    for rel in relations:
        edges = db.query(CodexEdge).filter(
            CodexEdge.relation == rel,
            CodexEdge.valid_until == None
        ).limit(50).all()
        for edge in edges:
            source_ent = db.get(CodexEntity, edge.source_id)
            if source_ent and source_ent.id not in seen:
                entities.append(source_ent)
                seen.add(source_ent.id)

    # Fallback: if nothing found yet, look for entities whose properties
    # contain a key from the relation filter (e.g., 'name', 'role')
    if not entities and relations:
        for ent in db.query(CodexEntity).all():
            if ent.id in seen:
                continue
            props = ent.properties or {}
            if any(rel in props for rel in relations):
                entities.append(ent)
                seen.add(ent.id)

    # Rank by mention recency
    def mention_score(entity):
        from sqlalchemy import text
        try:
            count = db.execute(
                text("SELECT COUNT(*) FROM episodic_memory WHERE raw_text ILIKE :name AND timestamp > NOW() - INTERVAL '30 days'"),
                {"name": f"%{entity.canonical_name}%"}
            ).scalar()
            return count or 0
        except Exception:
            return 0

    entities.sort(key=lambda e: (mention_score(e), e.last_updated or datetime.min), reverse=True)
    return entities[:limit]
