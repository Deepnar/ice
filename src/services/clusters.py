"""Cluster service (E0): explicit cluster creation & turn assignment (C5)."""
import uuid
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session

from src.memory.models import ContextCluster, EpisodicClusterLink, EpisodicMemory
from src.services.errors import NotFoundError


def create_cluster(db: Session, name: str, description: str = "") -> dict:
    cluster = ContextCluster(name=name, description=description,
                             created_at=datetime.now(timezone.utc))
    db.add(cluster)
    db.commit()
    db.refresh(cluster)
    return {"id": str(cluster.id), "name": cluster.name}


def assign_turns(db: Session, cluster_id: str, turn_ids: List[str]) -> dict:
    cluster = db.query(ContextCluster).filter_by(id=uuid.UUID(cluster_id)).first()
    if not cluster:
        raise NotFoundError("Cluster not found")
    assigned = 0
    for tid in dict.fromkeys(turn_ids):
        turn = db.query(EpisodicMemory).filter_by(id=uuid.UUID(tid)).first()
        if turn:
            # C5's membership table drives clustering and retrieval. Keep the
            # legacy primary pointer for existing clients and cold archiving.
            if db.get(EpisodicClusterLink, (turn.id, cluster.id)) is None:
                db.add(EpisodicClusterLink(episodic_id=turn.id, cluster_id=cluster.id))
            turn.cluster_id = cluster.id
            assigned += 1
    db.commit()
    return {"assigned": assigned}
