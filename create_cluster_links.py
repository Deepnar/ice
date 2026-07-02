#!/usr/bin/env python3
"""Create the episodic_cluster_links table if it doesn't exist."""
from src.api.db import engine
from sqlalchemy import text

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS episodic_cluster_links (
            episodic_id UUID NOT NULL REFERENCES episodic_memory(id),
            cluster_id UUID NOT NULL REFERENCES context_clusters(id),
            PRIMARY KEY (episodic_id, cluster_id)
        )
    """))
    conn.commit()
    print("Table 'episodic_cluster_links' is now ready.")