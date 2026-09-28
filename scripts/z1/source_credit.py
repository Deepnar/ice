"""Shared row/batch provenance for Z1 and Z2 retrieval instruments."""


def fragment_source_ids(fragment) -> set[str]:
    """Source row for episodic evidence; source batch for derived evidence."""
    ids = set()
    if fragment.source_batch_id:
        ids.add(str(fragment.source_batch_id))
    ids.update(str(batch) for batch in (fragment.origin_batch_ids or ()))
    return ids


def gold_source_ids(row_id, batch_by_row: dict[str, str]) -> set[str]:
    """Both IDs of one gold turn; neither ID may be guessed from the other."""
    if row_id is None:
        return set()
    row_id = str(row_id)
    return {row_id} | ({batch_by_row[row_id]} if row_id in batch_by_row else set())


def covered_gold_turns(row_ids, fragments, batch_by_row: dict[str, str]) -> int:
    """Count gold *turns*, not the two IDs each turn may carry."""
    returned = {source_id for fragment in fragments
                for source_id in fragment_source_ids(fragment)}
    return sum(bool(gold_source_ids(row_id, batch_by_row) & returned)
               for row_id in row_ids)
