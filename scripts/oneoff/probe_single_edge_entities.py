"""Are single-edge entities GENUINE one-mention facts, or missed connections?

64.2% of arm B's entities carry exactly one edge. That is either fine — the
corpus mentions the thing once, so one triplet is the whole truth — or it is
[G51](../../docs/ROADMAP.md#g51): the entity recurs across the corpus and the
extractor never linked it up.

**The decisive signal needs no model: how many TURNS mention the entity?**
One edge + one mentioning turn = genuine. One edge + many mentioning turns =
connections were missed, and the count of those turns bounds how many.

⚠ **Short names are excluded from the mention count, deliberately.** A name like
`2` or `in` matches almost every turn as a substring and would report the whole
store as "missed". Word-boundary regex over 4k names x 293 turns is also where
this stops being cheap. Names under `--min-name` characters are counted
separately and reported as UNMEASURED rather than folded into either bucket —
saying so in the number's own words beats a clean-looking total.
"""
from __future__ import annotations
import argparse, os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from sqlalchemy import text                    # noqa: E402
from src.api.db import SessionLocal            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="(live)")
    ap.add_argument("--min-name", type=int, default=4)
    ap.add_argument("--examples", type=int, default=12)
    args = ap.parse_args()
    db = SessionLocal()

    rows = db.execute(text("""
        with deg as (
          select e.id, e.canonical_name as name,
                 (select count(*) from codex_edges x
                   where (x.source_id=e.id or x.target_id=e.id)
                     and x.valid_until is null) as d
          from codex_entities e
          where e.context_payload not like '[merged into %%' or e.context_payload is null
        ),
        single as (select * from deg where d = 1)
        select s.name,
               (select count(*) from episodic_memory m
                 where length(s.name) >= :minlen
                   and m.raw_text ~* ('\\m' || regexp_replace(s.name,
                        '([.^$*+?()\\[\\]{}|\\\\])', '\\\\\\1', 'g') || '\\M')
               ) as turns_mentioning,
               length(s.name) as namelen
        from single s
    """), {"minlen": args.min_name}).fetchall()

    # ⚑ A THIRD BUCKET, ADDED AFTER THE FIRST RUN. The initial version reported
    # `with` (244 turns), `about` (201) and `more` (174) as the worst "missed
    # connections". They are not missed connections — they are STOPWORDS THAT
    # BECAME ENTITIES. `_NER_STOP` holds only 24 words and none of these. They
    # come from the LLM extractor, not the NER: an ungrounded triplet is still
    # stored (at `codex_conf_rejected`), so a subject the whitelist never
    # confirmed still mints a row. Counting them as "missed" would have inflated
    # G51's backlog with junk and hidden the real number.
    STOP = {
        "with","about","more","know","still","good","different","some","both",
        "look","year","doing","this","that","these","those","there","here",
        "what","when","where","which","while","would","could","should","been",
        "being","have","having","just","like","very","much","many","most",
        "also","only","even","then","than","them","they","their","your","yours",
        "from","into","over","under","after","before","other","another","same",
        "such","each","every","because","thing","things","stuff","something",
        "anything","everything","nothing","someone","anyone","everyone","time",
        "times","way","ways","part","parts","kind","sort","lot","lots","bit",
        "point","case","fact","idea","thought","feel","feels","felt","want",
        "wants","need","needs","make","makes","made","take","takes","come",
        "comes","goes","going","gets","said","says","tell","tells","told",
        "think","thinks","really","actually","maybe","perhaps","always",
        "never","often","sometimes","again","back","down","out","off","up",
    }
    total = len(rows)
    short = [r for r in rows if r.namelen < args.min_name]
    junk = [r for r in rows if r.namelen >= args.min_name
            and r.name.strip().lower() in STOP]
    long_ = [r for r in rows if r.namelen >= args.min_name
             and r.name.strip().lower() not in STOP]
    genuine = [r for r in long_ if r.turns_mentioning <= 1]
    missed = [r for r in long_ if r.turns_mentioning >= 2]
    heavy = [r for r in long_ if r.turns_mentioning >= 5]

    # ⚑ SAY WHICH STORE THIS ACTUALLY MEASURED. The first run of this probe
    # raced a snapshot restore: the store was being wiped underneath it, so it
    # saw 0 single-edge entities and divided by zero. Crashing was the right
    # outcome — a silent 0 would have been reported as "no missed links". The
    # totals below let the reader confirm the arm, since `--arm` is only a label.
    ents = db.execute(text("select count(*) from codex_entities")).scalar()
    edges = db.execute(text(
        "select count(*) from codex_edges where valid_until is null")).scalar()
    print(f"=== SINGLE-EDGE ENTITIES — {args.arm} ===")
    print(f"store as measured: {ents:,} entities · {edges:,} live edges "
          f"(arm A = 8,470 / arm B = 6,271)")
    if total == 0:
        print("⛔ ZERO single-edge entities — the store is empty or mid-restore. "
              "Nothing measured; re-run when a snapshot is fully loaded.")
        db.close()
        return 1
    print(f"entities with exactly ONE edge: {total:,}")
    print(f"  name < {args.min_name} chars — UNMEASURED (substring noise): "
          f"{len(short):,} ({100*len(short)/total:.1f}%)")
    print(f"  JUNK — a stopword that became an entity: "
          f"{len(junk):,} ({100*len(junk)/total:.1f}%)   <- not a missed link")
    if long_:
        n = len(long_)
        print(f"  measurable: {n:,}")
        print(f"    GENUINE  (mentioned in <=1 turn): {len(genuine):,} "
              f"({100*len(genuine)/n:.1f}%)")
        print(f"    MISSED   (mentioned in >=2 turns): {len(missed):,} "
              f"({100*len(missed)/n:.1f}%)")
        print(f"      of which >=5 turns: {len(heavy):,} "
              f"({100*len(heavy)/n:.1f}%)")
        # ⚑ SPLIT BY WORD COUNT — because "1 edge + many mentions" turns out to
        # measure JUNK, not missed links. A genuinely important entity
        # mentioned in 138 turns would have accumulated edges; `will`,
        # `enough`, `between`, `once`, `itself` and `across` have exactly one
        # because a common word got extracted by accident a single time. A
        # stoplist is whack-a-mole here (mine caught 15 of them), so the
        # defensible cut is structural: MULTI-WORD names are overwhelmingly
        # real ('project timeline', 'obsidian citadel'), single common words
        # overwhelmingly are not. The single-word bucket is reported but NOT
        # claimed as missed links.
        multi = [r for r in long_ if " " in r.name.strip()]
        single = [r for r in long_ if " " not in r.name.strip()]
        m_missed = [r for r in multi if r.turns_mentioning >= 2]
        s_missed = [r for r in single if r.turns_mentioning >= 2]
        print(f"\n  SPLIT BY SHAPE (the junk/real cut):")
        print(f"    MULTI-WORD names  n={len(multi):,}  "
              f"missed (>=2 turns): {len(m_missed):,} "
              f"({100*len(m_missed)/max(len(multi),1):.1f}%)   <- credible missed links")
        print(f"    single-word names n={len(single):,}  "
              f"'missed': {len(s_missed):,} "
              f"({100*len(s_missed)/max(len(single),1):.1f}%)   <- mostly junk, NOT claimed")
        print(f"\n  credible missed links (multi-word, >=2 turns), top:")
        for r in sorted(multi, key=lambda r: -r.turns_mentioning)[:args.examples]:
            if r.turns_mentioning >= 2:
                print(f"    {r.turns_mentioning:>3} turns · 1 edge · {r.name[:56]}")
        print(f"\n  worst offenders overall (mostly junk — see above):")
        for r in sorted(long_, key=lambda r: -r.turns_mentioning)[:args.examples]:
            print(f"    {r.turns_mentioning:>3} turns · 1 edge · {r.name[:56]}")
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
