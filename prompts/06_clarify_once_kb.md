# 06 — Clarify-once persistent knowledge base
**Read:** docs/03_CLARIFY_ONCE_MEMORY_SPEC.md completely, docs/01 (3, 6), docs/06 (T-KB-*).

**Task:** implement the spec as written.
1. `kb/normalize.py` with >= 15 unit tests (articles, casing, punctuation, trailing "camera/area", plural).
2. `kb/store.py`: CRUD on `referents`, `referent_history`, `clarification_log`, `pending_queries`. Commit before returning. `lookup(site_id, phrase) -> targets | None`, `suggest_similar(site_id, phrase)` using a string-similarity threshold from config.
3. Integrate into `query/resolver.py`: resolution order explicit camera id -> KB -> manifest label -> unresolved. Unresolved phrases create a pending query and a clarification response (`type:"clarification"`, contents per docs/01 section 6).
4. `POST /api/clarify` handler logic (the FastAPI route itself comes in prompt 07, so expose a plain function now): validates targets, writes the referent (or alias via `same_as`), logs, loads the pending query, re-runs retrieval, returns the answer.
5. ROI semantics from spec section 5 inside retrieval.
6. Tests T-KB-1..T-KB-9 as automated tests where possible. T-KB-3 must spawn a real subprocess server (or a subprocess performing the writes), `kill -9` it right after the confirmed write, restart a new process and assert there is no clarification.

**Constraints:** never ask when the KB hits; never write without commit; cancel path must not persist anything. Do not add fuzzy auto-merge.
**Acceptance:** paste the pytest output including the restart test. Show `sqlite3 data/omniwatch.db 'select * from referents'` after a clarification.
