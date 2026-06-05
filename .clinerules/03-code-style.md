# Code Style & Conventions

## Python
- Every Python file must start with a module docstring: one sentence on what it does, one sentence on what it deliberately does NOT handle.
- Always use type hints on all function signatures and class attributes.
- Follow PEP 8; 4-space indentation; max line length 100.
- Import order: standard library → third-party → local (separated by blank lines).
- Use `dataclasses` or `pydantic` for structured data. Use `pydantic` for all API-facing models.
- All FastAPI route handlers and database calls must use `async`/`await`.
- Use `uv run python -c "import <module>"` to sanity-check imports after adding a new package.

## Naming
- Variables and functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private methods/attributes: prefix with `_`

## Error Handling
- Do not wrap everything in try/except. Let exceptions propagate to FastAPI's error handlers for request-scoped code.
- In background workers: catch exceptions, log with structlog including the correlation_id, then retry or fail gracefully depending on the task type.
- For idempotency-critical operations, always log the exception before deciding to retry.

## Database Rules
- Before processing any background event, check the `idempotency_keys` table. If the key exists, return immediately — do not process again.
- The `lossless_flag` on `episodic_memory` is NEVER set during pre-flight. It is set exclusively by the post-flight evaluator after the full exchange is complete.
- All schema changes go through Alembic migrations. Never call `Base.metadata.create_all()` in production code.
- All writes that must be atomic go inside a single SQLAlchemy transaction.

## Testing
- Write unit tests for all pure logic functions. Place in `tests/` mirroring the source structure.
- Use pytest. Run with `uv run pytest`.
- Follow the pattern of any existing test file in the same directory.

## General
- Only create files or modify code that directly implements the current step.
- Ask before changing any configuration file: `.env`, `pyproject.toml`, `docker/docker-compose.yml`, `alembic.ini`.
- If a task requires an architectural decision not covered by BLUEPRINT.md or ARCHITECTURE.md, stop and ask. Do not invent architecture.
- Add a comment above any non-obvious implementation choice explaining why it was done that way.