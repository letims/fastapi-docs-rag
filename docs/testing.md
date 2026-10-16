# Tests and coverage

```bash
make test        # full suite; FAILS if line+branch coverage < 90%; writes htmlcov/index.html
make cov-open    # open the HTML report (macOS)
```

DB-backed tests run against `TEST_DATABASE_URL` if set (CI uses a pgvector service container),
otherwise against a throwaway Postgres from the `pgserver` dev dependency, so no Docker is needed.
The agent loop is tested offline with fake providers (`tests/test_agent.py`): routing, each stop rule,
the grader parser and its fail-open path. Nothing there calls Groq or Voyage, so those tests say the
code does what it is written to do, not that the loop improves answers (that is what the evals measure).
The gate lives in `pyproject.toml` (`[tool.coverage.report] fail_under = 90`) and in CI.
