# Tests and coverage

```bash
make test        # full suite; FAILS if line+branch coverage < 90%; writes htmlcov/index.html
make cov-open    # open the HTML report (macOS)
```

DB-backed tests run against `RAGNAROK_TEST_DB` if set (CI uses a pgvector service container),
otherwise against a throwaway Postgres from the `pgserver` dev dependency, so no Docker is needed.
The gate lives in `pyproject.toml` (`[tool.coverage.report] fail_under = 90`) and in CI.
