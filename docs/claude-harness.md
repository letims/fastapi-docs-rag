# How the Claude Code harness operates

One retrieval change, end to end: Claude works autonomously on a branch; permissions, hooks, evals and a read-only reviewer decide what is safe; Leticia reviews the branch diff and pushes. Source of truth for each step: `.claude/settings.json`, `.claude/hooks/`, `.claude/agents/`, `evals/compare.py`, `.github/workflows/`.

```mermaid
sequenceDiagram
    autonumber
    actor L as Leticia
    participant C as Claude Code<br/>(auto mode)
    participant P as Permissions<br/>+ Sandbox
    participant G as PreToolUse<br/>guard
    participant R as PostToolUse<br/>ruff
    participant E as Eval harness<br/>(eval-compare)
    participant V as rag-eval-reviewer<br/>(read-only)
    participant S as Stop hook<br/>(unit tests)
    participant GH as GitHub<br/>(CI + @claude)

    L->>C: Task: "cap rerank results per document"
    Note over C: Loads CLAUDE.md (~1.6k tokens)<br/>+ path-scoped rules + retrieval-eval skill
    opt Ambiguous, irreversible or spends API money
        C->>L: Short plan / question
        L-->>C: Go
    end

    C->>G: git switch -c claude/cap-per-source
    G-->>C: allowed (not main)

    loop Implement until checks pass
        C->>P: Edit app/rag/retrieve.py + test
        P-->>C: auto-approved (acceptEdits / auto)
        P->>R: after each edit
        R-->>C: ruff fix + format, leftovers reported
        C->>P: Bash command
        alt curl, cat .env, rm -rf, git push, commit on main
            P--xC: deny rule matches
            G--xC: or guard exit 2, with the reason
        else uv run pytest, make check, git commit on branch
            P-->>C: allowed, runs in sandbox (network allowlist)
        end
    end

    C->>E: make eval-compare MODE=rerank
    E->>E: Retrieval eval (cached) + provenance<br/>diff vs evals/baseline/rerank.json
    alt Drop > 1 question, any abstention drop, cost +25%
        E--xC: exit 1, REGRESSION + questions that moved
        C->>C: Iterate or revert the idea
    else No regression
        E-->>C: Delta table (e.g. full_recall 0.94 → 1.00)
    end

    opt Answer-level check (Groq quota)
        C->>L: Run make eval-judge? (ask rule)
        L-->>C: Yes
        C->>E: eval-judge → grounded_rate, cost/query
    end

    C->>V: Diff + compare table
    V-->>C: VERDICT + findings<br/>(leakage, injection, re-embed churn, bounds)
    C->>G: git commit on claude/cap-per-source
    G-->>C: allowed

    C->>S: Turn ends
    S->>S: pytest -m "not db" (~5 s)
    alt Tests fail and stop_hook_active is false
        S--xC: exit 2, failure tail: keep working
    else Pass, or second attempt (loop guard)
        S-->>C: may stop
    end
    C-->>L: Result + evidence + branch diff

    L->>L: Review branch diff
    L->>GH: git push (only Leticia pushes)
    par On the PR
        GH->>GH: ci.yml: lint, JSON schemas, tests (90% gate)
    and Retrieval code touched
        GH->>GH: eval.yml: index 40 docs, rerank eval<br/>vs ci-dev40-rerank, table in job summary
    end
    opt Leticia writes @claude on the PR
        GH->>GH: claude.yml: write-access only, 8 turns,<br/>same .claude rules from base branch
    end
```
