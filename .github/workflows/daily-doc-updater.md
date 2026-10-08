---
description: Daily workflow that finds documentation out of sync with recent code changes and opens a pull request with the fixes.
intent: Keep repository documentation accurate by updating docs that drifted from code merged in the last 24 hours.
on:
  schedule: daily
  workflow_dispatch:
  skip-if-match:
    query: 'is:pr is:open in:title "[docs]" label:documentation'
    max: 1
permissions:
  contents: read
  pull-requests: read
  issues: read
tools:
  github:
    mode: gh-proxy
    toolsets: [default]
network:
  allowed:
    - defaults
    - python
safe-outputs:
  create-pull-request:
    title-prefix: "[docs] "
    labels: [documentation]
    draft: false
    allowed-files:
      - readme.md
      - "**/*.md"
      - "**/*.rst"
      - "docs/**"
      - tutorial/**
---

# Daily Documentation Updater

## Task

Keep the documentation of `${{ github.repository }}` in sync with the code.

1. **Window**: consider commits merged to the default branch in the last 24 hours (`gh api` / `git log --since="24 hours ago"`).
2. **Find changes**: for each commit, inspect the diff under `src/` and `pyproject.toml` for new, removed, or renamed public APIs, changed signatures or defaults, new CLI options, changed dependencies or supported Python versions, and behavior changes.
3. **Find stale docs**: compare these against the documentation: `readme.md`, other `*.md`/`*.rst` files, and the tutorial in `tutorial/` (code examples must still be valid). Only flag things the code changes actually made wrong or incomplete.
4. **Fix**: edit only documentation files (never source code or tests). Match the existing style, keep edits minimal and factual, and verify every claim against the code.
5. **Output**: if you made edits, call `create-pull-request` with a concise title and a body listing each doc file changed and the code commit(s) that motivated it.

## Safe Outputs

- Use `create-pull-request` only when at least one documentation file was updated.
- Call `noop` with a short reason if there were no code changes in the window, or all docs are already accurate.
