# Secret & Identifier Gates

Two complementary gates prevent secrets and operator-identifying literals from
landing in the public repo:

| Gate | Enforcement point | Authoritative? |
| --- | --- | --- |
| gitleaks scan | pre-commit hook + CI (`secret-scan.yml`) | CI is authoritative |
| `.env` guard | pre-commit hook + CI | CI is authoritative |
| Operator identifier deny-list | pre-commit hook (local) + CI | CI is authoritative |

---

## Gate 1 — gitleaks

### What it catches

gitleaks v8 uses the built-in ruleset (`useDefault = true`) plus any rules in
`.gitleaks.toml` at the repo root. It detects token shapes for HuggingFace,
OpenAI, Anthropic, GitHub PATs, AWS access keys, generic API key assignments,
and more.

### Where it runs

**Pre-commit hook** (`.github/hooks/pre-commit`): if `gitleaks` is on `PATH`,
runs `gitleaks protect --staged -c .gitleaks.toml --no-banner` against the
staged index. If gitleaks is absent, the hook prints a warning and continues —
dev machines without it are not hard-blocked. Install with:

```bash
brew install gitleaks
```

**CI** (`secret-scan.yml` → job `gitleaks`): downloads gitleaks v8.30.1,
runs it against the entire PR commit range
(`origin/<base>..HEAD`). Fails the PR on any finding.

The CI job also explicitly checks that no bare `.env` file (files whose name
IS `.env`, not `.env.example` or `.env.test`) is tracked by git.

### Allowlist policy

`.gitleaks.toml` extends the default ruleset and narrows false-positive scope:

- **Excluded paths** — `.venv/`, `node_modules/`, `.cache/`, `.test_outputs/`,
  `test-results/`, `htmlcov/`, `*.ipynb`, `data/eval/runs/` are skipped.
  These are vendored, generated, or historically-frozen trees; real secrets
  there are handled at the source, not the scan.

- **Allowed regex patterns** — narrow patterns for:
  - `max_tokens=4096` / `max_tokens: 4096` — LLM config, never a secret.
  - Model fingerprint digests (SHA-256/MD5 after `hash:`/`checksum:` etc.).
  - RFC/ADR prose titles containing "key" or "secret" as English words.
  - Shell `${VAR}` / `$VAR` references — variable names are not values.
  - Explicit placeholder strings (`changeme`, `hf_xxx`, `<redacted>`, etc.).

- **NOT disabled** — `generic-api-key` and all provider token rules remain
  active. Allowlist entries must be narrow; never add a blanket disable.

To add a new allowlist entry: edit `.gitleaks.toml`, keep the regex tight,
add a comment explaining *why* it is a false positive, verify with
`gitleaks git -c .gitleaks.toml --log-opts="-3"`.

---

## Gate 2 — Operator identifier deny-list

The operator's real name, email, domain, or other identifying literals must
never appear in a commit. The deny-list of those literals is itself sensitive —
committing it would publish the values. The solution:

### CI (authoritative)

The literal list lives in a **GitHub Actions secret** named
`SECURITY_IDENTIFIER_DENYLIST` (newline-separated). The operator populates it
via **Settings → Secrets and variables → Actions → New repository secret**.

The `secret-scan.yml` job `identifier-denylist`:

1. Reads the secret into the `DENYLIST` env var.
2. If empty/unset: prints a warning and passes (so the gate never blocks before
   the secret is populated).
3. If set: builds the full PR diff, then `grep -F` each literal against the
   diff. On a match it prints the diff FILE path — never the matched literal —
   and fails the job.

### Pre-commit (best-effort, local)

The hook checks for `.git/identifier-denylist` (one literal per line). This
file is:

- Inside `.git/` — not tracked, never committed, not visible to anyone who
  clones the repo.
- Populated by the operator manually on each machine:

```bash
cat > .git/identifier-denylist <<'EOF'
your.real@email.com
your-real-domain.com
EOF
```

If the file is absent the hook skips silently. If present, it `grep -F`s each
literal against the staged diff and fails, printing only that a match was found
(never the literal value).

---

## Checklist when the gate fires

1. **gitleaks finding**: remove the secret from staged files. Replace with an
   environment variable reference (`${SECRET_NAME}`). If it is a documented
   false positive, add a narrow entry to `.gitleaks.toml` allowlist with an
   explanation, then re-verify.

2. **`.env` guard**: rename to `.env.example` with placeholder values. Ensure
   `.env` is in `.gitignore`.

3. **Identifier finding**: remove or redact the literal from the staged file.
   Do not put the literal in a code comment, log line, or fixture — even
   redacted history is history.
