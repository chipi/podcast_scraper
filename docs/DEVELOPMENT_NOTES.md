# Development Notes

## Markdown Linting

### Catching Table Formatting Issues Locally

To catch markdown table formatting issues (MD060) before pushing:

```bash
# Run markdown linting locally
make lint-markdown

# Or directly with markdownlint
markdownlint "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site

# Auto-fix issues (when possible)
markdownlint --fix "**/*.md" --ignore node_modules --ignore .venv --ignore .build/site
```

### Common Table Formatting Issues

**MD060/table-column-style** errors occur when:

1. **Compact style**: Tables need spaces around pipes
   - ❌ Bad: `|Column1|Column2|`
   - ✅ Good: `| Column1 | Column2 |`

2. **Aligned style**: Table separator row must align with header
   - ❌ Bad: `| Header |` followed by `|-------|` (misaligned)
   - ✅ Good: `| Header |` followed by `| ------ |` (aligned)

### Pre-commit Hook

Consider adding a pre-commit hook to catch these automatically:

```bash
# Install pre-commit (if not already installed)
pip install pre-commit

# Add to .pre-commit-config.yaml:
#   - repo: https://github.com/igorshubovych/markdownlint-cli
#     rev: v0.38.0
#     hooks:
#       - id: markdownlint
```

### CI/CD Integration

The GitHub Actions workflow runs `make lint-markdown` which includes:

- `markdownlint "**/*.md"` with proper ignores
- Fails the build if any errors are found
- All markdown files are checked, not just changed files
