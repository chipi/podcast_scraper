# Markdown Style Guide

**Short contract for Cursor AI and contributors.**

This is a concise reference. For detailed guidance, see [Markdown Linting Guide](MARKDOWN_LINTING_GUIDE.md).

**💡 For Cursor Users:** Pin this file in Cursor's context (Rules/Context features) so it's always
available when writing markdown. This helps Cursor follow these rules automatically.

## Core Rules

### Structure

- **One H1 per document** - Only the title should be `# Heading`
- **Blank line after headings** - Always add a blank line after `#`, `##`, etc.
- **80–100 char wrap** - Prefer wrapping at 100 characters (matches code style)

### Lists

- **Use `-` for unordered lists** - Consistent marker throughout
- **Blank lines around lists** - Add blank line before and after list blocks
- **Consistent indentation** - Use 2 spaces for nested lists

### Code

- **Code fences always specify language** - Use ` ```python`, ` ```bash`, ` ```yaml`, etc.
- **No bare code blocks** - Never use ` ``` ` without a language
- **Inline code with backticks** - Use `` `code` `` not `'code'`

### Formatting

- **No trailing spaces** - Trim whitespace at end of lines
- **No multiple blank lines** - Maximum one blank line between sections
- **Tables: avoid unless necessary** - Prefer lists when possible
- **Table formatting** - Use spaces around pipes: `| Column |` (compact style)

### Links

- **Descriptive link text** - Use `[Link Text](url)` not `[url](url)`
- **Relative paths for internal links** - Use `rfc/RFC-019.md` not `docs/rfc/RFC-019.md`

### Paragraphs

- **Prefer short paragraphs** - 2–5 lines maximum
- **One idea per paragraph** - Keep paragraphs focused

## Quick Checklist

Before finishing any markdown file, verify:

- [ ] Only one H1 (`#`) in the document
- [ ] Blank line after all headings
- [ ] No trailing spaces
- [ ] Lists have blank lines before and after
- [ ] Code fences have language specifiers
- [ ] Tables use consistent spacing around pipes
- [ ] No multiple consecutive blank lines

## Automated Fixing

Run before committing:

```bash

# Fix common issues automatically

make fix-md

# Or directly

python scripts/fix_markdown.py

# Verify with linter

make lint-markdown
```text
**For format-on-save (recommended):**

1. Install Prettier extension in Cursor/VSCode
2. `.vscode/settings.json` is already configured for format-on-save on markdown files
3. `.prettierrc` provides formatting rules (100 char wrap, preserves prose)

**Format on save will handle:**
- Wrapping at 100 characters
- Spacing and list formatting
- Trailing whitespace removal

**markdownlint handles:**
- Heading structure (one H1, blank lines after headings)
- List indentation
- Code fence language specifiers
- Table formatting

## See Also

- [Markdown Linting Guide](MARKDOWN_LINTING_GUIDE.md) - Complete markdown linting guide
- `scripts/fix_markdown.py` - Automated fixer script (see repository root)
- `.markdownlint.json` - Linter configuration (see repository root)
- `.prettierrc` - Prettier formatting rules (see repository root)
- `.vscode/settings.json` - Editor settings for format-on-save (see repository root)
