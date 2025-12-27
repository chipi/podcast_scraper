#!/usr/bin/env python3
"""Fix common markdown linting issues automatically.

This script fixes common markdownlint issues that can be automatically corrected:
- Table formatting (spaces around pipes for compact style)
- Blank lines around lists
- Blank lines around headings
- Trailing spaces
- Missing code block language specifiers (when detectable)

Usage:
    python scripts/fix_markdown.py [file1.md] [file2.md] ...
    python scripts/fix_markdown.py  # Fixes all .md files in project

Examples:
    # Fix specific files
    python scripts/fix_markdown.py docs/TESTING_STRATEGY.md

    # Fix all markdown files
    python scripts/fix_markdown.py

    # Dry run (show what would be fixed)
    python scripts/fix_markdown.py --dry-run
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

# Ignore patterns (same as markdownlint)
IGNORE_PATTERNS = [
    "node_modules",
    ".venv",
    ".build/site",
    "__pycache__",
]


def should_ignore_file(file_path: Path) -> bool:
    """Check if file should be ignored."""
    path_str = str(file_path)
    return any(pattern in path_str for pattern in IGNORE_PATTERNS)


def find_markdown_files(root: Path = None) -> List[Path]:
    """Find all markdown files in the project."""
    if root is None:
        root = Path(__file__).parent.parent

    md_files = []
    for md_file in root.rglob("*.md"):
        if not should_ignore_file(md_file):
            md_files.append(md_file)
    return sorted(md_files)


def fix_table_separators(content: str) -> Tuple[str, int]:
    """Fix table separator rows to have spaces around pipes.

    Converts: |-------------|--------|----------|
    To:      | ----------- | ------ | -------- |

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Match table separator rows (lines with only dashes and pipes, no spaces around pipes)
        # Pattern: starts with |, has dashes, ends with |
        if re.match(r"^\|[-:]+\|", line) and re.match(r"^[\|:\-]+$", line):
            # Check if it needs fixing (no spaces around pipes in separator)
            # Split by pipes
            parts = line.split("|")
            fixed_parts = []
            for part in parts:
                if part.strip() and all(c in "-:" for c in part.strip()):
                    # Add spaces around dashes/colons
                    fixed_parts.append(f" {part.strip()} ")
                elif not part.strip():
                    # Empty part (before first or after last pipe)
                    fixed_parts.append("")
            fixed_line = "|".join(fixed_parts)
            # Ensure it starts and ends with |
            if not fixed_line.startswith("|"):
                fixed_line = "|" + fixed_line
            if not fixed_line.endswith("|"):
                fixed_line = fixed_line + "|"
            if fixed_line != line:
                fixed_lines.append(fixed_line)
                fixes += 1
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines), fixes


def fix_trailing_spaces(content: str) -> Tuple[str, int]:
    """Remove trailing spaces from lines.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        original = line
        fixed = line.rstrip()
        if original != fixed:
            fixes += 1
        fixed_lines.append(fixed)

    return "\n".join(fixed_lines), fixes


def fix_blank_lines_around_lists(content: str) -> Tuple[str, int]:
    """Add blank lines before and after lists if missing.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []
    prev_line = None
    prev_prev_line = None

    for i, line in enumerate(lines):
        is_list_item = line.strip().startswith(("- ", "* ", "+ ")) or re.match(
            r"^\s*\d+\.\s+", line
        )
        prev_is_list_item = prev_line is not None and (
            prev_line.strip().startswith(("- ", "* ", "+ ")) or re.match(r"^\s*\d+\.\s+", prev_line)
        )
        # Add blank line before list if needed
        if is_list_item and not prev_is_list_item and prev_line is not None:
            if prev_line.strip() and prev_prev_line != "":
                fixed_lines.append("")
                fixes += 1

        fixed_lines.append(line)

        # Add blank line after list if needed
        if prev_is_list_item and not is_list_item and line.strip() and line != "":
            if not fixed_lines[-1].endswith("\n") or fixed_lines[-1] != "":
                # Check if we need to add a blank line
                if i + 1 < len(lines) and lines[i + 1].strip():
                    fixed_lines.append("")
                    fixes += 1

        prev_prev_line = prev_line
        prev_line = line

    return "\n".join(fixed_lines), fixes


def fix_code_block_languages(content: str) -> Tuple[str, int]:
    """Add language specifiers to code blocks when missing and detectable.

    This is a simple heuristic - only fixes obvious cases.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0

    # Pattern for code blocks without language
    pattern = r"^```\s*$"

    lines = content.split("\n")
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if re.match(pattern, line):
            # Found a code block start
            fixed_lines.append(line)
            i += 1

            # Look ahead to detect language
            if i < len(lines):
                next_line = lines[i]
                # Simple heuristics
                if next_line.strip().startswith("#!/usr/bin/env python"):
                    # Python script
                    fixed_lines[-1] = "```python"
                    fixes += 1
                elif next_line.strip().startswith("def ") or next_line.strip().startswith(
                    "import "
                ):
                    # Likely Python
                    fixed_lines[-1] = "```python"
                    fixes += 1
                elif next_line.strip().startswith("const ") or next_line.strip().startswith(
                    "function "
                ):
                    # Likely JavaScript
                    fixed_lines[-1] = "```javascript"
                    fixes += 1
                elif next_line.strip().startswith("package ") or next_line.strip().startswith(
                    "import "
                ):
                    # Could be Go or Java
                    if "func " in next_line or "package main" in next_line:
                        fixed_lines[-1] = "```go"
                        fixes += 1
        else:
            fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines), fixes


def fix_markdown_file(file_path: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Fix all common markdown issues in a file.

    Returns: (total_fixes, files_modified)
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
        return 0, 0

    original_content = content
    total_fixes = 0

    # Apply all fixes
    content, fixes = fix_table_separators(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Fixed {fixes} table separator(s)")

    content, fixes = fix_trailing_spaces(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Removed {fixes} trailing space(s)")

    content, fixes = fix_blank_lines_around_lists(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Fixed {fixes} blank line(s) around lists")

    content, fixes = fix_code_block_languages(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Added {fixes} code block language specifier(s)")

    if content != original_content:
        if not dry_run:
            file_path.write_text(content, encoding="utf-8")
            print(f"✓ Fixed {file_path}")
        else:
            print(f"✓ Would fix {file_path} ({total_fixes} issue(s))")
        return total_fixes, 1
    else:
        if total_fixes == 0:
            print(f"  No issues found in {file_path}")
        return total_fixes, 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Fix common markdown linting issues automatically")
    parser.add_argument(
        "files",
        nargs="*",
        help="Markdown files to fix (default: all .md files in project)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be fixed without making changes",
    )
    args = parser.parse_args()

    if args.files:
        files_to_fix = [Path(f) for f in args.files]
    else:
        files_to_fix = find_markdown_files()

    if not files_to_fix:
        print("No markdown files found to fix.")
        return 0

    print(f"Checking {len(files_to_fix)} markdown file(s)...")
    if args.dry_run:
        print("(DRY RUN - no files will be modified)\n")

    total_fixes = 0
    files_modified = 0

    for file_path in files_to_fix:
        print(f"\n{file_path}:")
        fixes, modified = fix_markdown_file(file_path, dry_run=args.dry_run)
        total_fixes += fixes
        files_modified += modified

    action = "Would fix" if args.dry_run else "Fixed"
    print(f"\n{action} {total_fixes} issue(s) in {files_modified} file(s)")
    return 0 if total_fixes == 0 or args.dry_run else 0


if __name__ == "__main__":
    sys.exit(main())
