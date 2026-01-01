#!/usr/bin/env python3
"""Fix common markdown linting issues automatically.

This script fixes common markdownlint issues that can be automatically corrected:
- Table formatting (spaces around pipes for compact style)
- Blank lines around lists (MD032)
- Blank lines around fenced code blocks (MD031)
- Blank lines around headings (MD022)
- Trailing spaces
- Missing code block language specifiers (when detectable)
- Convert indented code blocks to fenced code blocks (MD046)
- Fix heading increment issues (MD001)
- Detect duplicate headings (MD024) - reports only, requires manual fix

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


def fix_blank_lines_around_headings(content: str) -> Tuple[str, int]:
    """Add blank lines before and after headings if missing.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []

    for i, line in enumerate(lines):
        is_heading = re.match(r"^#{1,6}\s+", line)
        prev_line = lines[i - 1] if i > 0 else None
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if is_heading:
            # Add blank line before heading if needed
            if (
                prev_line is not None
                and prev_line.strip()
                and not prev_line.strip().startswith("#")
            ):
                if fixed_lines and fixed_lines[-1].strip():
                    fixed_lines.append("")
                    fixes += 1

            fixed_lines.append(line)

            # Add blank line after heading if needed
            if (
                next_line is not None
                and next_line.strip()
                and not next_line.strip().startswith("#")
            ):
                if not (
                    next_line.strip().startswith(("- ", "* ", "+ "))
                    or re.match(r"^\s*\d+\.\s+", next_line)
                ):
                    fixed_lines.append("")
                    fixes += 1
        else:
            fixed_lines.append(line)

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


def fix_missing_code_block_languages(content: str) -> Tuple[str, int]:
    """Add language specifiers to code blocks that are missing them.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Match code block start without language
        if re.match(r"^```\s*$", line):
            # Look ahead to detect language
            language = ""
            j = i + 1
            # Check next 10 lines for language hints
            while j < len(lines) and j < i + 11:
                next_line = lines[j]
                if re.match(r"^```", next_line):
                    # End of code block, stop looking
                    break
                stripped = next_line.strip()
                # Python hints
                if any(
                    keyword in stripped
                    for keyword in ["def ", "import ", "from ", "class ", "->", "if __name__"]
                ):
                    language = "python"
                    break
                # JavaScript hints
                if any(
                    keyword in stripped for keyword in ["const ", "function ", "let ", "var ", "=>"]
                ):
                    language = "javascript"
                    break
                # Go hints
                if any(keyword in stripped for keyword in ["func ", "package ", "import ("]):
                    language = "go"
                    break
                # Bash/shell hints
                if stripped.startswith("#!/") or any(
                    keyword in stripped for keyword in ["export ", "echo ", "$"]
                ):
                    if "python" in stripped or ".py" in stripped:
                        language = "python"
                    elif "bash" in stripped or "sh" in stripped:
                        language = "bash"
                    else:
                        language = "bash"
                    break
                # YAML hints
                if any(keyword in stripped for keyword in ["---", "version:", "apiVersion:"]):
                    language = "yaml"
                    break
                # JSON hints
                if stripped.startswith("{") or stripped.startswith("["):
                    language = "json"
                    break
                j += 1

            if language:
                fixed_lines.append(f"```{language}")
                fixes += 1
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
        i += 1

    return "\n".join(fixed_lines), fixes


def fix_blank_lines_around_code_fences(content: str) -> Tuple[str, int]:
    """Add blank lines before and after fenced code blocks if missing.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []
    in_code_block = False

    for i, line in enumerate(lines):
        is_code_fence = re.match(r"^```", line)
        prev_line = lines[i - 1] if i > 0 else None
        next_line = lines[i + 1] if i + 1 < len(lines) else None

        if is_code_fence and not in_code_block:
            # Starting a code block
            in_code_block = True

            # Check if we need a blank line before
            if (
                prev_line is not None
                and prev_line.strip()
                and not prev_line.strip().startswith("```")
            ):
                # Need blank line before
                fixed_lines.append("")
                fixes += 1

            fixed_lines.append(line)

        elif is_code_fence and in_code_block:
            # Ending a code block
            in_code_block = False
            fixed_lines.append(line)

            # Check if we need a blank line after
            if (
                next_line is not None
                and next_line.strip()
                and not next_line.strip().startswith("```")
            ):
                # Need blank line after
                fixed_lines.append("")
                fixes += 1

        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines), fixes


def fix_indented_code_blocks(content: str) -> Tuple[str, int]:
    """Convert indented code blocks to fenced code blocks.

    Detects blocks of 4+ spaces at the start of lines and converts them to fenced blocks.
    This is a conservative heuristic - only converts when clearly code-like.

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []
    i = 0
    in_indented_block = False
    indented_lines = []

    while i < len(lines):
        line = lines[i]
        # Check if line is indented with 4+ spaces
        is_indented = len(line) > 0 and line[0] == " " and len(line) - len(line.lstrip()) >= 4
        prev_line = lines[i - 1] if i > 0 else None

        # Skip if we're already in a fenced code block
        if prev_line and re.match(r"^```", prev_line):
            fixed_lines.append(line)
            i += 1
            continue

        # Check if we're starting an indented code block
        if is_indented and not in_indented_block:
            # More aggressive detection: check if it looks like code
            stripped_indented = line.strip()
            looks_like_code = (
                stripped_indented.startswith(
                    (
                        "def ",
                        "class ",
                        "import ",
                        "from ",
                        "if ",
                        "for ",
                        "return ",
                        "=",
                        "(",
                        "[",
                        "{",
                        "#",
                        "Args:",
                        "Returns:",
                        "Raises:",
                        "Example:",
                        "Note:",
                    )
                )
                or "(" in stripped_indented
                and ")" in stripped_indented
                or "->" in stripped_indented
                or stripped_indented.endswith(":")
                or re.match(r'^["\']', stripped_indented)  # String literal
                or re.match(r"^\d+\.", stripped_indented)  # Numbered list item that's actually code
            )

            # Check context - if previous line suggests code context
            prev_suggests_code = False
            if prev_line:
                prev_stripped = prev_line.strip()
                prev_suggests_code = (
                    prev_stripped.endswith(":")
                    or prev_stripped.endswith("```")
                    or prev_stripped.startswith("```")
                    or prev_stripped.startswith("#")
                    or "Example" in prev_stripped
                    or "Code" in prev_stripped
                )

            # Also check if we're inside a fenced code block (skip those)
            in_fenced_block = False
            for j in range(i - 1, max(0, i - 20), -1):
                if j < len(lines) and re.match(r"^```", lines[j]):
                    in_fenced_block = True
                    break

            if (
                not in_fenced_block
                and looks_like_code
                and (prev_suggests_code or not prev_line or not prev_line.strip())
            ):
                in_indented_block = True
                indented_lines = [line]
            else:
                fixed_lines.append(line)
                i += 1
                continue

        elif is_indented and in_indented_block:
            # Continue collecting indented lines
            indented_lines.append(line)

        elif in_indented_block:
            # End of indented block
            # Convert if we have at least 1 line (more aggressive)
            if len(indented_lines) >= 1:
                # Try to detect language from content
                language = ""
                code_content = "\n".join([line.strip() for line in indented_lines[:10]])
                if any(
                    keyword in code_content
                    for keyword in ["def ", "import ", "from ", "class ", "->"]
                ):
                    language = "python"
                elif any(
                    keyword in code_content for keyword in ["const ", "function ", "let ", "var "]
                ):
                    language = "javascript"
                elif any(keyword in code_content for keyword in ["func ", "package "]):
                    language = "go"
                elif (
                    any(keyword in code_content for keyword in ["public ", "private ", "class "])
                    and "{" in code_content
                ):
                    language = "java"

                # Add blank line before if needed
                if (
                    fixed_lines
                    and fixed_lines[-1].strip()
                    and not fixed_lines[-1].strip().startswith("```")
                ):
                    fixed_lines.append("")

                # Add fenced code block with language
                # Always add language to avoid MD040 errors
                if not language:
                    language = "text"  # Default to text if we can't detect
                fixed_lines.append(f"```{language}")
                for ind_line in indented_lines:
                    # Keep original indentation
                    fixed_lines.append(ind_line)
                fixed_lines.append("```")
                fixes += 1
            else:
                # Not enough lines or doesn't look like code, keep as-is
                fixed_lines.extend(indented_lines)

            in_indented_block = False
            indented_lines = []
            fixed_lines.append(line)  # Add current line

        else:
            fixed_lines.append(line)

        i += 1

    # Handle case where file ends with indented block
    if in_indented_block and len(indented_lines) >= 1:
        language = ""
        code_content = "\n".join([line.strip() for line in indented_lines[:10]])
        if any(keyword in code_content for keyword in ["def ", "import ", "from "]):
            language = "python"
        if fixed_lines and fixed_lines[-1].strip():
            fixed_lines.append("")
        fixed_lines.append(f"```{language}" if language else "```")
        fixed_lines.extend(indented_lines)
        fixed_lines.append("```")
        fixes += 1

    return "\n".join(fixed_lines), fixes


def fix_heading_increments(content: str) -> Tuple[str, int]:
    """Fix heading levels that increment by more than one level.

    Example: # Heading 1 followed by ### Heading 3 should be ## Heading 2

    Returns: (fixed_content, number_of_fixes)
    """
    fixes = 0
    lines = content.split("\n")
    fixed_lines = []
    prev_heading_level = 0

    for line in lines:
        match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if match:
            current_level = len(match.group(1))
            # If current level is more than prev + 1, fix it
            if current_level > prev_heading_level + 1:
                new_level = prev_heading_level + 1
                fixed_line = "#" * new_level + " " + match.group(2)
                fixed_lines.append(fixed_line)
                fixes += 1
                prev_heading_level = new_level
            else:
                fixed_lines.append(line)
                prev_heading_level = current_level
        else:
            fixed_lines.append(line)
            # Reset heading level if we hit a non-heading line (optional - can be removed)
            # prev_heading_level = 0

    return "\n".join(fixed_lines), fixes


def detect_duplicate_headings(content: str) -> List[Tuple[int, str]]:
    """Detect duplicate headings in the file.

    Returns: List of (line_number, heading_text) tuples for duplicates.
    """
    duplicates = []
    heading_counts = {}
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        match = re.match(r"^#{1,6}\s+(.+)$", line)
        if match:
            heading_text = match.group(1).strip()
            if heading_text in heading_counts:
                heading_counts[heading_text].append(i)
                duplicates.append((i, heading_text))
            else:
                heading_counts[heading_text] = [i]

    return duplicates


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

    content, fixes = fix_blank_lines_around_headings(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Fixed {fixes} blank line(s) around headings")

    content, fixes = fix_code_block_languages(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Added {fixes} code block language specifier(s)")

    content, fixes = fix_missing_code_block_languages(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Added {fixes} missing language specifier(s) to code blocks")

    content, fixes = fix_blank_lines_around_code_fences(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Fixed {fixes} blank line(s) around code fences")

    content, fixes = fix_indented_code_blocks(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Converted {fixes} indented code block(s) to fenced blocks")

    content, fixes = fix_heading_increments(content)
    total_fixes += fixes
    if fixes > 0:
        print(f"  Fixed {fixes} heading increment issue(s)")

    # Fix multiple consecutive blank lines (MD012)
    lines = content.split("\n")
    fixed_lines = []
    prev_blank = False
    blank_fixes = 0
    for line in lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            blank_fixes += 1
            continue  # Skip extra blank lines
        fixed_lines.append(line)
        prev_blank = is_blank
    if blank_fixes > 0:
        content = "\n".join(fixed_lines)
        total_fixes += blank_fixes
        print(f"  Fixed {blank_fixes} multiple blank line(s)")

    # Detect duplicate headings (report only, requires manual fix)
    duplicates = detect_duplicate_headings(content)
    if duplicates:
        print(f"  ⚠ Found {len(duplicates)} duplicate heading(s) (requires manual fix):")
        for line_num, heading_text in duplicates[:5]:  # Show first 5
            print(f"    Line {line_num}: {heading_text}")
        if len(duplicates) > 5:
            print(f"    ... and {len(duplicates) - 5} more")

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
