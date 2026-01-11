#!/usr/bin/env python3
"""Backup script for .cache directory.

Creates a compressed backup of the ML model cache, excluding temporary files,
locks, and incomplete downloads. Saves to user's home directory by default.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List


def get_backup_dir(output_dir: Path | None = None) -> Path:
    """Get the backup directory path."""
    if output_dir:
        return Path(output_dir).expanduser().resolve()
    # Default: ~/podcast_scraper_cache_backups
    return Path.home() / "podcast_scraper_cache_backups"


def should_exclude(path: Path, root: Path) -> bool:
    """Check if a file/directory should be excluded from backup."""
    # Check path name and structure for exclusions

    # Exclude temporary/incomplete files
    if path.name.endswith(".incomplete"):
        return True
    if path.name.endswith(".lock"):
        return True
    if path.name == ".DS_Store":
        return True

    # Exclude entire .locks directory (HuggingFace lock files)
    if ".locks" in path.parts:
        return True

    # Exclude hidden files/dirs (except .cache itself and README.md)
    if path.name.startswith(".") and path.name not in (".cache", "README.md"):
        return True

    # Exclude git directories if any
    if ".git" in path.parts:
        return True

    # Exclude __pycache__ directories
    if "__pycache__" in path.parts:
        return True

    # Exclude temporary files
    if path.name.endswith(".tmp") or path.name.endswith(".temp"):
        return True

    return False


def get_cache_size(cache_dir: Path) -> int:
    """Calculate total size of cache directory in bytes."""
    total_size = 0
    for path in cache_dir.rglob("*"):
        if path.is_file() and not should_exclude(path, cache_dir):
            try:
                total_size += path.stat().st_size
            except (OSError, PermissionError):
                pass
    return total_size


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def create_backup(
    cache_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
    verbose: bool = False,
) -> Path | None:
    """Create a compressed backup of the cache directory."""
    if not cache_dir.exists():
        print(f"Error: Cache directory does not exist: {cache_dir}", file=sys.stderr)
        return None

    if not cache_dir.is_dir():
        print(f"Error: Cache path is not a directory: {cache_dir}", file=sys.stderr)
        return None

    # Calculate size before backup
    print("Calculating cache size...")
    cache_size = get_cache_size(cache_dir)
    print(f"Cache size: {format_size(cache_size)}")

    # Create backup directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"cache_backup_{timestamp}.tar.gz"
    backup_path = output_dir / backup_name

    if dry_run:
        print(f"\n[DRY RUN] Would create backup: {backup_path}")
        print(f"  Source: {cache_dir}")
        print(f"  Size: {format_size(cache_size)}")
        return None

    print(f"\nCreating backup: {backup_path}")
    print(f"  Source: {cache_dir}")
    print(f"  Size: {format_size(cache_size)}")

    # Create tar.gz archive
    excluded_count = 0
    included_count = 0
    total_size = 0

    try:
        import os

        with tarfile.open(backup_path, "w:gz", compresslevel=6) as tar:
            # Walk through cache directory using os.walk for proper directory filtering
            for root, dirs, files in os.walk(cache_dir):
                root_path = Path(root)

                # Filter out excluded directories (modify dirs in-place to skip them)
                dirs[:] = [d for d in dirs if not should_exclude(root_path / d, cache_dir)]

                # Add files
                for file in files:
                    file_path = root_path / file
                    if should_exclude(file_path, cache_dir):
                        excluded_count += 1
                        if verbose:
                            print(f"  Excluding: {file_path.relative_to(cache_dir)}")
                        continue

                    try:
                        # Create archive name relative to cache_dir (preserves .cache/ structure)
                        # This ensures restore extracts to .cache/ in project root
                        arcname = file_path.relative_to(cache_dir.parent)
                        tar.add(file_path, arcname=str(arcname))
                        included_count += 1
                        total_size += file_path.stat().st_size
                    except Exception as e:
                        excluded_count += 1
                        if verbose:
                            print(f"  Warning: Could not add {file_path}: {e}")

        backup_size = backup_path.stat().st_size
        compression_ratio = (1 - backup_size / cache_size) * 100 if cache_size > 0 else 0

        print("\n✓ Backup created successfully!")
        print(f"  File: {backup_path}")
        print(f"  Size: {format_size(backup_size)} (compressed from {format_size(cache_size)})")
        print(f"  Compression: {compression_ratio:.1f}%")
        print(f"  Files included: {included_count}")
        if excluded_count > 0:
            print(f"  Files excluded: {excluded_count}")

        return backup_path

    except Exception as e:
        print(f"Error creating backup: {e}", file=sys.stderr)
        if backup_path.exists():
            backup_path.unlink()
        return None


def list_backups(backup_dir: Path) -> List[Path]:
    """List all backup files in the backup directory."""
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob("cache_backup_*.tar.gz"), reverse=True)


def cleanup_old_backups(backup_dir: Path, keep: int = 5) -> int:
    """Remove old backups, keeping only the most recent N backups."""
    backups = list_backups(backup_dir)
    if len(backups) <= keep:
        return 0

    removed = 0
    for backup in backups[keep:]:
        try:
            backup.unlink()
            removed += 1
            print(f"  Removed old backup: {backup.name}")
        except Exception as e:
            print(f"  Warning: Could not remove {backup.name}: {e}")

    return removed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Backup .cache directory (ML models) to compressed archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create backup in default location (~/podcast_scraper_cache_backups)
  python scripts/cache/backup_cache.py

  # Create backup in custom location
  python scripts/cache/backup_cache.py --output ~/backups

  # Dry run to see what would be backed up
  python scripts/cache/backup_cache.py --dry-run

  # List existing backups
  python scripts/cache/backup_cache.py --list

  # Clean up old backups (keep only 5 most recent)
  python scripts/cache/backup_cache.py --cleanup 5
        """,
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output directory for backups (default: ~/podcast_scraper_cache_backups)",
    )
    parser.add_argument(
        "--cache-dir",
        "-c",
        type=str,
        default=".cache",
        help="Cache directory to backup (default: .cache)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be backed up without creating backup",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (show excluded files)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List existing backups",
    )
    parser.add_argument(
        "--cleanup",
        type=int,
        metavar="N",
        help="Remove old backups, keeping only the N most recent",
    )

    args = parser.parse_args()

    # Get project root (assume script is in scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cache_dir = project_root / args.cache_dir

    if args.list:
        backup_dir = get_backup_dir(Path(args.output) if args.output else None)
        backups = list_backups(backup_dir)
        if not backups:
            print(f"No backups found in {backup_dir}")
            return 0

        print(f"Found {len(backups)} backup(s) in {backup_dir}:\n")
        for i, backup in enumerate(backups, 1):
            size = backup.stat().st_size
            mtime = datetime.fromtimestamp(backup.stat().st_mtime)
            print(f"  {i}. {backup.name}")
            print(f"     Size: {format_size(size)}")
            print(f"     Date: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        return 0

    if args.cleanup is not None:
        backup_dir = get_backup_dir(Path(args.output) if args.output else None)
        if not backup_dir.exists():
            print(f"No backup directory found: {backup_dir}")
            return 0

        backups = list_backups(backup_dir)
        if len(backups) <= args.cleanup:
            print(f"Only {len(backups)} backup(s) found, keeping all (requested: {args.cleanup})")
            return 0

        print(f"Cleaning up old backups (keeping {args.cleanup} most recent)...")
        removed = cleanup_old_backups(backup_dir, keep=args.cleanup)
        print(f"Removed {removed} old backup(s)")
        return 0

    # Create backup
    output_dir = get_backup_dir(Path(args.output) if args.output else None)
    backup_path = create_backup(cache_dir, output_dir, dry_run=args.dry_run, verbose=args.verbose)

    if backup_path is None and not args.dry_run:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
