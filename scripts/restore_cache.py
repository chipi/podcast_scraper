#!/usr/bin/env python3
"""Restore script for .cache directory.

Restores a compressed backup of the ML model cache to the current project's .cache directory.
Looks for backups in the user's home directory by default.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def get_backup_dir(backup_dir: Path | None = None) -> Path:
    """Get the backup directory path."""
    if backup_dir:
        return Path(backup_dir).expanduser().resolve()
    # Default: ~/podcast_scraper_cache_backups
    return Path.home() / "podcast_scraper_cache_backups"


def list_backups(backup_dir: Path) -> List[Path]:
    """List all backup files in the backup directory."""
    if not backup_dir.exists():
        return []
    return sorted(backup_dir.glob("cache_backup_*.tar.gz"), reverse=True)


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_backup_info(backup_path: Path) -> dict:
    """Get information about a backup file."""
    stat = backup_path.stat()
    return {
        "path": backup_path,
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime),
        "name": backup_path.name,
    }


def select_backup(
    backup_dir: Path, backup_name: Optional[str] = None, auto_select_latest: bool = False
) -> Path | None:
    """Select a backup file, either by name, automatically (latest), or interactively."""
    backups = list_backups(backup_dir)

    if not backups:
        print(f"No backups found in {backup_dir}", file=sys.stderr)
        return None

    if backup_name:
        # Try to find exact match
        for backup in backups:
            if backup.name == backup_name:
                return backup

        # Try partial match
        matching = [b for b in backups if backup_name in b.name]
        if len(matching) == 1:
            return matching[0]
        elif len(matching) > 1:
            print(f"Multiple backups match '{backup_name}':", file=sys.stderr)
            for i, backup in enumerate(matching, 1):
                info = get_backup_info(backup)
                print(f"  {i}. {backup.name}", file=sys.stderr)
            return None
        else:
            print(f"No backup found matching '{backup_name}'", file=sys.stderr)
            return None

    # Auto-select latest backup (for non-interactive use)
    if auto_select_latest:
        latest = backups[0]  # Already sorted by date (newest first)
        info = get_backup_info(latest)
        print(f"Auto-selecting most recent backup: {info['name']}")
        print(f"  Size: {format_size(info['size'])}")
        print(f"  Date: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        return latest

    # Interactive selection
    print(f"Available backups in {backup_dir}:\n")
    for i, backup in enumerate(backups, 1):
        info = get_backup_info(backup)
        print(f"  {i}. {info['name']}")
        print(f"     Size: {format_size(info['size'])}")
        print(f"     Date: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
        print()

    try:
        choice = input("Select backup number (or 'q' to quit): ").strip()
        if choice.lower() == "q":
            return None

        index = int(choice) - 1
        if 0 <= index < len(backups):
            return backups[index]
        else:
            print(f"Invalid selection: {choice}", file=sys.stderr)
            return None
    except (ValueError, KeyboardInterrupt, EOFError):
        print("\nCancelled.", file=sys.stderr)
        return None


def verify_backup(backup_path: Path) -> bool:
    """Verify that a backup file is valid and can be opened."""
    if not backup_path.exists():
        print(f"Error: Backup file does not exist: {backup_path}", file=sys.stderr)
        return False

    if not backup_path.is_file():
        print(f"Error: Backup path is not a file: {backup_path}", file=sys.stderr)
        return False

    try:
        # Try to open and read the tar file
        with tarfile.open(backup_path, "r:gz") as tar:
            # Check if it's a valid tar file
            members = tar.getmembers()
            if not members:
                print("Warning: Backup file appears to be empty", file=sys.stderr)
                return False

            # Check for .cache structure
            has_cache = any(m.name.startswith(".cache/") for m in members)
            if not has_cache:
                print("Warning: Backup doesn't appear to contain .cache directory", file=sys.stderr)
                return False

        return True
    except tarfile.ReadError as e:
        print(f"Error: Backup file is corrupted or not a valid tar.gz: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error: Could not verify backup: {e}", file=sys.stderr)
        return False


def restore_backup(
    backup_path: Path,
    target_cache_dir: Path,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = False,
) -> bool:
    """Restore a backup to the target cache directory."""
    # Verify backup first
    if not verify_backup(backup_path):
        return False

    # Check if target directory exists
    if target_cache_dir.exists():
        if not force:
            print(f"Warning: Target directory already exists: {target_cache_dir}")
            print("  Use --force to overwrite, or manually remove it first.")
            return False

        if not dry_run:
            print(f"Removing existing cache directory: {target_cache_dir}")
            try:
                shutil.rmtree(target_cache_dir)
            except Exception as e:
                print(f"Error: Could not remove existing cache directory: {e}", file=sys.stderr)
                return False

    # Get backup info
    info = get_backup_info(backup_path)
    print(f"\nRestoring backup: {info['name']}")
    print(f"  Size: {format_size(info['size'])}")
    print(f"  Date: {info['mtime'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Target: {target_cache_dir}")

    if dry_run:
        print("\n[DRY RUN] Would extract backup to target directory")
        return True

    # Create target directory
    target_cache_dir.mkdir(parents=True, exist_ok=True)

    # Extract backup
    print("\nExtracting backup...")
    extracted_count = 0
    total_size = 0

    try:
        with tarfile.open(backup_path, "r:gz") as tar:
            # Get all members
            members = tar.getmembers()

            for member in members:
                # Only extract .cache/ paths (skip other files)
                if not member.name.startswith(".cache/"):
                    if verbose:
                        print(f"  Skipping: {member.name}")
                    continue

                try:
                    # Strip .cache/ prefix to get relative path
                    # Example: .cache/whisper/base.en.pt -> whisper/base.en.pt
                    relative_path = member.name[len(".cache/") :]

                    if not relative_path:  # Skip .cache/ directory itself
                        continue

                    # Build target path
                    target_path = target_cache_dir / relative_path

                    # Extract to target location
                    if member.isdir():
                        target_path.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        # Ensure parent directory exists
                        target_path.parent.mkdir(parents=True, exist_ok=True)

                        # Extract file
                        with tar.extractfile(member) as source:
                            if source:
                                with open(target_path, "wb") as target:
                                    shutil.copyfileobj(source, target)

                        # Preserve permissions
                        target_path.chmod(member.mode)

                        extracted_count += 1
                        total_size += member.size

                        if verbose and extracted_count % 100 == 0:
                            print(f"  Extracted {extracted_count} files...")

                except Exception as e:
                    print(f"  Warning: Could not extract {member.name}: {e}", file=sys.stderr)

        print("\n✓ Restore completed successfully!")
        print(f"  Files extracted: {extracted_count}")
        print(f"  Total size: {format_size(total_size)}")
        print(f"  Target: {target_cache_dir}")

        # Verify the restore
        if target_cache_dir.exists() and any(target_cache_dir.iterdir()):
            print("  ✓ Cache directory exists and contains files")
        else:
            print("  ⚠ Warning: Cache directory is empty after restore", file=sys.stderr)
            return False

        return True

    except Exception as e:
        print(f"Error restoring backup: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        # Clean up partial restore
        if target_cache_dir.exists():
            print("Cleaning up partial restore...", file=sys.stderr)
            try:
                shutil.rmtree(target_cache_dir)
            except Exception:
                pass
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Restore .cache directory (ML models) from compressed backup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-select most recent backup and restore to .cache
  python scripts/restore_cache.py

  # Restore most recent backup to custom directory
  python scripts/restore_cache.py --target my-restored-cache

  # Restore a specific backup by name
  python scripts/restore_cache.py --backup cache_backup_20250108-120000.tar.gz

  # Restore specific backup to custom directory
  python scripts/restore_cache.py --backup 20250108 --target my-cache

  # Dry run to see what would be restored
  python scripts/restore_cache.py --backup 20250108 --dry-run

  # Force overwrite existing directory
  python scripts/restore_cache.py --target my-cache --force

  # Restore from custom backup location
  python scripts/restore_cache.py --backup-dir ~/my_backups
        """,
    )
    parser.add_argument(
        "--backup-dir",
        "-d",
        type=str,
        help="Backup directory to search (default: ~/podcast_scraper_cache_backups)",
    )
    parser.add_argument(
        "--backup",
        "-b",
        type=str,
        help=(
            (
                "Backup file name to restore (partial match supported). "
                "If omitted, automatically selects the most recent backup "
                "when running non-interactively, "
                "or prompts for selection when running interactively."
            )
        ),
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        help="Target cache directory (default: .cache in project root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be restored without actually restoring",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite existing .cache directory if it exists",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output (show extraction progress)",
    )

    args = parser.parse_args()

    # Get project root (assume script is in scripts/ directory)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    # Get backup directory
    backup_dir = get_backup_dir(Path(args.backup_dir) if args.backup_dir else None)

    # Get target cache directory
    if args.target:
        target_cache_dir = Path(args.target).expanduser().resolve()
    else:
        target_cache_dir = project_root / ".cache"

    # Select backup
    # Auto-select latest if running non-interactively (backup not specified and stdin is not a tty)
    auto_select = args.backup is None and not sys.stdin.isatty()
    backup_path = select_backup(backup_dir, args.backup, auto_select_latest=auto_select)
    if backup_path is None:
        return 1

    # Restore backup
    success = restore_backup(
        backup_path,
        target_cache_dir,
        dry_run=args.dry_run,
        force=args.force,
        verbose=args.verbose,
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
