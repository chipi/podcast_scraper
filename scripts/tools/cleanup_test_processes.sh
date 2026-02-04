#!/bin/bash
# Cleanup leftover test processes (pytest workers, Python processes, etc.)
# This script helps clean up processes that may have been left running after tests

set -euo pipefail

echo "🔍 Checking for leftover Python/test processes..."
echo ""

# Find all Python processes
PYTHON_PROCS=$(ps aux | grep -E "python|pytest" | grep -v grep | grep -v "$0" || true)

if [ -z "$PYTHON_PROCS" ]; then
    echo "✅ No Python/test processes found running"
    exit 0
fi

echo "Found Python/test processes:"
echo "$PYTHON_PROCS"
echo ""

# Count processes
PROC_COUNT=$(echo "$PYTHON_PROCS" | wc -l | tr -d ' ')
echo "Total processes found: $PROC_COUNT"
echo ""

# Ask for confirmation
read -p "Kill these processes? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. No processes killed."
    exit 0
fi

# Kill processes
echo "Killing processes..."
echo ""

# Kill pytest processes first
pkill -f pytest 2>/dev/null || true

# Kill Python processes related to podcast_scraper
pkill -f "python.*podcast_scraper" 2>/dev/null || true

# Kill pytest-xdist worker processes (gw0, gw1, etc.)
pkill -f "gw[0-9]" 2>/dev/null || true

# Wait a moment
sleep 1

# Check if any remain
REMAINING=$(ps aux | grep -E "python|pytest" | grep -v grep | grep -v "$0" || true)

if [ -z "$REMAINING" ]; then
    echo "✅ All processes cleaned up successfully"
else
    echo "⚠️  Some processes may still be running:"
    echo "$REMAINING"
    echo ""
    echo "You may need to kill them manually:"
    echo "  pkill -9 -f pytest"
    echo "  pkill -9 python"
fi
