#!/bin/bash
# Quick check: Is the last metrics entry recent?
# Usage: ./.metrics/check_last_entry.sh   (from repo root)

METRICS_FILE=".metrics/rule-adherence.jsonl"

if [ ! -f "$METRICS_FILE" ]; then
    echo "❌ Metrics file not found: $METRICS_FILE"
    exit 1
fi

LAST_ENTRY=$(tail -1 "$METRICS_FILE" 2>/dev/null)
if [ -z "$LAST_ENTRY" ]; then
    echo "❌ Metrics file is empty"
    exit 1
fi

# Extract date from last entry
LAST_DATE=$(echo "$LAST_ENTRY" | python3 -c "import sys, json; print(json.load(sys.stdin).get('date', 'unknown'))" 2>/dev/null)
TODAY=$(date +%Y-%m-%d)

if [ "$LAST_DATE" = "$TODAY" ]; then
    echo "✅ Last entry is from today ($LAST_DATE)"
    echo "Last entry:"
    echo "$LAST_ENTRY" | python3 -m json.tool 2>/dev/null || echo "$LAST_ENTRY"
else
    echo "⚠️  Last entry is from $LAST_DATE (today is $TODAY)"
    echo "Last entry:"
    echo "$LAST_ENTRY" | python3 -m json.tool 2>/dev/null || echo "$LAST_ENTRY"
fi
