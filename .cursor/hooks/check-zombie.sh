#!/bin/bash
# Check for unkillable (UE state) Python processes at session start.
# If found, warn the agent so it can inform the user before doing anything.
input=$(cat)

zombie_count=$(ps aux 2>/dev/null | grep -E '[Pp]ython|[Pp]ytest' | \
  awk '$8 ~ /U/' | grep -v grep | wc -l | tr -d ' ')

if [ "$zombie_count" -gt 0 ]; then
  zombies=$(ps aux | grep -E '[Pp]ython|[Pp]ytest' | awk '$8 ~ /U/' | grep -v grep)
  cat <<EOF
{
  "additional_context": "WARNING: $zombie_count unkillable Python process(es) in UE state detected. These cannot be killed and will cause system instability. Inform the user immediately and recommend rebooting. Processes:\n$zombies"
}
EOF
else
  echo '{"additional_context": "System check: no zombie Python processes found."}'
fi
exit 0
