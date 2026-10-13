#!/usr/bin/env bash
# Turn the reprocess workflow's inline `episode_ids` input (stdin) into one id per line (stdout).
#
# Ids are separated by line breaks or spaces, NEVER by commas: RSS guids in tag-URI form
# (`tag:soundcloud,2010:tracks/2067916744`, `tag:blogger.com,1999:blog-…`) always contain one.
# Splitting on commas cut those into fragments that matched nothing, so 8 #2187 backfill
# episodes could not be sent at all (2026-10-13; 70 of 2,635 prod ids contain a comma, none a
# space). Blank entries are dropped. exit 1 when no id is left.
set -euo pipefail
ids=$(tr -s '[:space:]' '\n' | grep -v '^$' || true)
[ -n "$ids" ] || exit 1
printf '%s\n' "$ids"
