#!/usr/bin/env bash
# Shared helpers for corpus snapshot manifest scripts (RFC-084 / ADR-092).

corpus_snapshot_repo_root() {
  if [[ -n "${CORPUS_SNAPSHOT_REPO_ROOT:-}" ]]; then
    printf '%s\n' "$CORPUS_SNAPSHOT_REPO_ROOT"
    return 0
  fi
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
  printf '%s\n' "$here"
}

corpus_snapshot_require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command not found: $cmd" >&2
    return 1
  fi
}

corpus_snapshot_read_reader_range() {
  local root min max
  root="$(corpus_snapshot_repo_root)"
  corpus_snapshot_require_cmd jq || return 1
  min="$(jq -r '.supported_corpus_format_version_min' "$root/config/corpus_snapshot_reader_support.json")"
  max="$(jq -r '.supported_corpus_format_version_max' "$root/config/corpus_snapshot_reader_support.json")"
  if [[ -z "$min" || "$min" == "null" || -z "$max" || "$max" == "null" ]]; then
    echo "ERROR: invalid reader support config under $root/config/corpus_snapshot_reader_support.json" >&2
    return 1
  fi
  printf '%s %s\n' "$min" "$max"
}

corpus_snapshot_read_producer_format() {
  local root schema corpus
  root="$(corpus_snapshot_repo_root)"
  corpus_snapshot_require_cmd jq || return 1
  schema="$(jq -r '.schema_version' "$root/config/corpus_snapshot_format.json")"
  corpus="$(jq -r '.corpus_format_version' "$root/config/corpus_snapshot_format.json")"
  if [[ -z "$schema" || "$schema" == "null" || -z "$corpus" || "$corpus" == "null" ]]; then
    echo "ERROR: invalid producer format config under $root/config/corpus_snapshot_format.json" >&2
    return 1
  fi
  printf '%s %s\n' "$schema" "$corpus"
}

corpus_snapshot_version_supported() {
  local version="$1" min="$2" max="$3"
  if [[ "$version" =~ ^[0-9]+$ ]] && (( version >= min && version <= max )); then
    return 0
  fi
  return 1
}
