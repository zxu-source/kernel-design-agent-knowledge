#!/usr/bin/env bash
# Poll a detached 8-card notebook job without keeping its websocket alive.
# Usage: monitor_8card_remote_job.sh <remote-rc-file> <remote-log-file> <local-evidence-dir>
set -Eeuo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 <remote-rc-file> <remote-log-file> <local-evidence-dir>" >&2
  exit 2
fi

remote_rc=$1
remote_log=$2
evidence_dir=$3
root=/home/kirin_14379/projects/ai4qz
ctl="$root/scripts/910b8ctl"
mkdir -p "$evidence_dir"

while :; do
  stamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if "$ctl" run "if test -f '$remote_rc'; then printf 'DONE '; cat '$remote_rc'; tail -240 '$remote_log'; else printf 'RUNNING\\n'; fi" \
      >>"$evidence_dir/monitor.log" 2>&1; then
    if grep -q '^DONE ' "$evidence_dir/monitor.log"; then
      "$ctl" run "cat '$remote_log'" >"$evidence_dir/correctness.log" 2>&1 || true
      printf '%s completed\n' "$stamp" >>"$evidence_dir/monitor.log"
      exit 0
    fi
  else
    printf '%s poll transport failure; retrying\n' "$stamp" >>"$evidence_dir/monitor.log"
  fi
  sleep 60
done
