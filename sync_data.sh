#!/bin/bash
#
# Download flight data from the drone onto the ground station.
#
# Pulls the drone's data/ tree (px4_logs, rosbags, serial_logs, tactip_images,
# and any other subfolders) into the local repo's data/ folder. Uses rsync so
# that only files missing or incomplete on the GCS are transferred, and
# interrupted transfers resume instead of being silently treated as "done".
#
# This script NEVER deletes anything on the drone. Once a run reports a clean,
# fully-verified sync you can delete the logs on the drone yourself over SSH.
#
# Usage:
#   ./sync_data.sh            # sync (size+mtime quick check, resumable)
#   ./sync_data.sh --verify   # same, but re-checksum every file (slower, paranoid)
#   ./sync_data.sh --dry-run   # show what would transfer, change nothing

set -euo pipefail

# ==== LOAD CONFIG FROM .env ====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .env.example to .env and fill in your values." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

# Expand ~ in SSH_KEY if present
SSH_KEY="${SSH_KEY/#\~/$HOME}"

# Base data dirs. Prefer the new *_DATA_DIR vars; fall back to the old names so
# an existing .env keeps working.
REMOTE_DATA_DIR="${REMOTE_DATA_DIR:-${REMOTE_DIR:-}}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${LOCAL_DIR:-$SCRIPT_DIR/data}}"

if [[ -z "$REMOTE_DATA_DIR" ]]; then
    echo "Error: set REMOTE_DATA_DIR in $ENV_FILE (path to data/ on the drone)." >&2
    exit 1
fi

# ==== PARSE ARGS ====
RSYNC_EXTRA=()
MODE="sync"
for arg in "$@"; do
    case "$arg" in
        --verify)  RSYNC_EXTRA+=(--checksum) ;;
        --dry-run) RSYNC_EXTRA+=(--dry-run); MODE="dry-run" ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

LOG_FILE="$LOCAL_DATA_DIR/log_data_sync.txt"
mkdir -p "$LOCAL_DATA_DIR"

echo "Syncing drone data ($MODE)"
echo "  from: $REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_DIR/"
echo "  to:   $LOCAL_DATA_DIR/"
echo

# Trailing slashes: copy the CONTENTS of the remote data dir into the local one.
# -a          archive (recursive, preserves times/perms) — handles files & dirs
# --partial   keep partly-transferred files so the next run resumes them
# --info=...  concise per-file + overall progress
rsync -a --partial --human-readable \
    --info=progress2,name0,stats1 \
    "${RSYNC_EXTRA[@]}" \
    -e "ssh -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_DIR/" \
    "$LOCAL_DATA_DIR/" 2>&1 | tee -a "$LOG_FILE"

echo
if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry run only — nothing was written. Drop --dry-run to transfer."
else
    echo "Sync completed at $(date)."
    echo "Data on the drone is UNTOUCHED. Once you've confirmed the files above,"
    echo "you can delete them on the drone over SSH yourself."
fi
