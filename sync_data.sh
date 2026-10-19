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

# Per-run itemized record of what rsync actually transferred, used to build the
# "Downloaded" report below. Kept separate from the terminal/log output so the
# live progress view stays clean. Cleaned up on exit.
RUN_LOG="$(mktemp)"
trap 'rm -f "$RUN_LOG"' EXIT

# Trailing slashes: copy the CONTENTS of the remote data dir into the local one.
# -a                 archive (recursive, preserves times/perms) — files & dirs
# --partial          keep partly-transferred files so the next run resumes them
# --info=...         concise overall progress on the terminal
# --log-file[-format] write one 'XFER|<itemize>|<path>' line per updated item to
#                    RUN_LOG (independent of stdout) so we can report precisely
#                    what was downloaded.
rsync -a --partial --human-readable \
    --info=progress2,name0,stats1 \
    --log-file="$RUN_LOG" --log-file-format='XFER|%i|%n' \
    "${RSYNC_EXTRA[@]}" \
    -e "ssh -i $SSH_KEY" \
    "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DATA_DIR/" \
    "$LOCAL_DATA_DIR/" 2>&1 | tee -a "$LOG_FILE"

# ==== REPORT WHAT WAS DOWNLOADED ====
# Parse the per-run log. Each transferred item was logged (behind rsync's own
# timestamp/pid prefix) as 'XFER|<itemize>|<path>'. A leading '>' in the
# itemize code means the item was received from the drone, i.e. downloaded.
declare -A folder_got       # downloaded-file count per top-level folder
downloaded_files=()         # every downloaded file, relative to the data dir
top_level_files=()          # downloaded files sitting directly in the data dir

while IFS= read -r entry; do
    entry="${entry#XFER|}"              # -> '<code>|<path>'
    code="${entry%%|*}"
    path="${entry#*|}"
    [[ "$code" == '>'* ]] || continue           # only received (downloaded) files
    [[ -n "$path" && "$path" != */ ]] || continue   # skip dirs / blank lines
    downloaded_files+=("$path")
    if [[ "$path" == */* ]]; then
        top="${path%%/*}"
        folder_got["$top"]=$(( ${folder_got["$top"]:-0} + 1 ))
    else
        top_level_files+=("$path")
    fi
done < <(grep -o 'XFER|.*' "$RUN_LOG" || true)

# Build the report. A top-level folder is collapsed to a single line when every
# file it now holds locally came from this run ("entirely downloaded");
# otherwise the specific files that were downloaded under it are listed. Files
# that live directly in the data dir are always listed individually.
report=()
if (( ${#folder_got[@]} )); then
    for top in $(printf '%s\n' "${!folder_got[@]}" | sort); do
        want=${folder_got["$top"]}
        have=$(find "$LOCAL_DATA_DIR/$top" -type f 2>/dev/null | wc -l)
        if (( have == want )); then
            report+=("$top/   (entire folder — $want file(s))")
        else
            for p in "${downloaded_files[@]}"; do
                [[ "$p" == "$top/"* ]] && report+=("$p")
            done
        fi
    done
fi
if (( ${#top_level_files[@]} )); then
    for p in "${top_level_files[@]}"; do report+=("$p"); done
fi

echo
if (( ${#report[@]} == 0 )); then
    echo "Nothing to download — local data already up to date."
else
    if [[ "$MODE" == "dry-run" ]]; then
        echo "Would download:"
    else
        echo "Downloaded:"
    fi
    for line in "${report[@]}"; do
        echo "  $line"
    done
fi

echo
if [[ "$MODE" == "dry-run" ]]; then
    echo "Dry run only — nothing was written. Drop --dry-run to transfer."
else
    echo "Sync completed at $(date)."
    echo "Data on the drone is UNTOUCHED. Once you've confirmed the files above,"
    echo "you can delete them on the drone over SSH yourself."
fi
