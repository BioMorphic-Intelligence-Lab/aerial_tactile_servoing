#!/bin/bash

# ==== LOAD CONFIG FROM .env ====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found. Copy .env.example to .env and fill in your values." >&2
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

# Expand ~ in SSH_KEY if present
SSH_KEY="${SSH_KEY/#\~/$HOME}"

LOG_FILE="$LOCAL_DIR/log_ros2bag_sync.txt"

# ==== CREATE LOCAL DIR IF NEEDED ====
mkdir -p "$LOCAL_DIR"

# ==== GET DIRECTORY LISTS ====
REMOTE_DIRS=$(ssh -i $SSH_KEY $REMOTE_USER@$REMOTE_HOST "ls -1 $REMOTE_DIR")
LOCAL_DIRS=$(ls -1 "$LOCAL_DIR")

DOWNLOADED=()

# ==== COMPARE AND SYNC ====
for dir in $REMOTE_DIRS; do
    if [[ -d "$LOCAL_DIR/$dir" ]]; then
        continue
    fi

    echo "Downloading new directory: $dir"
    rsync -avz \
        -e "ssh -i $SSH_KEY" \
        "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR/$dir" \
        "$LOCAL_DIR/" >> "$LOG_FILE" 2>&1

    DOWNLOADED+=("$dir")
done

# ==== SUMMARY ====
if [ ${#DOWNLOADED[@]} -eq 0 ]; then
    echo "No new directories found."
else
    echo "Downloaded directories:"
    for d in "${DOWNLOADED[@]}"; do
        echo " - $d"
    done
fi

echo "Sync completed at $(date)"
