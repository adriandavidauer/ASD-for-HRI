#!/bin/bash
# Copy the benchmark datasets to the cluster (#93): only the files the benchmark reads with
# --no_download, in the layout cluster/benchmark.slr mounts as /app/data.
#
#   cluster/sync_data.sh                      # all four datasets (about 58 GB)
#   cluster/sync_data.sh unitalk wasd         # only these
#   CLUSTER_USER=me SRC=/Data/data cluster/sync_data.sh ava
#
# Run it on the PC that holds the data. It can be stopped and started again: rsync only copies
# what is missing or changed.
set -euo pipefail

SRC=${SRC:-/Data/data}                                  # local data root
HOST=${HOST:-lnode01}
CLUSTER_USER=${CLUSTER_USER:-$USER}
DEST=${DEST:-/mnt/beegfs/data/asd4hri}                  # = DATA in cluster/benchmark.slr

cd "$SRC"                                               # before the globs below expand
# relative to SRC; "./" marks where the path on the cluster starts (rsync -R)
unitalk=(./videos/val/*.mp4 ./csv/val_orig.csv)
wasd=(./wasd/clips ./wasd/csv/val_orig.csv ./wasd/csv/wasd_metadata.csv)
easycom=(./easycom/csv ./easycom/Main/Video_Compressed ./easycom/Main/Face_Bounding_Boxes
         ./easycom/Main/Speech_Transcriptions)
ava=(./ava/videos ./ava/annotations)

datasets=("$@")
[ ${#datasets[@]} -eq 0 ] && datasets=(unitalk wasd easycom ava)

ssh "$CLUSTER_USER@$HOST" "mkdir -p '$DEST'"
for d in "${datasets[@]}"; do
  case $d in unitalk|wasd|easycom|ava) ;; *) echo "unknown dataset: $d" >&2; exit 1 ;; esac
  declare -n paths=$d
  echo "=== $d"
  rsync -aR --partial --info=progress2 "${paths[@]}" "$CLUSTER_USER@$HOST:$DEST/"
done
echo "done: $DEST on $HOST"
