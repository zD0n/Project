#!/bin/bash
# CREMA-D at 224x224, batch 32: {leaf, mel} x {cnn_vit, convnext} = 4 runs.
#
# Why 224: ConvNeXt's four stages downsample by 32, so 224 gives the 7x7 map
# before global pooling that the architecture was designed around, against 2x2
# at run_server.sh's default 64. It is also the only size where
# CONVNEXT_PRETRAINED=1 is worth trying, since the ImageNet weights were
# trained there.
#
# Separate from the run_server.sh `all` sweep on purpose -- that one holds
# TARGET_SIZE=64 across every run so its six results stay comparable to each
# other. These four are the bigger-input arm.
#
# Each combination is its own run_server.sh call, so one failure cannot take
# the others down; the summary at the end is the authority on what completed.

set -e

DATASET=cremad
_here=$(cd "$(dirname "$0")" && pwd)

export NUM_EPOCHS=${NUM_EPOCHS:-30}
export SEED=${SEED:-42}
export TARGET_SIZE=${TARGET_SIZE:-224}
export BATCH_SIZE=${BATCH_SIZE:-32}

_combos="leaf:cnn_vit leaf:convnext mel:cnn_vit mel:convnext"
_n=$(echo $_combos | wc -w)
_i=0
_summary=""
_failed=0
_t_all=$(date +%s)

echo "############################################################"
echo "# $DATASET @ ${TARGET_SIZE}x${TARGET_SIZE}, batch $BATCH_SIZE : $_n runs"
echo "#   NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED"
echo "############################################################"

for _c in $_combos; do
    _fe=${_c%%:*}
    _md=${_c##*:}
    _i=$((_i + 1))
    _t0=$(date +%s)

    echo
    echo "------------------------------------------------------------"
    echo "[$_i/$_n] $DATASET | FRONTEND=$_fe MODEL=$_md @ $TARGET_SIZE"
    echo "------------------------------------------------------------"

    set +e
    bash "$_here/run_server.sh" "$DATASET" Run8.py \
        "FRONTEND=$_fe" "MODEL=$_md" \
        "TARGET_SIZE=$TARGET_SIZE" "BATCH_SIZE=$BATCH_SIZE" \
        "NUM_EPOCHS=$NUM_EPOCHS" "SEED=$SEED" "$@"
    _rc=$?
    set -e
    _mins=$(( ($(date +%s) - _t0) / 60 ))

    if [ $_rc -eq 0 ]; then
        _summary="${_summary}  ok      ${_fe}/${_md}@${TARGET_SIZE} (${_mins}m)\n"
    else
        _summary="${_summary}  FAILED  ${_fe}/${_md}@${TARGET_SIZE} (rc=$_rc, ${_mins}m)\n"
        _failed=$((_failed + 1))
        echo "[$_i/$_n] FAILED (rc=$_rc) -- continuing with the rest" >&2
    fi
done

echo
echo "############################################################"
echo "# done in $(( ($(date +%s) - _t_all) / 60 )) min"
printf "$_summary"
echo "#"
echo "# results: results/LeafTorch_ViT/cremad/sweep.csv  (leaf)"
echo "#          results/MelTorch_ViT/cremad/sweep.csv   (mel)"
echo "# rows carry target_size, so the 224 runs are distinguishable from 64"
echo "############################################################"

[ "$_failed" -eq 0 ] || exit 1
