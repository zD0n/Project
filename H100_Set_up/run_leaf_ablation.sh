#!/bin/bash
# =============================================================================
# The LEAF freeze/unfreeze ablation on Run10: 2^4 = 16 cells.
#
#   bash run_leaf_ablation.sh                        # iemocap, convnext, 16 cells
#   bash run_leaf_ablation.sh cremad
#   bash run_leaf_ablation.sh iemocap cnn_vit
#   bash run_leaf_ablation.sh iemocap convnext NUM_EPOCHS=2   # smoke the wiring
#   bash run_leaf_ablation.sh iemocap convnext TIME_BUDGET_MIN=170
#   CELLS="1111 0000" bash run_leaf_ablation.sh      # just those two
#
# Portal submission (no argument slot) -- edit these instead:
DATASET=${DATASET:-iemocap}      # iemocap | cremad
MODEL=${MODEL:-convnext}         # one classifier: this grid varies the frontend
EXTRA=${EXTRA:-}
RESULTS_ROOT=${RESULTS_ROOT:-"results/LEAF Ablation"}
# =============================================================================
#
# WHAT IS BEING ABLATED. Not "LEAF vs mel" -- run_224.sh already answers that.
# This asks which *parts* of LEAF earn their gradients. Run10 exposes the four
# parameter groups inside the frontend as separate switches:
#
#   LEARN_FILTERS      Gabor centre frequencies and bandwidths
#   LEARN_POOLING      Gaussian lowpass width and bias
#   LEARN_COMPRESSION  PCEN alpha / delta / root
#   LEARN_SMOOTHING    PCEN's EMA coefficient (the "s" in sPCEN)
#
# 1 trains the group with the classifier, 0 leaves it at its initialization --
# the op still runs, so each cell measures the value of *learning* the stage,
# not of the stage existing. Full factorial, so interactions are visible: if
# filters only pay off once pooling is also free to move, one-at-a-time would
# have missed it and reported both as worthless.
#
# The two corners are the ones to read first:
#
#   1111   full LEAF
#   0000   LEAF frozen at initialization -- a fixed Gabor filterbank. This is
#          the honest baseline for the learnability claim, and is NOT the same
#          as FRONTEND=mel: same graph, same PCEN, only the gradients removed.
#
# PCEN=1 THROUGHOUT. Run8/Run10 default the frontend to log compression, which
# builds no PCEN module at all -- and then LEARN_COMPRESSION and LEARN_SMOOTHING
# would have nothing to freeze, collapsing 16 cells into 4 run four times each.
# Run10 refuses that combination outright; this script sets PCEN=1 so the
# question does not arise. It does mean the 1111 cell here is not identical to
# a default Run8 leaf run, which is log-compressed.
#
# ONE CLASSIFIER, NOT THREE. 16 cells is already the length of a long job. The
# frontend question does not need to be crossed with the classifier question --
# run this again with MODEL=cnn_vit if the ranking needs confirming on another
# head.
#
# WHERE THE NUMBERS LAND. All 16 cells append to one file:
#
#   results/LEAF Ablation/LeafTorch_ViT/<corpus>/sweep.csv
#
# with learn_filters / learn_pooling / learn_compression / learn_smoothing as
# the four factor columns and test_uar as the response -- 16 rows that read
# directly as the ablation table. Its own RESULTS_ROOT so these rows do not mix
# into the 224 grid's sweep.csv, where the frontend was constant.
#
# RESUMING. Every finished cell writes its run_tag into sweep.csv, and this
# script skips cells already there. A job killed by the scheduler therefore
# costs at most the cell in flight: resubmit the identical command and it picks
# up where it stopped. TIME_BUDGET_MIN makes that deliberate rather than
# accidental -- see below.

set -e

[ $# -gt 0 ] && { DATASET=$1; shift; }
[ $# -gt 0 ] && { MODEL=$1; shift; }
[ $# -gt 0 ] && EXTRA="$*"

_here=$(cd "$(dirname "$0")" && pwd)

case "$MODEL" in
    vit|cnn_vit|convnext|vit_local|cnn_vit_local|vit_b16) ;;
    *) echo "unknown MODEL '$MODEL' (use vit, cnn_vit, convnext, vit_local, cnn_vit_local or vit_b16)" >&2; exit 1 ;;
esac
case "$DATASET" in
    iemocap|cremad) ;;
    both|all) echo "one corpus per submission -- 32 cells is past a 24h wall clock" >&2; exit 1 ;;
    *) echo "unknown DATASET '$DATASET' (use iemocap or cremad)" >&2; exit 1 ;;
esac

# Apply EXTRA here as well as forwarding it, so the banner reports what will
# actually run rather than the defaults. run_server.sh exports it again, which
# is harmless. Same idiom as run_224.sh.
for _kv in $EXTRA; do
    case "$_kv" in
        *=*) export "$_kv" ;;
        *) echo "ignoring '$_kv' (expected VAR=VAL)" >&2 ;;
    esac
done

# The 16 cells, as bit strings in the order filters/pooling/compression/
# smoothing -- the same order Run10 stamps into its run_tag, so a cell here and
# a row in sweep.csv are matched by eye.
#
# Ordered so the two corners come first: if the job dies early, 1111 and 0000
# are the pair that still says something on their own.
CELLS=${CELLS:-"1111 0000
                0111 1011 1101 1110
                1100 1010 1001 0110 0101 0011
                1000 0100 0010 0001"}

NUM_EPOCHS=${NUM_EPOCHS:-30}
SEED=${SEED:-42}
TARGET_SIZE=${TARGET_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
PATCH=${PATCH:-16}
export RESULTS_ROOT

# Stop starting cells that cannot finish. The scheduler's kill is not graceful:
# a cell cut off at epoch 25 of 30 writes no sweep.csv row, so an hour is
# thrown away. Set this slightly under the real wall limit -- 170 for a 3h job
# -- and the loop stops while there is still room for the cell it would start.
#
# Unlike run_224.sh, where this variable is silently ignored, it is honoured
# here.
TIME_BUDGET_MIN=${TIME_BUDGET_MIN:-0}

# Fragmentation guard, as in run_224.sh -- the partition GPU is shared and this
# job runs in whatever the other tenants leave behind.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Resolve the sweep path against the project folder, not against this script's
# caller. run_server.sh cds to wherever Run10.py sits next to Dataset/, and
# Run10 resolves a relative RESULTS_ROOT from there -- so looking it up
# relative to $PWD would silently never match, and every cell would rerun.
# Same discovery order run_server.sh uses.
_proj=""
for _d in "$PROJECT_DIR" "$PWD" "$_here" "$_here/.."; do
    [ -n "$_d" ] || continue
    if [ -f "$_d/Run10.py" ] && [ -d "$_d/Dataset" ]; then
        _proj=$(cd "$_d" && pwd)
        break
    fi
done
case "$RESULTS_ROOT" in
    /*) _sweep="$RESULTS_ROOT/LeafTorch_ViT/$DATASET/sweep.csv" ;;
    *)  _sweep="${_proj:-.}/$RESULTS_ROOT/LeafTorch_ViT/$DATASET/sweep.csv" ;;
esac

_n=$(echo $CELLS | wc -w)
_i=0
_summary=""
_failed=0
_done=0
_cell_max_min=0     # longest cell seen, the basis for the budget estimate
_budget_hit=0
_t_all=$(date +%s)

echo "############################################################"
echo "# Run10 LEAF ablation | $DATASET | leaf/$MODEL @ ${TARGET_SIZE}x${TARGET_SIZE}"
echo "#   $_n cells of filters/pooling/compression/smoothing, PCEN=1"
echo "#   NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED BATCH_SIZE=$BATCH_SIZE PATCH=$PATCH"
echo "#   RESULTS_ROOT=$RESULTS_ROOT"
[ "$TIME_BUDGET_MIN" -gt 0 ] && echo "#   TIME_BUDGET_MIN=$TIME_BUDGET_MIN (stops before starting a cell it cannot finish)"
echo "#   $EXTRA"
echo "############################################################"

for _cell in $CELLS; do
    _i=$((_i + 1))

    case "$_cell" in
        [01][01][01][01]) ;;
        *) echo "[$_i/$_n] skipping '$_cell' -- expected 4 bits, e.g. 1011" >&2
           _summary="${_summary}  BADCELL ${_cell}\n"
           continue ;;
    esac

    _f=${_cell:0:1}; _p=${_cell:1:1}; _c=${_cell:2:1}; _s=${_cell:3:1}
    _tag="leaf-${MODEL}-ts${TARGET_SIZE}-p${PATCH}-e${NUM_EPOCHS}-s${SEED}-leaf${_cell}"

    # Already finished in a previous submission? Run10 writes run_tag into the
    # sweep row precisely so this check can exist. Cheap grep, and it makes the
    # whole grid restartable without bookkeeping.
    if [ -f "$_sweep" ] && grep -q "$_tag" "$_sweep"; then
        echo
        echo "[$_i/$_n] $_cell -- already in sweep.csv, skipping"
        _summary="${_summary}  done    ${_cell} (earlier job)\n"
        _done=$((_done + 1))
        continue
    fi

    # Budget check before starting, using the longest cell seen rather than the
    # last: a skipped cell returns instantly, and letting that set the
    # expectation would start a real cell with minutes left.
    if [ "$TIME_BUDGET_MIN" -gt 0 ] && [ "$_cell_max_min" -gt 0 ]; then
        _left=$(( TIME_BUDGET_MIN - ($(date +%s) - _t_all) / 60 ))
        if [ "$_left" -lt "$_cell_max_min" ]; then
            echo
            echo "stopping before cell $_cell: ${_left}m of budget left," >&2
            echo "  and the longest cell so far took ${_cell_max_min}m." >&2
            echo "  Resubmit the same command -- finished cells are skipped." >&2
            _budget_hit=1
            break
        fi
    fi

    _t0=$(date +%s)
    echo
    echo "------------------------------------------------------------"
    echo "[$_i/$_n] cell $_cell | filters=$_f pooling=$_p compression=$_c smoothing=$_s"
    echo "------------------------------------------------------------"

    # One process per cell, so a crash in one cannot take the rest down.
    set +e
    bash "$_here/run_server.sh" "$DATASET" Run10.py \
        "FRONTEND=leaf" "MODEL=$MODEL" "PCEN=1" \
        "LEARN_FILTERS=$_f" "LEARN_POOLING=$_p" \
        "LEARN_COMPRESSION=$_c" "LEARN_SMOOTHING=$_s" \
        "TARGET_SIZE=$TARGET_SIZE" "BATCH_SIZE=$BATCH_SIZE" "PATCH=$PATCH" \
        "NUM_EPOCHS=$NUM_EPOCHS" "SEED=$SEED" $EXTRA
    _rc=$?
    set -e

    _mins=$(( ($(date +%s) - _t0) / 60 ))
    [ "$_mins" -gt "$_cell_max_min" ] && _cell_max_min=$_mins

    if [ $_rc -eq 0 ]; then
        _summary="${_summary}  ok      ${_cell} (${_mins}m)\n"
        _done=$((_done + 1))
    else
        _summary="${_summary}  FAILED  ${_cell} (rc=$_rc, ${_mins}m)\n"
        _failed=$((_failed + 1))
        echo "[$_i/$_n] FAILED (rc=$_rc) -- continuing with the rest" >&2
    fi
done

echo
echo "############################################################"
echo "# done in $(( ($(date +%s) - _t_all) / 60 )) min -- $_done of $_n cells complete, $_failed failed"
printf "$_summary"
echo "#"
echo "# table: $_sweep"
echo "#"
echo "# $_n rows, four factor columns:"
echo "#   learn_filters  learn_pooling  learn_compression  learn_smoothing"
echo "# Compare on test_uar. 1111 is full LEAF, 0000 is the frozen frontend --"
echo "# the gap between those two is the whole learnability claim, and the"
echo "# single-1 rows say which stage is carrying it."
echo "############################################################"

if [ "$_budget_hit" = "1" ]; then
    echo
    echo "Stopped on TIME_BUDGET_MIN with cells left. Resubmit to continue." >&2
    exit 0
fi

[ "$_failed" -eq 0 ] || exit 1
