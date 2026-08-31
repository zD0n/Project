#!/bin/bash
# =============================================================================
# The full 224x224 grid on Run8: {mel, leaf} x {vit, cnn_vit, convnext}.
#
#   6 runs per corpus, NUM_EPOCHS=30, SEED=42, TARGET_SIZE=224, BATCH_SIZE=32.
#
#   bash run_224.sh                      # iemocap, all 6
#   bash run_224.sh cremad
#   bash run_224.sh both                 # 12 runs
#   bash run_224.sh iemocap mel          # the 3 mel runs only
#   bash run_224.sh cremad leaf NUM_EPOCHS=2   # smoke test the wiring
#
# Portal submission (no argument slot) -- edit these three lines instead:
DATASET=${DATASET:-iemocap}      # iemocap | cremad | both
FRONTEND=${FRONTEND:-both}       # mel | leaf | both  (space-separated ok)
EXTRA=${EXTRA:-}                 # e.g. "NUM_EPOCHS=2 CONVNEXT_PRETRAINED=1"
RESULTS_ROOT=${RESULTS_ROOT:-"results/Ablation Study"}
# =============================================================================
#
# WHY THIS FILE EXISTS. run_<corpus>_224.sh already runs 224, but only
# {leaf, mel} x {cnn_vit, convnext} and only one corpus per file. This one adds
# the plain `vit` arm -- patch embedding straight off the frontend output, no
# CNN stem -- and folds the corpus choice into a parameter, so the whole grid
# is one submission.
#
# THE ARMS. All three read the same 224x224 frontend output; they differ only
# in what turns it into a vector.
#
#   vit        patchify 224 into 16x16 patches, ViT from scratch. The control
#              for cnn_vit: whether the CNN stem is earning its parameters.
#   cnn_vit    conv stem, then the ViT over its feature map. Run8's default.
#   convnext   ConvNeXt-tiny, no attention at all. At 224 its four stages leave
#              the 7x7 map the architecture was designed around, against 2x2 at
#              run_server.sh's default TARGET_SIZE=64 -- which is the reason
#              this grid is at 224 and not 64.
#
# 224 is also the only size where CONVNEXT_PRETRAINED=1 or VIT_PRETRAINED=1 are
# worth trying, since those ImageNet weights were trained there. Both are off
# by default: pass them in EXTRA.
#
# PATCH=16, NOT THE 8 RUN8 DEFAULTS TO. The ViT arms cut the frontend output
# into PATCH x PATCH squares, so the token count is (TARGET_SIZE/PATCH)^2 and
# attention cost is its square. At 224:
#
#   PATCH=8    784 patches   602 MiB per attention map per layer   ~3.5 GiB
#   PATCH=16   196 patches    38 MiB per attention map per layer   ~0.2 GiB
#
# 602 MiB is the exact allocation that killed the first two arms of the
# 2026-08-23 iemocap run. PATCH=8 is right at TARGET_SIZE=64, where it gives 64
# patches, but it does not transfer: at 224 it makes the sequence 12x longer
# for a frontend output whose real resolution has not increased. 16 is the
# standard ViT-at-224 tokenization and keeps the token count in the same range
# as the 64-input runs. The CNN stem in `cnn_vit` is stride-1, so it does not
# reduce this either -- both ViT arms are affected identically.
#
# Set PATCH=8 to reproduce what run_<corpus>_224.sh did. ConvNeXt ignores
# PATCH, so that arm is the same either way.
#
# WHERE THE NUMBERS LAND. This grid is an ablation -- two frontends crossed
# with three classifiers, everything else held fixed -- so it writes under
# "results/Ablation Study/" rather than into the main results/ tree, where its
# rows would be appended to the same sweep.csv as the TARGET_SIZE=64 sweep and
# the CV runs. Nothing else is moved: the frontend and dataset subfolders are
# unchanged underneath, so
#
#   results/Ablation Study/MelTorch_ViT/iemocap/sweep.csv
#   results/Ablation Study/LeafTorch_ViT/iemocap/sweep.csv
#
# hold three rows each -- one per classifier -- which is the ablation table.
# Export RESULTS_ROOT to send them somewhere else; it is read by Run8.py.
#
# Note the space in the folder name: RESULTS_ROOT is exported into the
# environment rather than passed through EXTRA, because EXTRA is word-split on
# spaces and "Study" would be dropped as a malformed VAR=VAL pair.
#
# Each combination is its own run_server.sh call, so one failure cannot take
# the others down; the summary at the end is the authority on what completed.

set -e

[ $# -gt 0 ] && { DATASET=$1; shift; }
[ $# -gt 0 ] && { FRONTEND=$1; shift; }
[ $# -gt 0 ] && EXTRA="$*"

_here=$(cd "$(dirname "$0")" && pwd)

case "$DATASET" in
    both|all) DATASETS="iemocap cremad" ;;
    *)        DATASETS="$DATASET" ;;
esac
case "$FRONTEND" in
    both|all) FRONTENDS="mel leaf" ;;
    *)        FRONTENDS="$FRONTEND" ;;
esac
for _fe in $FRONTENDS; do
    case "$_fe" in
        mel|leaf) ;;
        *) echo "unknown FRONTEND '$_fe' (use mel, leaf or both)" >&2; exit 1 ;;
    esac
done

MODELS=${MODELS:-"vit cnn_vit convnext"}

# Apply EXTRA here, not only by forwarding it, so the banner below reports what
# will actually run. Without this, `run_224.sh iemocap both NUM_EPOCHS=2`
# printed the default 30 while every run trained for 2 -- a log that misstates
# its own experiment. run_server.sh exports EXTRA again, which is harmless.
for _kv in $EXTRA; do
    case "$_kv" in
        *=*) export "$_kv" ;;
        *) echo "ignoring '$_kv' (expected VAR=VAL)" >&2 ;;
    esac
done

# Pick a subset with MODELS, which the EXTRA loop above has already had its
# chance to set:  ... MODELS=convnext  runs that arm alone.
#
# Commas as well as spaces, because the argument list is word-split: a
# space-separated MODELS="vit convnext" cannot survive being passed as an
# argument, so MODELS=vit,convnext is the form that works there. Both are
# accepted, since the environment can still carry spaces.
MODELS=$(echo "$MODELS" | tr ',' ' ')
for _md in $MODELS; do
    case "$_md" in
        vit|cnn_vit|convnext|vit_local|cnn_vit_local|vit_b16) ;;
        # Caught here rather than in Python: an unknown name would otherwise
        # surface only after the corpus had loaded, once per combination.
        *) echo "unknown MODEL '$_md' (use vit, cnn_vit, convnext, vit_local, cnn_vit_local or vit_b16)" >&2; exit 1 ;;
    esac
done

# The fixed settings of this grid. ${VAR:-default} so EXTRA or the environment
# can still override any of them -- NUM_EPOCHS=2 for a smoke test, say.
NUM_EPOCHS=${NUM_EPOCHS:-30}
SEED=${SEED:-42}
TARGET_SIZE=${TARGET_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
PATCH=${PATCH:-16}
export RESULTS_ROOT

# Fragmentation guard. The partition GPU is shared, so this job runs in
# whatever is left after the other tenants; an allocator holding many
# fixed-size segments can fail to place a large tensor even when the free
# total looks sufficient. Costs nothing when the card is empty.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

_n=$(( $(echo $FRONTENDS | wc -w) * $(echo $MODELS | wc -w) * $(echo $DATASETS | wc -w) ))
_i=0
_summary=""
_failed=0
_t_all=$(date +%s)

echo "############################################################"
echo "# Run8 @ ${TARGET_SIZE}x${TARGET_SIZE} | [$DATASETS] x [$FRONTENDS] x [$MODELS] = $_n runs"
echo "#   NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED BATCH_SIZE=$BATCH_SIZE PATCH=$PATCH"
echo "#   RESULTS_ROOT=$RESULTS_ROOT"
echo "#   $EXTRA"
echo "############################################################"

for _ds in $DATASETS; do
    for _fe in $FRONTENDS; do
        for _md in $MODELS; do
            _i=$((_i + 1))
            _t0=$(date +%s)

            echo
            echo "------------------------------------------------------------"
            echo "[$_i/$_n] $_ds | FRONTEND=$_fe MODEL=$_md @ $TARGET_SIZE"
            echo "------------------------------------------------------------"

            set +e
            bash "$_here/run_server.sh" "$_ds" Run8.py \
                "FRONTEND=$_fe" "MODEL=$_md" \
                "TARGET_SIZE=$TARGET_SIZE" "BATCH_SIZE=$BATCH_SIZE" "PATCH=$PATCH" \
                "NUM_EPOCHS=$NUM_EPOCHS" "SEED=$SEED" $EXTRA
            _rc=$?
            set -e
            _mins=$(( ($(date +%s) - _t0) / 60 ))

            if [ $_rc -eq 0 ]; then
                _summary="${_summary}  ok      ${_ds} ${_fe}/${_md}@${TARGET_SIZE} (${_mins}m)\n"
            else
                _summary="${_summary}  FAILED  ${_ds} ${_fe}/${_md}@${TARGET_SIZE} (rc=$_rc, ${_mins}m)\n"
                _failed=$((_failed + 1))
                echo "[$_i/$_n] FAILED (rc=$_rc) -- continuing with the rest" >&2
            fi
        done
    done
done

echo
echo "############################################################"
echo "# done in $(( ($(date +%s) - _t_all) / 60 )) min"
printf "$_summary"
echo "#"
echo "# results: $RESULTS_ROOT/MelTorch_ViT/<corpus>/sweep.csv   (mel runs)"
echo "#          $RESULTS_ROOT/LeafTorch_ViT/<corpus>/sweep.csv  (leaf runs)"
echo "#"
echo "# One row per classifier in each -- the ablation table. Rows carry"
echo "# frontend, model and target_size, so mel vs leaf can also be read"
echo "# across the two files. Compare on test_uar."
echo "############################################################"

[ "$_failed" -eq 0 ] || exit 1
