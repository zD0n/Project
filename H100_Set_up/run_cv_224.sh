#!/bin/bash
# =============================================================================
# The 224x224 grid again, but 5-fold cross-validated: Run9, POOL=cls,
# {mel, leaf} x {vit, cnn_vit, convnext}.
#
#   15 trainings per frontend per corpus (3 models x 5 folds).
#
# SUBMIT ONE FRONTEND PER JOB. At ~30-60 min per training, 15 runs is 8-15h --
# inside a 24h wall clock, where both frontends together would not be:
#
#   bash run_cv_224.sh iemocap mel
#   bash run_cv_224.sh iemocap leaf
#
# Portal submission (no argument slot) -- edit these three lines instead:
DATASET=${DATASET:-iemocap}      # iemocap | cremad | both
FRONTEND=${FRONTEND:-mel}        # mel | leaf | both  (both warns; see above)
EXTRA=${EXTRA:-}                 # e.g. "NUM_EPOCHS=2" to smoke test
RESULTS_ROOT=${RESULTS_ROOT:-"results/Ablation Study"}
# =============================================================================
#
# WHAT THIS IS. The same six combinations as run_224.sh, but scored under
# Run9's 5-fold CV instead of one speaker-independent split, so each comes back
# as a mean +- std rather than a single number. That is the version to trust
# when two arms land within a point or two of each other, which on a single
# split is inside the noise.
#
# NOT COMPARABLE ROW-FOR-ROW WITH run_224.sh ON IEMOCAP. Run8 splits happy from
# excited (5 classes); Run9 defaults to IEMOCAP_CLASSES=4, the standard
# neutral / sad / angry / happy+excited protocol the paper and the wider SER
# literature report on. Different label sets mean different chance levels, so a
# number from one grid does not belong beside a number from the other.
#
# Each grid is internally consistent -- every arm inside it shares a class set,
# which is what an ablation needs -- so both are valid on their own terms. Treat
# these CV numbers as the result and the run_224.sh ones as the cheap screen.
# Pass IEMOCAP_CLASSES=5 to match Run8 instead, at the cost of the
# literature-standard number.
#
# POOL=cls, NOT RUN9'S DEFAULT attn. POOL=cls is bit-identical to the Run8
# model, so the classifier being ablated here is the same one run_224.sh
# ablates -- what changes is the protocol around it (5-fold CV, and on IEMOCAP
# the 4-way class set noted above). Run9's
# self-attentive pooling is a different question, and run_cv.sh is where it is
# asked, as `cls` vs `attn` on identical folds. Pass POOL=attn in EXTRA to
# borrow it here, but then these numbers no longer speak to the Run8 grid.
#
# PATCH=16, for the reason spelled out in run_224.sh: at 224 the default 8
# gives 784 tokens and a 602 MiB attention map per layer, which is what OOMed
# the ViT arms on a shared GPU. 16 gives 196 tokens and 38 MiB. Held constant
# across every arm here, and matching run_224.sh so the two grids differ in
# protocol alone rather than in tokenization as well.
#
# WHERE THE NUMBERS LAND. Under "results/Ablation Study/", beside the Run8
# grid but in its own folder, since Run9 writes *Torch_CV and Run8 *Torch_ViT:
#
#   results/Ablation Study/MelTorch_CV/<corpus>/sweep.csv    3 rows, mean+std
#   results/Ablation Study/LeafTorch_CV/<corpus>/sweep.csv   3 rows, mean+std
#   results/Ablation Study/*/<corpus>/folds.csv              the 5 rows behind
#                                                            each mean
#
# RESTARTS ARE CHEAP. Run9 appends a row to folds.csv per finished fold and
# skips folds already recorded, so a job killed by the wall clock resumes at
# the fold it died on -- just resubmit the same command. The sweep.csv row for
# a combination is written only once all five of its folds are in.

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

# Apply EXTRA here, not only by forwarding it, so the banner reports what will
# actually run rather than the defaults it is about to be handed.
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
        vit|cnn_vit|convnext|vit_local|cnn_vit_local) ;;
        # Caught here rather than in Python: an unknown name would otherwise
        # surface only after the corpus had loaded, once per combination.
        *) echo "unknown MODEL '$_md' (use vit, cnn_vit, convnext, vit_local or cnn_vit_local)" >&2; exit 1 ;;
    esac
done

NUM_EPOCHS=${NUM_EPOCHS:-30}
SEED=${SEED:-42}
TARGET_SIZE=${TARGET_SIZE:-224}
BATCH_SIZE=${BATCH_SIZE:-32}
PATCH=${PATCH:-16}
POOL=${POOL:-cls}
CV_FOLDS=${CV_FOLDS:-5}
# Minutes this job is allowed to run. 0 = no budget, the original behaviour:
# hand Run9 the whole combination and let it loop the folds itself.
#
# Set it and the loop below runs folds one at a time instead, stopping before
# it starts a fold it cannot finish. That matters because the scheduler's kill
# is not graceful: a fold cut off at epoch 25 of 30 writes nothing, so a 3h
# limit that lands mid-fold throws away an hour. Folds already written to
# folds.csv are skipped on the next submission, so the work accumulates across
# jobs instead of restarting.
#
# Set it slightly under the real limit -- 170 for a 3h job -- so the last fold
# has room to write its row.
TIME_BUDGET_MIN=${TIME_BUDGET_MIN:-0}
export RESULTS_ROOT

# Fragmentation guard, as in run_224.sh -- the partition GPU is shared and this
# job runs in whatever the other tenants leave behind.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

_ncombo=$(( $(echo $FRONTENDS | wc -w) * $(echo $MODELS | wc -w) * $(echo $DATASETS | wc -w) ))
_i=0
_summary=""
_failed=0
_fold_max_min=0     # longest fold seen, the basis for the budget estimate
_budget_hit=0
_t_all=$(date +%s)

echo "############################################################"
echo "# Run9 5-fold @ ${TARGET_SIZE}x${TARGET_SIZE} | [$DATASETS] x [$FRONTENDS] x [$MODELS]"
echo "#   $_ncombo combinations x $CV_FOLDS folds = $(( _ncombo * CV_FOLDS )) trainings"
echo "#   POOL=$POOL NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED BATCH_SIZE=$BATCH_SIZE PATCH=$PATCH"
echo "#   RESULTS_ROOT=$RESULTS_ROOT"
[ "$TIME_BUDGET_MIN" -gt 0 ] && echo "#   TIME_BUDGET_MIN=$TIME_BUDGET_MIN (folds run one at a time, stopping before the limit)"
echo "#   $EXTRA"
echo "############################################################"

if [ "$(echo $FRONTENDS | wc -w)" -gt 1 ] || [ "$(echo $DATASETS | wc -w)" -gt 1 ]; then
    echo
    echo "note: $(( _ncombo * CV_FOLDS )) trainings in one job is likely past a 24h" >&2
    echo "      wall clock. One frontend and one corpus per submission is the" >&2
    echo "      intended split; finished folds are skipped on resubmit, so a" >&2
    echo "      job killed by the scheduler is not wasted." >&2
fi

for _ds in $DATASETS; do
    for _fe in $FRONTENDS; do
        for _md in $MODELS; do
            _i=$((_i + 1))
            _t0=$(date +%s)

            echo
            echo "------------------------------------------------------------"
            echo "[$_i/$_ncombo] $_ds | FRONTEND=$_fe MODEL=$_md POOL=$POOL @ $TARGET_SIZE"
            echo "            $CV_FOLDS folds, resuming any already in folds.csv"
            echo "------------------------------------------------------------"

            # One process per combination (or per fold under a time budget),
            # so a crash in one cannot take the rest down.
            _rc=0
            if [ "$TIME_BUDGET_MIN" -le 0 ]; then
                # No budget: hand Run9 the whole combination, it loops folds.
                set +e
                bash "$_here/run_server.sh" "$_ds" Run9.py "FRONTEND=$_fe" "MODEL=$_md" "POOL=$POOL" "TARGET_SIZE=$TARGET_SIZE" "BATCH_SIZE=$BATCH_SIZE" "PATCH=$PATCH" "CV_FOLDS=$CV_FOLDS" "NUM_EPOCHS=$NUM_EPOCHS" "SEED=$SEED" $EXTRA
                _rc=$?
                set -e
            else
                # Budgeted: one fold per process, stopping while there is still
                # time to finish the next one.
                #
                # The estimate is the longest fold seen so far, not the last
                # one: a fold already in folds.csv returns in seconds, and
                # letting that set the expectation would start a real fold with
                # minutes left. Costs one corpus reload per fold (~1 min
                # against ~60), which is the price of never losing a fold.
                _k=0
                while [ "$_k" -lt "$CV_FOLDS" ]; do
                    _left=$(( TIME_BUDGET_MIN - ($(date +%s) - _t_all) / 60 ))
                    if [ "$_fold_max_min" -gt 0 ] && [ "$_left" -lt "$_fold_max_min" ]; then
                        echo
                        echo "stopping before fold $_k: ${_left}m of budget left," >&2
                        echo "  and the longest fold so far took ${_fold_max_min}m." >&2
                        echo "  Resubmit the same command -- finished folds are skipped." >&2
                        _budget_hit=1
                        break
                    fi
                    _tf=$(date +%s)
                    set +e
                    bash "$_here/run_server.sh" "$_ds" Run9.py "FRONTEND=$_fe" "MODEL=$_md" "POOL=$POOL" "TARGET_SIZE=$TARGET_SIZE" "BATCH_SIZE=$BATCH_SIZE" "PATCH=$PATCH" "CV_FOLDS=$CV_FOLDS" "FOLD=$_k" "NUM_EPOCHS=$NUM_EPOCHS" "SEED=$SEED" $EXTRA
                    _frc=$?
                    set -e
                    _fmin=$(( ($(date +%s) - _tf) / 60 ))
                    [ "$_fmin" -gt "$_fold_max_min" ] && _fold_max_min=$_fmin
                    [ "$_frc" -ne 0 ] && _rc=$_frc
                    _k=$((_k + 1))
                done
            fi
            _mins=$(( ($(date +%s) - _t0) / 60 ))

            if [ $_rc -eq 0 ]; then
                _summary="${_summary}  ok      ${_ds} ${_fe}/${_md} (${_mins}m)\n"
            else
                _summary="${_summary}  FAILED  ${_ds} ${_fe}/${_md} (rc=$_rc, ${_mins}m)\n"
                _failed=$((_failed + 1))
                echo "[$_i/$_ncombo] FAILED (rc=$_rc) -- continuing with the rest" >&2
            fi
        done
    done
done

echo
echo "############################################################"
echo "# done in $(( ($(date +%s) - _t_all) / 60 )) min"
printf "$_summary"
if [ "$_budget_hit" = "1" ]; then
    echo "#"
    echo "# STOPPED ON THE TIME BUDGET with folds still to run. Everything that"
    echo "# finished is in folds.csv; resubmit the identical command to continue."
    echo "# A combination's sweep.csv row appears once all $CV_FOLDS folds are in."
fi
echo "#"
echo "# results: $RESULTS_ROOT/{Mel,Leaf}Torch_CV/<corpus>/sweep.csv"
echo "#          one row per model: 5-fold mean test_wa / test_uar, plus std"
echo "#          folds.csv holds the five rows behind each mean"
echo "#"
echo "# Compare on test_uar, and read test_uar_std before believing a gap --"
echo "# with 5 folds a 1-point difference is usually inside the noise. This is"
echo "# the error bar the single-split run_224.sh numbers do not have."
echo "############################################################"

[ "$_failed" -eq 0 ] || exit 1
