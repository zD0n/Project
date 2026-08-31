#!/bin/bash
# =============================================================================
# Run9 -- single-stream audio, 5-fold cross-validated, on IEMOCAP and CREMA-D.
#
# Brings two text-free pieces of Wu, Zhang & Woodland (ICASSP 2021,
# arXiv:2010.14102) onto this repo's LEAF+ViT stack: their 5-head
# self-attentive temporal pooling, and their leave-one-session-out 5-fold CV.
# =============================================================================
#
#   bash run_cv.sh                       # iemocap, both pooling arms
#   bash run_cv.sh cremad
#   bash run_cv.sh both
#   bash run_cv.sh iemocap attn          # the attentive-pooling arm only
#   bash run_cv.sh iemocap all NUM_EPOCHS=2   # smoke test the wiring
#
# Portal submission (no argument slot) -- edit these two lines instead:
DATASET=${DATASET:-iemocap}      # iemocap | cremad | both
ARMS=${ARMS:-all}                # all | cls | attn | margin  (space-separated ok)
EXTRA=${EXTRA:-}                 # e.g. "NUM_EPOCHS=2 SEED=42"
# =============================================================================
#
# THE ARMS. A single `attn` number proves nothing on its own -- the point is
# whether the paper's pooling beats reading the CLS token, on identical folds.
#
#   cls      POOL=cls  LOSS=ce         control. This is Run8's model, so the
#                                      arm also tells you what Run8 scores
#                                      under 5-fold CV rather than one split.
#   attn     POOL=attn LOSS=ce         5-head self-attentive pooling over the
#                                      patch grid -- the paper's mechanism,
#                                      though over (time, freq) patches rather
#                                      than the paper's 1-D frame sequence.
#   margin   POOL=attn LOSS=amsoftmax  + the paper's large-margin softmax.
#
# `all` runs cls and attn -- the comparison that matters. Add `margin`
# explicitly when you want the third.
#
# Each arm is a full 5-fold CV, so `all` on one corpus is 10 trainings.
# Compare on test_uar in sweep.csv, which carries the mean and the std.
#
# CONVNEXT. MODEL is passed straight through, so
#
#   MODEL=convnext TARGET_SIZE=128 bash run_cv.sh iemocap all
#
# runs the same two arms on ConvNeXt: `cls` is its native global average pool
# (Run8's model, bit-identical) and `attn` pools its final stage map instead.
#
# TARGET_SIZE matters more here than for the ViTs. ConvNeXt downsamples by 32,
# so the default 64 leaves a 2x2 map -- four tokens for five heads, too few for
# the attention to say anything the average does not. Use 128 (4x4) or 224
# (7x7). Run9 prints a warning rather than letting a 4-token run pass quietly.
#
# NOTE ON SCOPE. The paper's actual contribution is fusing an audio+text branch
# with a cross-utterance one. Both are text-driven and this is audio-only, so
# neither is reproduced here -- see the header of Run9.py. What is reproduced
# is the pooling, the loss and the protocol.

set -e

[ $# -gt 0 ] && { DATASET=$1; shift; }
[ $# -gt 0 ] && { ARMS=$1; shift; }
[ $# -gt 0 ] && EXTRA="$*"

_here=$(cd "$(dirname "$0")" && pwd)

# ---------------------------------------------------------------------------
# CRLF guard.
#
# This repo is edited on Windows and executed on a Linux node inside
# Singularity. A script that arrives with CRLF endings dies immediately, with
# "$'\r': command not found" and a syntax error on the first case statement,
# because the carriage return becomes part of every token.
#
# By the time this line runs, THIS file has already been parsed, so the guard
# can do nothing for it -- run_cv.sh itself must arrive with LF endings. What
# it can do is protect the scripts it calls, which is where the failure would
# otherwise resurface one step later. It skips $0 deliberately: rewriting a
# script while bash is still reading it corrupts the rest of the run.
#
# Detection counts CR bytes rather than using grep -- some greps (git-bash, and
# any build reading in text mode) strip CR before matching and call a CRLF file
# clean. Piping through `tr -dc` is byte-exact everywhere.
# ---------------------------------------------------------------------------
for _f in "$_here"/*.sh; do
    [ -f "$_f" ] && [ -w "$_f" ] || continue
    # "$0" is whatever path the caller typed and "$_f" is absolute, so compare
    # basenames against $_here rather than the raw strings.
    [ "$_f" = "$_here/$(basename "$0")" ] && continue
    [ "$(tr -dc '\r' < "$_f" | wc -c)" -gt 0 ] || continue
    echo "note: stripping CRLF from $(basename "$_f")" >&2
    tr -d '\r' < "$_f" > "$_f.lf" && mv "$_f.lf" "$_f"
done

case "$DATASET" in
    both|all) DATASETS="iemocap cremad" ;;
    *)        DATASETS="$DATASET" ;;
esac
case "$ARMS" in
    all) ARMS="cls attn" ;;
esac

# Apply EXTRA here, not only by forwarding it, so the banner below reports what
# will actually run. Without this, `run_cv.sh iemocap all NUM_EPOCHS=2` printed
# the default 30 while every arm trained for 2 -- a log that misstates its own
# experiment. run_server.sh exports EXTRA again, which is harmless.
for _kv in $EXTRA; do
    case "$_kv" in
        *=*) export "$_kv" ;;
        *) echo "ignoring '$_kv' (expected VAR=VAL)" >&2 ;;
    esac
done

# Settings shared by every arm, so the comparison is controlled.
export NUM_EPOCHS=${NUM_EPOCHS:-30}
export SEED=${SEED:-42}
export BATCH_SIZE=${BATCH_SIZE:-32}
export FIXED_SECONDS=${FIXED_SECONDS:-3}
export FRONTEND=${FRONTEND:-leaf}
export MODEL=${MODEL:-cnn_vit}
export TARGET_SIZE=${TARGET_SIZE:-64}
export CONVNEXT_SIZE=${CONVNEXT_SIZE:-tiny}
export IEMOCAP_CLASSES=${IEMOCAP_CLASSES:-4}

_n=$(( $(echo $ARMS | wc -w) * $(echo $DATASETS | wc -w) ))
_i=0
_summary=""
_failed=0
_t_all=$(date +%s)

echo "############################################################"
echo "# Run9 | [$DATASETS] x [$ARMS] = $_n x 5-fold CV"
echo "#   FRONTEND=$FRONTEND MODEL=$MODEL TARGET_SIZE=$TARGET_SIZE"
echo "#   NUM_EPOCHS=$NUM_EPOCHS BATCH_SIZE=$BATCH_SIZE SEED=$SEED"
echo "#   IEMOCAP_CLASSES=$IEMOCAP_CLASSES (4 = the paper's 4-way)"
echo "############################################################"

for _ds in $DATASETS; do
    for _arm in $ARMS; do
        case "$_arm" in
            cls)    _flags="POOL=cls LOSS=ce" ;;
            attn)   _flags="POOL=attn LOSS=ce" ;;
            margin) _flags="POOL=attn LOSS=amsoftmax" ;;
            *)      echo "unknown arm '$_arm' (use cls, attn, margin or all)" >&2
                    exit 1 ;;
        esac

        _i=$((_i + 1))
        _t0=$(date +%s)
        echo
        echo "------------------------------------------------------------"
        echo "[$_i/$_n] $_ds | arm=$_arm ($_flags)"
        echo "------------------------------------------------------------"

        # Each arm is its own process: a crash in one cannot take down the
        # rest, and the summary below is the authority on what completed.
        set +e
        bash "$_here/run_server.sh" "$_ds" Run9.py $_flags $EXTRA
        _rc=$?
        set -e
        _mins=$(( ($(date +%s) - _t0) / 60 ))

        if [ $_rc -eq 0 ]; then
            _summary="${_summary}  ok      ${_ds}/${_arm} (${_mins}m)\n"
        else
            _summary="${_summary}  FAILED  ${_ds}/${_arm} (rc=$_rc, ${_mins}m)\n"
            _failed=$((_failed + 1))
            echo "[$_i/$_n] FAILED (rc=$_rc) -- continuing with the rest" >&2
        fi
    done
done

echo
echo "############################################################"
echo "# done in $(( ($(date +%s) - _t_all) / 60 )) min"
printf "$_summary"
echo "#"
_fedir=$(echo "$FRONTEND" | sed 's/^./\U&/')
echo "# results: results/${_fedir}Torch_CV/<corpus>/"
echo "#   sweep.csv   one row per arm: 5-fold mean test_wa / test_uar + std"
echo "#   folds.csv   the per-fold rows behind each mean"
echo "#"
echo "# Compare arms on test_uar, and read test_uar_std before believing a"
echo "# gap -- with 5 folds a 1-point difference is usually inside the noise."
echo "############################################################"

[ "$_failed" -eq 0 ] || exit 1
