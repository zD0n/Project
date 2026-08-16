#!/bin/bash
# =============================================================================
# EDIT THESE THREE LINES, then upload and select this file in the portal.
# (The portal only offers a file picker -- no place to pass arguments.)
# =============================================================================
# The ${VAR:-default} form matters: a wrapper script that does
#   DATASET=cremad exec bash run_server.sh
# passes DATASET in the environment, and a plain `DATASET=iemocap` here would
# overwrite it -- silently running the wrong corpus.
DATASET=${DATASET:-iemocap}    # iemocap | cremad
SCRIPT=${SCRIPT:-Run7.py}      # Run5.py / Run8.py (PyTorch only) | Run6.py | Run7.py (need TF)
EXTRA=${EXTRA:-}               # e.g. "NUM_EPOCHS=2 MODEL=convnext SEED=42"
# =============================================================================
#
# Command-line arguments still work if the environment allows them:
#   bash run_server.sh cremad Run5.py NUM_EPOCHS=2
# They override the values above.

set -e

[ $# -gt 0 ] && { DATASET=$1; shift; }
[ $# -gt 0 ] && { SCRIPT=$1; shift; }
[ $# -gt 0 ] && EXTRA="$*"

# `all` in the script slot runs the full 3-classifier x 2-frontend sweep on
# Run8 instead of a single experiment -- one portal submission, six results.
# Everything else (setup, preflight, EXTRA) behaves identically.
SWEEP=0
if [ "$SCRIPT" = "all" ] || [ "$SCRIPT" = "sweep" ]; then
    SWEEP=1
    SCRIPT=Run8.py
fi

# Maps a corpus name to its folder under Dataset/. Echoes the folder, or
# nothing if the name is not recognised.
_data_dir_for() {
    case "$1" in
        cremad|CREMA-D|crema-d) echo CREMA-D ;;
        iemocap|IEMOCAP)        echo IEMOCAP ;;
    esac
}

# `all`/`both` in the dataset slot runs every corpus. With `all` in the script
# slot too, that is 6 combinations x 2 corpora = 12 runs.
case "$DATASET" in
    all|both) DATASETS="cremad iemocap" ;;
    *)        DATASETS="$DATASET" ;;
esac

for _ds in $DATASETS; do
    if [ -z "$(_data_dir_for "$_ds")" ]; then
        echo "unknown DATASET '$_ds' (use iemocap, cremad, or all)" >&2
        exit 1
    fi
done
# Kept for the single-run path below; the sweep re-points it per corpus.
DATA_DIR=$(_data_dir_for "${DATASETS%% *}")

# Find the folder holding Run*.py / Model/ / Dataset/. Singularity's --pwd
# usually puts us there already; fall back to this script's directory and its
# parent, so a wrapper living in Work/run/ still finds Work/.
_here=$(cd "$(dirname "$0")" && pwd)
for _d in "$PROJECT_DIR" "$PWD" "$_here" "$_here/.."; do
    [ -n "$_d" ] || continue
    if [ -f "$_d/$SCRIPT" ] && [ -d "$_d/Dataset" ]; then
        PROJECT_DIR=$(cd "$_d" && pwd)
        break
    fi
done
if [ -z "$PROJECT_DIR" ]; then
    echo "error: could not find $SCRIPT next to a Dataset/ folder." >&2
    echo "       looked in: $PWD, $_here, $_here/.." >&2
    echo "       set PROJECT_DIR=/path/to/project if it is elsewhere." >&2
    exit 1
fi
cd "$PROJECT_DIR"
echo "working dir: $(pwd)"

# The Run*.py scripts default DATASET_DIR to "Dataset2", so point a symlink at
# whichever corpus is being run instead of editing the code. The sweep calls
# this again between corpora.
_point_dataset() {
    if [ ! -d "Dataset/$1" ]; then
        echo "error: Dataset/$1 not found in $(pwd)" >&2
        return 1
    fi
    rm -f Dataset2
    ln -s "Dataset/$1" Dataset2
}

# Fail now if any requested corpus is missing -- a 12-run sweep should not die
# hours in because the second one was never uploaded.
for _ds in $DATASETS; do
    _dd=$(_data_dir_for "$_ds")
    [ -d "Dataset/$_dd" ] || { echo "error: Dataset/$_dd not found in $(pwd)" >&2; exit 1; }
done

_point_dataset "$DATA_DIR"
mkdir -p results

# Python cannot import a package from a hyphenated directory. The upstream repo
# folder is "leaf-audio" and the package inside it is "leaf_audio"; the
# Dockerfile copies the inner one out. Do the same here with a symlink, so
# uploading the whole repo folder still works.
if [ ! -d leaf_audio ]; then
    for _cand in leaf-audio/leaf_audio leaf_audio_repo/leaf_audio; do
        if [ -d "$_cand" ]; then
            ln -sfn "$_cand" leaf_audio
            echo "linked leaf_audio -> $_cand"
            break
        fi
    done
fi
# Same shape for the PyTorch frontend if the FrontEnd/ folder was uploaded whole.
if [ ! -d leaf_pytorch ] && [ -d FrontEnd/leaf_pytorch ]; then
    ln -sfn FrontEnd/leaf_pytorch leaf_pytorch
    echo "linked leaf_pytorch -> FrontEnd/leaf_pytorch"
fi

# The TF LEAF frontend OOMs at the built-in defaults; these fit a 12 GB card.
export BATCH_SIZE=${BATCH_SIZE:-8}
export FIXED_SECONDS=${FIXED_SECONDS:-3}

# TF and PyTorch in one process can trip over CUDA 12's lazy module loading,
# surfacing as CUDA_ERROR_INVALID_HANDLE on TF's first GPU op. Eager loading
# avoids it. Harmless when it was never a problem.
export CUDA_MODULE_LOADING=${CUDA_MODULE_LOADING:-EAGER}
# XLA's JIT is another source of TF/torch context friction and buys nothing here.
export TF_XLA_FLAGS=${TF_XLA_FLAGS:---tf_xla_auto_jit=0}

for kv in $EXTRA; do
    case "$kv" in
        *=*) export "$kv" ;;
        *) echo "ignoring '$kv' (expected VAR=VAL)" >&2 ;;
    esac
done

PY=python
command -v $PY >/dev/null 2>&1 || PY=python3

# leaf_audio subclasses tf.keras layers in the Keras 2 style.
#   TF 2.15 bundles Keras 2 -> use it as-is. Setting TF_USE_LEGACY_KERAS here
#            makes TF look for the separate tf_keras package and fail with
#            "Keras cannot be imported".
#   TF 2.16+ ships Keras 3 -> needs the tf-keras package plus this flag.
if $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tf_keras') else 1)" 2>/dev/null; then
    export TF_USE_LEGACY_KERAS=1
    echo "tf_keras present -> TF_USE_LEGACY_KERAS=1"
else
    unset TF_USE_LEGACY_KERAS
fi

echo "python: $($PY -V 2>&1)"
echo "TF_USE_LEGACY_KERAS='${TF_USE_LEGACY_KERAS-<unset>}'"
$PY - <<'EOF'
import importlib.util as u
for m in ("torch", "torchaudio", "numpy", "pandas", "soundfile", "einops", "timm",
          "tensorflow", "gin", "keras", "tf_keras"):
    print("  {:<12s} {}".format(m, "yes" if u.find_spec(m) else "MISSING"))
try:
    import torch
    print("  torch cuda   {} ({})".format(
        torch.cuda.is_available(),
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"))
except Exception as e:
    print("  torch failed:", e)
# The exact failure Run6/Run7 hit at import time: tf.keras is lazy-loaded, so a
# broken Keras only shows up when something touches it.
try:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    _ = tf.keras.layers.Conv1D
    import keras
    print("  tf.keras     OK (tf {}, keras {})".format(tf.__version__, keras.__version__))
except Exception as e:
    print("  tf.keras     FAILED: {}: {}".format(type(e).__name__, e))
EOF

# Anything missing from the image can be added here without a rebuild; it lands
# in ~/.local, which is bind-mounted and persists across jobs.
$PY -c "import importlib.util as u,sys; sys.exit(0 if all(u.find_spec(m) for m in ('soundfile','einops','pandas')) else 1)" || {
    echo "installing missing python packages into ~/.local ..."
    $PY -m pip install --user --no-cache-dir gin-config soundfile einops pandas "numpy==1.26.4"
}

case "$SCRIPT" in
  Run6.py|Run7.py)
    $PY -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('tensorflow') else 1)" || {
        echo "error: $SCRIPT needs TensorFlow, which this image does not have." >&2
        echo "       Use Run5.py or Run8.py, or build the image with tensorflow[and-cuda]==2.15.1" >&2
        exit 1
    } ;;
esac

# Run8 MODEL=convnext imports timm, and the model code itself from the mounted
# ConvNeXt/ folder. Both fail well into the run -- after the whole corpus has
# been loaded -- so check them up front.
if [ "$SCRIPT" = "Run8.py" ] && { [ "${MODEL:-}" = "convnext" ] || [ "$SWEEP" = "1" ]; }; then
    [ -f ConvNeXt/models/convnext.py ] || {
        echo "error: MODEL=convnext needs ConvNeXt/models/convnext.py under $(pwd)." >&2
        echo "       Upload the repo's ConvNeXt/ folder alongside Run8.py." >&2
        exit 1
    }
    # ConvNeXt imports two symbols timm keeps only as deprecated shims, so test
    # those rather than merely that timm is importable.
    _timm_ok() {
        $PY -c "from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model" 2>/dev/null
    }
    if ! _timm_ok; then
        # timm depends on torchvision, and pip is free to satisfy that with a
        # PyPI build whose own torch requirement pulls a newer torch into
        # ~/.local -- which then shadows the container's CUDA-matched torch for
        # every later job. Pin both to what is already installed.
        _torch_v=$($PY -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")
        _tv_v=$($PY -c "import torchvision; print(torchvision.__version__.split('+')[0])" 2>/dev/null || echo "")
        _cons=$(mktemp)
        [ -n "$_torch_v" ] && echo "torch==$_torch_v" >> "$_cons"
        [ -n "$_tv_v" ] && echo "torchvision==$_tv_v" >> "$_cons"
        echo "installing timm into ~/.local (holding torch=${_torch_v:-any} torchvision=${_tv_v:-any}) ..."
        $PY -m pip install --user --no-cache-dir -c "$_cons" "timm==1.0.24" || true
        rm -f "$_cons"
        # If torchvision was genuinely absent, the constraint above could not
        # hold it and pip chose a version; say so rather than leaving it silent.
        if [ -z "$_tv_v" ]; then
            echo "note: torchvision was not present and has been installed as a timm dependency." >&2
        fi
        $PY -c "import torch; assert torch.__version__.startswith('${_torch_v:-0}')" 2>/dev/null || {
            echo "error: installing timm changed torch (was ${_torch_v:-unknown}, now" >&2
            echo "       $($PY -c 'import torch; print(torch.__version__)' 2>&1)). The CUDA build may" >&2
            echo "       no longer match this node. Remove ~/.local/lib/python*/site-packages/torch*" >&2
            echo "       and rebuild the image from leaf_vit_h100.def instead." >&2
            exit 1
        }
    fi
    _timm_ok || {
        echo "error: MODEL=convnext needs timm, and installing it failed." >&2
        echo "       Rebuild the image from leaf_vit_h100.def, which pins timm==1.0.24." >&2
        exit 1
    }
fi

if [ "$SWEEP" != "1" ]; then
    echo "== $SCRIPT on $DATA_DIR | BATCH_SIZE=$BATCH_SIZE FIXED_SECONDS=$FIXED_SECONDS $EXTRA =="
    exec $PY "$SCRIPT"
fi

# ---------------------------------------------------------------------------
# Sweep: 3 classifiers x 2 frontends, one process each.
#
# Sequential, not parallel: they share one GPU, and interleaving them would
# make the per-epoch timings in training_log.csv meaningless as a comparison.
#
# Each combination is its own process so a crash in one cannot corrupt the
# next -- the loop records the failure and carries on, and the summary at the
# end is the authority on what actually completed. EXTRA still applies to every
# run, so `... all NUM_EPOCHS=2` smoke-tests the whole grid.
#
# The 224 x batch-32 ConvNeXt runs live in run_<corpus>_224.sh, not here.
# ---------------------------------------------------------------------------
export NUM_EPOCHS=${NUM_EPOCHS:-30}
export SEED=${SEED:-42}

_combos="leaf:vit leaf:cnn_vit leaf:convnext mel:vit mel:cnn_vit mel:convnext"
_n=$(( $(echo $_combos | wc -w) * $(echo $DATASETS | wc -w) ))
_i=0
_summary=""
_failed=0
_t_all=$(date +%s)

echo
echo "############################################################"
echo "# sweep: $_n runs over [$DATASETS]"
echo "#   NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED BATCH_SIZE=$BATCH_SIZE FIXED_SECONDS=$FIXED_SECONDS"
echo "#   $EXTRA"
echo "############################################################"

for _ds in $DATASETS; do
    _dd=$(_data_dir_for "$_ds")
    _point_dataset "$_dd"
    echo
    echo "============================================================"
    echo "corpus: $_dd  (Dataset2 -> Dataset/$_dd)"
    echo "============================================================"

    for _c in $_combos; do
        _fe=${_c%%:*}
        _md=${_c##*:}
        _i=$((_i + 1))
        _t0=$(date +%s)

        echo
        echo "------------------------------------------------------------"
        echo "[$_i/$_n] $_dd | FRONTEND=$_fe MODEL=$_md"
        echo "------------------------------------------------------------"

        # set +e around the run: one failed combination must not abort the sweep.
        set +e
        FRONTEND=$_fe MODEL=$_md $PY "$SCRIPT"
        _rc=$?
        set -e
        _mins=$(( ($(date +%s) - _t0) / 60 ))

        if [ $_rc -eq 0 ]; then
            _summary="${_summary}  ok      ${_dd} ${_fe}/${_md} (${_mins}m)\n"
        else
            _summary="${_summary}  FAILED  ${_dd} ${_fe}/${_md} (rc=$_rc, ${_mins}m)\n"
            _failed=$((_failed + 1))
            echo "[$_i/$_n] FAILED (rc=$_rc) -- continuing with the rest" >&2
        fi
    done
done

echo
echo "############################################################"
echo "# sweep done in $(( ($(date +%s) - _t_all) / 60 )) min"
printf "$_summary"
echo "#"
echo "# results: results/LeafTorch_ViT/<dataset>/sweep.csv   (leaf runs)"
echo "#          results/MelTorch_ViT/<dataset>/sweep.csv    (mel runs)"
echo "# 4 sweep.csv files in total -- frontend x corpus. compare on test_uar"
echo "############################################################"

# Non-zero exit if anything failed, so the scheduler does not report success
# for a sweep that only half ran.
[ "$_failed" -eq 0 ] || exit 1
