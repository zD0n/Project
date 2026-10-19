
DATASET=${DATASET:-iemocap}
SCRIPT=Run9.py

MODEL=${1:-${MODEL:-vit}}
POOL=${POOL:-attn}
DEPTH=${DEPTH:-6}
SPEC_AUGMENT=${SPEC_AUGMENT:-0}
FREQ_MASK=${FREQ_MASK:-8}
TIME_MASK=${TIME_MASK:-24}
NUM_EPOCHS=${EPOCHS:-60}
LR=${LR:-3e-4}

[ $# -gt 0 ] && shift

for kv in "$@"; do
    case "$kv" in
        *=*) export "$kv" ;;
        *) echo "ignoring '$kv' (expected VAR=VAL)" >&2 ;;
    esac
done

FRONTENDS=${FRONTENDS:-"mfcc mel cqt sincnet leaf"}
# FRONTENDS=${FRONTENDS:-"cqt"}

export MODEL
export SPEC_AUGMENT

export POOL
export LOSS


export NUM_EPOCHS

export BATCH_SIZE=${BATCH_SIZE:-32}
export LR
export SEED=${SEED:-99}
export TARGET_SIZE=${TARGET_SIZE:-128}
export PATCH=${PATCH:-8}
export DIM=${DIM:-256}
export DEPTH
export HEADS=${HEADS:-8}
export MLP_DIM=${MLP_DIM:-1024}
export ATT_HEADS=${ATT_HEADS:-5}
export IEMOCAP_CLASSES=${IEMOCAP_CLASSES:-4}
export CV_FOLDS=${CV_FOLDS:-0}
export FIXED_SECONDS=${FIXED_SECONDS:-5}
export COORD_CHANNELS=${COORD_CHANNELS:-1}
export NORMALIZE=${NORMALIZE:-1}
export PCEN=${PCEN:-1}
export FREQ_MASK
export TIME_MASK

cd "$(dirname "$0")"

_failed=0
_total=0
for _fe in $FRONTENDS; do
    _total=$((_total + 1))
    echo
    echo "------------------------------------------------------------"
    echo "[$_fe] $SCRIPT | MODEL=$MODEL  NUM_EPOCHS=$NUM_EPOCHS SEED=$SEED"
    echo "------------------------------------------------------------"
    FRONTEND=$_fe python "$SCRIPT"
    if [ $? -ne 0 ]; then
        echo "FRONTEND=$_fe FAILED -- continuing with the rest" >&2
        _failed=$((_failed + 1))
    fi
done

echo
if [ "$_failed" -eq 0 ]; then
    echo "all $_total frontends finished (MODEL=$MODEL)"
else
    echo "$_failed of $_total frontends FAILED (MODEL=$MODEL)" >&2
    exit 1
fi
