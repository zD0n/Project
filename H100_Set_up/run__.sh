
DATASET=${DATASET:-iemocap}
SCRIPT=Run9.py

# Select Model
export FRONTEND=${FRONTEND:-leaf}
export MODEL=${MODEL:-cnn_vit_bigger}

export POOL=${POOL:-attn}
export LOSS=${LOSS:-ce}


export NUM_EPOCHS=${NUM_EPOCHS:-30}

export BATCH_SIZE=${BATCH_SIZE:-5}
export LR=${LR:-3e-4}
export SEED=${SEED:-99}
export TARGET_SIZE=${TARGET_SIZE:-64}
export PATCH=${PATCH:-8}
export DIM=${DIM:-256}
export DEPTH=${DEPTH:-6}
export HEADS=${HEADS:-8}
export MLP_DIM=${MLP_DIM:-1024}
export ATT_HEADS=${ATT_HEADS:-5}
export IEMOCAP_CLASSES=${IEMOCAP_CLASSES:-4}
export CV_FOLDS=${CV_FOLDS:-5}
export CONVNEXT_SIZE=${CONVNEXT_SIZE:-tiny}
export FIXED_SECONDS=${FIXED_SECONDS:-3}
export COORD_CHANNELS=${COORD_CHANNELS:-1}
export SPEC_AUGMENT=${SPEC_AUGMENT:-1}
export NORMALIZE=${NORMALIZE:-1}
export PCEN=${PCEN:-1}

# Load pretrained model
export LOAD_MODEL=${LOAD_MODEL:-}

cd "$(dirname "$0")"
exec python "$SCRIPT"
