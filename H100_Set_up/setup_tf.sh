#!/bin/bash
# One-time setup: add TensorFlow to a PyTorch container, for Run6.py / Run7.py.
# Installs into ~/.local (bind-mounted, so it persists across jobs).
#
# Run it INSIDE the container, interactively, before submitting a real job:
#
#   srun --partition=GPU_FOR_AI_STU --gres=gpu:1 --time=00:30:00 --pty \
#     singularity exec --nv -B /nfs-share-stgnode/home/663380260-7 \
#     --pwd /nfs-share-stgnode/home/663380260-7/Workshop \
#     /nfs-share-stgnode/containers/pytorch_25_04_py3.sif bash setup_tf.sh
#
# WARNING: this is not guaranteed to work. TF and PyTorch each bring their own
# CUDA wheels and can disagree. If it fails, ask your admins for a TensorFlow
# container instead -- that is the supported path.

set -e

PY=python
command -v $PY >/dev/null 2>&1 || PY=python3

PYVER=$($PY -c "import sys; print('%d.%d' % sys.version_info[:2])")
echo "container python: $PYVER"
echo ""

# TF 2.15 supports python 3.9-3.11 only. NVIDIA's 25.x images ship 3.12, where
# the oldest usable TF is 2.16 -- and 2.16 switched to Keras 3, which breaks
# leaf_audio's tf.keras.layers.Layer subclasses. tf-keras + TF_USE_LEGACY_KERAS
# restores the Keras 2 API those classes expect.
case "$PYVER" in
    3.9|3.10|3.11)
        echo "installing tensorflow 2.15.1 (Keras 2, matches the Dockerfile)"
        $PY -m pip install --user --no-cache-dir "tensorflow[and-cuda]==2.15.1" gin-config
        NEED_LEGACY=0
        ;;
    *)
        echo "python $PYVER is too new for TF 2.15 -- installing TF 2.17 + tf-keras"
        $PY -m pip install --user --no-cache-dir "tensorflow[and-cuda]==2.17.1" tf-keras gin-config
        NEED_LEGACY=1
        ;;
esac

echo ""
echo "verifying..."
if [ "$NEED_LEGACY" = "1" ]; then
    export TF_USE_LEGACY_KERAS=1
fi
$PY - <<'EOF'
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf, torch
print("  tensorflow", tf.__version__, "| GPUs:", len(tf.config.list_physical_devices("GPU")))
print("  torch     ", torch.__version__, "| cuda:", torch.cuda.is_available())
try:
    import leaf_audio.frontend as f
    leaf = f.Leaf(n_filters=8, sample_rate=16000)
    out = leaf(tf.zeros((1, 16000)), training=False)
    print("  leaf_audio OK, output shape", tuple(out.shape))
except Exception as e:
    print("  leaf_audio FAILED:", type(e).__name__, str(e)[:120])
EOF

echo ""
if [ "$NEED_LEGACY" = "1" ]; then
    echo "IMPORTANT: add this to run_server.sh / job.sbatch before running Run6/Run7:"
    echo "    export TF_USE_LEGACY_KERAS=1"
fi
echo "done."
