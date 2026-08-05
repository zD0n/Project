"""Isolate where the TF LEAF frontend produces NaN. Run inside the image:

  docker run --rm --gpus all -v "${PWD}\leafcheck.py:/app/leafcheck.py" model python leafcheck.py

No rebuild needed -- the script is mounted over /app.
"""
import os
import functools

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

import leaf_audio.frontend as F
from leaf_audio import initializers

SR = 16000
N = 64
WIN = 25


def finite(t):
    a = t.numpy() if hasattr(t, "numpy") else np.asarray(t)
    return bool(np.isfinite(a).all()), float(np.nanmin(a)), float(np.nanmax(a))


# Same waveform shape Run7 feeds: (batch, samples), 3 s at 16 kHz.
wav = np.random.randn(4, 3 * SR).astype(np.float32) * 0.05
x = tf.constant(wav)
print("input                 finite={} min={:+.4g} max={:+.4g}".format(*finite(x)))
print()

for preemp in (True, False):
    print("=" * 62)
    print("Leaf(preemp={}, learn_pooling=False, log_compression)".format(preemp))
    leaf = F.Leaf(
        learn_pooling=False,
        n_filters=N,
        window_len=WIN,
        sample_rate=SR,
        preemp=preemp,
        compression_fn=functools.partial(F.log_compression, log_offset=1e-5),
        complex_conv_init=initializers.GaborInit(
            sample_rate=SR, min_freq=60.0, max_freq=7800.0),
    )
    out = leaf(x, training=False)
    ok, lo, hi = finite(out)
    print("  full forward        finite={} min={:+.4g} max={:+.4g}".format(ok, lo, hi))

    # Walk the stages of Leaf.call by hand to find the first non-finite one.
    o = x[:, :, tf.newaxis]
    if preemp:
        o = leaf._preemp_conv(o)
        print("  after preemp_conv   finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))
    o = leaf._complex_conv(o)
    print("  after complex_conv  finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))
    o = leaf._activation(o)
    print("  after activation    finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))
    o = leaf._pooling(o)
    print("  after pooling       finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))
    o = tf.maximum(o, 1e-5)
    print("  after maximum       finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))
    o = leaf._compress_fn(o)
    print("  after compression   finite={} min={:+.4g} max={:+.4g}".format(*finite(o)))

    print("  weights:")
    for v in leaf.trainable_variables:
        vok, vlo, vhi = finite(v)
        print("    {:<34s} finite={} min={:+.4g} max={:+.4g}".format(
            v.name[:34], vok, vlo, vhi))
    print()
