
LEAF (learnable audio frontend) + Vision Transformer for speech emotion
recognition, on CREMA-D and IEMOCAP.

## Run scripts

| script | frontend | default model | results dir |
|---|---|---|---|
| `Run5.py` | `FrontEnd/leaf_pytorch` (PyTorch) | `VitGlobal` | `results/Leaf_VitGlobal/<dataset>/` |
| `Run6.py` | `leaf-audio/leaf_audio` (TensorFlow, GPU) | `VitCnnGlobal` | `results/LeafTF_VitCnnGlobal/<dataset>/` |
| `Run7.py` | same as Run6 | `VitCnnGlobal` | `results/LeafTF_VitCnnGlobal_Coord/<dataset>/` |

`Run7` = `Run6` plus methods drawn from the papers in `../Reasearch`: CoordViT
coordinate planes, SCQT-MaxViT time/frequency masking, per-sample
normalization, stochastic depth, and mixup. Each is a flag, so they can be
ablated one at a time.

## `MODEL` — classifier variant

Selectable in `Run6.py` and `Run7.py` via `-e MODEL=...`:

| `MODEL=` | Class | What it is |
|---|---|---|
| `vit` | `VitGlobal` | plain ViT — no CNN stem, global attention |
| `vit_local` | `VitLocal` | plain ViT, windowed local attention |
| `cnn_vit` | `VitCnnGlobal` | CNN stem + global attention (default) |
| `cnn_vit_local` | `VitCnnLocal` | CNN stem + local attention |

**`vit_local` requires `-e DIM_HEAD=32`.** `Model/VitLocal.py` mixes two
definitions of head width — the qkv projection uses the constructor's
`dim_head` while the padding recomputes `dim // heads` — so it only works when
they agree (`DIM_HEAD == DIM // HEADS`, i.e. 256 // 8 = 32). The other three
variants work at any `DIM_HEAD`.

Parameter counts move with `DIM_HEAD` (`cnn_vit` is 7.46M at 64, 5.89M at 32),
so hold it fixed when comparing variants or you are partly measuring model size.

## Running

Everything runs in Docker; see `cmd.txt` for ready-made commands.

```
docker build -t model .
docker run --rm --gpus all -v "${PWD}\Dataset\IEMOCAP:/app/Dataset2" -v "${PWD}\results:/app/results" -e BATCH_SIZE=8 -e FIXED_SECONDS=3 model python Run7.py
```

The image defaults to `Run5.py`; pass `python Run6.py` / `python Run7.py`
explicitly for the others. `BATCH_SIZE=8 FIXED_SECONDS=3` is required for
Run6/Run7 — the TensorFlow LEAF runs its convolution at full waveform
resolution and OOMs on 12 GB at the defaults.

Rebuild after editing any `Run*.py`, `Model/`, `leaf-audio/`, `Dockerfile` or
`requirements.txt`. Environment variables alone need no rebuild.

## Key environment variables

| var | default | notes |
|---|---|---|
| `NUM_EPOCHS` | 30 | |
| `BATCH_SIZE` | 32 | use 8 for Run6/Run7 |
| `FIXED_SECONDS` | 0 = auto (p75, capped 6s) | one fixed input window for every batch |
| `LR` / `LEAF_LR` | 3e-4 / 1e-5 | frontend learns much slower than the head |
| `MODEL` | `cnn_vit` | see table above (Run6/Run7) |
| `POOL` | `cls` | `mean` pools all tokens instead (Run7) |
| `COORD_CHANNELS` `SPEC_AUGMENT` `DROP_PATH` `MIXUP_ALPHA` | 1, 1, 0.1, 0.2 | Run7 methods; 0 disables |
| `SPLIT_MODE` | `speaker` | `random` leaks speakers — for measuring that leak only |
| `DATASET` | `auto` | inferred from the label vocabulary |
| `RESUME` | 0 | 1 continues from the previous run's LEAF weights |

## Data and evaluation

Both corpora are split **speaker-independently** — CREMA-D by actor id (91
actors), IEMOCAP by session+gender (10 speakers) — so no speaker appears in both
train and test.

| dataset | classes | usable clips |
|---|---|---|
| CREMA-D | 6: anger disgust fear neutral sad happy | 7442 |
| IEMOCAP | 5: neutral sad anger happy excited | 6896 of 10039 |

IEMOCAP drops frustration/other/surprise/disgust/fear — not in the class set.

Each run appends a row to `sweep.csv` in its results directory with every
parameter and its `test_wa` / `test_uar`. **Compare on `test_uar`** (unweighted
average recall) — that is what the SER literature reports, and weighted accuracy
flatters a model that ignores a small class.
