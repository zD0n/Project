
LEAF (learnable audio frontend) + Vision Transformer for speech emotion
recognition, on CREMA-D and IEMOCAP.

## Run scripts

| script | frontend | default model | results dir |
|---|---|---|---|
| `Run5.py` | `FrontEnd/leaf_pytorch` (PyTorch) | `VitGlobal` | `results/Leaf_VitGlobal/<dataset>/` |
| `Run6.py` | `leaf-audio/leaf_audio` (TensorFlow, GPU) | `VitCnnGlobal` | `results/LeafTF_VitCnnGlobal/<dataset>/` |
| `Run7.py` | same as Run6 | `VitCnnGlobal` | `results/LeafTF_VitCnnGlobal_Coord/<dataset>/` |
| `Run8.py` | `FrontEnd/leaf_pytorch` (PyTorch) | `VitCnnGlobal` | `results/LeafTorch_ViT/<dataset>/` |
| `Run9.py` | same as Run8 | `VitCnnGlobal` / `ConvNeXt` + attn pooling | `results/{Leaf,Mel}Torch_CV/<dataset>/` |

`Run7` = `Run6` plus methods drawn from the papers in `../Reasearch`: CoordViT
coordinate planes, SCQT-MaxViT time/frequency masking, and per-sample
normalization. Each is a flag, so they can be ablated one at a time.

`Run8` = `Run7`'s experiment with no TensorFlow: the same methods on the PyTorch
LEAF port, so frontend and classifier share one autograd graph on one device. It
is also the only script that can run **ConvNeXt** (`MODEL=convnext`).

`Run9` = Run8's pipeline with the paper's self-attentive pooling in place of
the ViT's CLS token, and the only script scored by **5-fold cross-validation**
rather than a single split. See below.

## `MODEL` — classifier variant

Selectable in `Run6.py`, `Run7.py` and `Run8.py` via `-e MODEL=...`:

| `MODEL=` | Class | What it is |
|---|---|---|
| `vit` | `VitGlobal` | plain ViT — no CNN stem, global attention |
| `vit_local` | `VitLocal` | plain ViT, windowed local attention |
| `cnn_vit` | `VitCnnGlobal` | CNN stem + global attention (default) |
| `cnn_vit_local` | `VitCnnLocal` | CNN stem + local attention |
| `convnext` | `ConvNeXt` | pure conv, no attention - **`Run8.py` / `Run9.py`** |
| `vit_b16` | timm `VisionTransformer` | standard ViT-B/16, can load ImageNet weights — **`Run8.py` only** |

### `MODEL=vit_b16`

The only classifier here that can start from pretrained weights. The ViTs in
`Model/` are hand-rolled at `dim=256, depth=6, patch=8`, a shape no published
checkpoint exists for, so they can only train from scratch; `vit_b16` builds
timm's `vit_base_patch16_224` instead (~86M parameters, 16×16 patches) and
`VIT_PRETRAINED=1` initialises it from ImageNet-21k.

| var | default | notes |
|---|---|---|
| `VIT_PRETRAINED` | 0 | 1 loads ImageNet-21k weights (~330 MB, needs network) |
| `VIT_TIMM_NAME` | `vit_base_patch16_224.augreg_in21k` | any timm ViT tag |
| `TRAIN_FRAC` | 1.0 | <1 subsamples the training clips; val/test untouched |

timm handles both mismatches against the checkpoint: `in_chans` folds the RGB
patch-embedding stem down to `VIT_CHANNELS`, and the 21841-way head is replaced
by a fresh `NUM_CLASSES` one. `TARGET_SIZE` must be a multiple of 16 and is
passed through as `img_size`, with the position embeddings interpolated to
match, so 64 (16 tokens) and 224 (196 tokens) both work.

This exists to replicate [arXiv:2502.12379](https://arxiv.org/abs/2502.12379),
which reports that ImageNet pre-training only helps a ViT on small datasets.
`H100_Set_up/run_paper_vit.sh` runs that paper's 2×2 — pretrained/scratch ×
full/small — on CREMA-D or IEMOCAP.

### `MODEL=convnext`

Runs [facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt),
vendored unmodified at `ConvNeXt/`, on the same LEAF features. The frontend,
split, SpecAugment and eval are untouched, so a `convnext` row in `sweep.csv` is
comparable to a `cnn_vit` row on everything but the classifier.

| var | default | notes |
|---|---|---|
| `CONVNEXT_SIZE` | `tiny` | `tiny` `small` `base` `large` `xlarge` |
| `DROP_PATH` | 0.1 | stochastic depth; raise it if train/test gap is wide |
| `CONVNEXT_PRETRAINED` | 0 | 1 downloads ImageNet weights (~110 MB for tiny) |
| `CONVNEXT_22K` | 0 | with the above, ImageNet-22k instead of 1k |

Two things to know before comparing it to the ViTs:

- **It is much bigger.** ConvNeXt-T is 27.8M parameters against `cnn_vit`'s
  7.46M, on ~5k training clips. Expect it to overfit unless you raise
  `DROP_PATH` / `WEIGHT_DECAY`, and do not read a lower UAR as "ConvNeXt is
  worse" without matching capacity.
- **The stem is aggressive for a 64×64 input.** Four stages downsample by 32×,
  so the map is 2×2 before global pooling. `TARGET_SIZE` below 32 is rejected;
  `TARGET_SIZE=128` gives it a 4×4 map to pool over.

`COORD_CHANNELS` works here too — the coordinate planes become extra stem input
channels. With `CONVNEXT_PRETRAINED=1` and `COORD_CHANNELS=1` the input is 3
channels, so the RGB stem loads as-is; at 1 channel it is averaged and rescaled.
The classifier head is always trained from scratch (6 or 5 classes, not 1000).

**`vit_local` requires `-e DIM_HEAD=32`.** `Model/VitLocal.py` mixes two
definitions of head width — the qkv projection uses the constructor's
`dim_head` while the padding recomputes `dim // heads` — so it only works when
they agree (`DIM_HEAD == DIM // HEADS`, i.e. 256 // 8 = 32). The other three
variants work at any `DIM_HEAD`.

Parameter counts move with `DIM_HEAD` (`cnn_vit` is 7.46M at 64, 5.89M at 32),
so hold it fixed when comparing variants or you are partly measuring model size.

## `Run9.py` — single-stream, 5-fold cross-validated

Brings the two text-free pieces of Wu, Zhang & Woodland, *"Emotion recognition
by fusing time synchronous and time asynchronous representations"*
([arXiv:2010.14102](https://arxiv.org/abs/2010.14102), ICASSP 2021) onto this
repo's LEAF+ViT stack. One stream, audio only:

```
waveform -> LEAF/mel -> (+coord planes) -> ViT -> pool -> FC -> softmax
```

**Be clear about what this is not.** The paper's actual contribution is fusing
an audio+text branch with a cross-utterance one. Both branches are text-driven,
this pipeline is audio-only, and neither is reproduced. What is reproduced:

- **5-head self-attentive pooling** (`POOL=attn`) instead of reading the ViT's
  CLS token — three heads initialised spiky, two smooth, as in the paper, via a
  learnable per-head temperature.
- **Leave-one-session-out 5-fold CV** — the paper's protocol. Run5–Run8 each
  report a single split, so their numbers carry no error bar; a Run9 row is a
  mean ± std over five folds.
- **Large-margin softmax** (`LOSS=amsoftmax`), the paper's loss family.

One honest caveat on the pooling: the paper pools a 1-D sequence of time
frames, but a ViT's tokens are a 2-D grid of (time, frequency) patches — 8×8 =
64 tokens at `TARGET_SIZE=64 PATCH=8`. So the attention weights time and
frequency jointly, not time alone. That follows from pooling a ViT rather than
a TDNN; it is not a tunable.

### The arms

`POOL=cls LOSS=ce` is Run8's model exactly, so the comparison is controlled and
the control arm doubles as "what Run8 scores under 5-fold CV".

| arm | flags | what it isolates |
|---|---|---|
| `cls` | `POOL=cls LOSS=ce` | control — Run8's model on the new protocol |
| `attn` | `POOL=attn LOSS=ce` | what the paper's attentive pooling buys |
| `margin` | `POOL=attn LOSS=amsoftmax` | + the large-margin softmax |

`run_cv.sh` runs `cls attn` by default; add `margin` explicitly.

### Evaluation

IEMOCAP uses the paper's **leave-one-session-out** protocol: 8 speakers train,
2 test, five folds. CREMA-D has no sessions, so its 91 actors are shuffled into
5 actor-disjoint groups — different corpus structure, same guarantee. `VAL_FRAC`
(0.125) of the *training* speakers is held out per fold for checkpoint
selection, never the test group.

`sweep.csv` gets one row per configuration with the 5-fold mean and std of
`test_wa` / `test_uar`; `folds.csv` holds the per-fold rows behind it. **Folds
are the unit of restart** — each finished fold is recorded immediately and a
rerun skips it, so a job killed in fold 3 resumes there. `FOLD=k` runs a single
fold, for splitting the CV across jobs.

Results land in `results/{Leaf,Mel}Torch_CV/<corpus>/`, deliberately apart from
Run8's `…Torch_ViT/` — a CV mean and a single-split score should not share a
`test_uar` column.

### Class sets

`IEMOCAP_CLASSES` selects the label mapping:

| value | classes | clips | notes |
|---|---|---|---|
| `4` | neutral sad anger happy(+excited) | 6896 | the paper's 4-way, default |
| `5` | excited kept separate | 6896 | matches Run8, for direct comparison |
| `5others` | 4-way + others | 10039 | the paper's 5-way (frustration etc.) |

The paper reports 5531 utterances for 4-way against 6896 here, so its
annotator-agreement filter differs and the task is not identical — its
77.57 WA / 78.41 UA is a landmark, not a target, doubly so without text. The
controlled comparison is Run9's arms against each other on the same folds.

### Run9 environment variables

Everything Run8 accepts, plus:

| var | default | notes |
|---|---|---|
| `POOL` | `attn` | `cls` reads the CLS token instead (Run8 behaviour) |
| `ATT_HEADS` | 5 | self-attentive heads, as in the paper |
| `ATT_SHARP` | 3 | how many start on the spiky temperature |
| `ATT_HIDDEN` | 64 | bottleneck width inside the attention |
| `EMBED_DIM` | 256 | pooled embedding width before the classifier |
| `IEMOCAP_CLASSES` | 4 | `4` `5` `5others` |
| `LOSS` | `ce` | `amsoftmax` — the paper's large-margin family |
| `AM_MARGIN` `AM_SCALE` | 0.2, 30 | AM-Softmax only |
| `CV_FOLDS` `FOLD` | 5, -1 | `FOLD=k` runs one fold, for job splitting |
| `VAL_FRAC` | 0.125 | share of *training* speakers held out per fold |
| `CONVNEXT_SIZE` `DROP_PATH` | tiny, 0.1 | `MODEL=convnext` only, as in Run8 |
| `CONVNEXT_PRETRAINED` `CONVNEXT_22K` | 0, 0 | ImageNet weights for ConvNeXt |

### ConvNeXt in Run9

`MODEL=convnext` works, with the same knobs as Run8 (`CONVNEXT_SIZE`,
`DROP_PATH`, `CONVNEXT_PRETRAINED`, `CONVNEXT_22K`):

```bash
MODEL=convnext TARGET_SIZE=128 bash H100_Set_up/run_cv.sh iemocap all
```

ConvNeXt has no token sequence in the ViT sense, but it has the same thing
under another name: stage 4 emits an `(N, C, H, W)` map that `forward_features`
immediately global-average-pools away. Flattening it gives `H*W` tokens of
width `C`, the grid the attention pools over. So `POOL=cls` is ConvNeXt's own
GAP head (verified bit-identical to the vendored `forward_features`, max abs
diff 0.0) and `POOL=attn` replaces that average with the paper's weighted one.

**`TARGET_SIZE` matters more here than for the ViTs.** ConvNeXt downsamples by
32, so the default 64 leaves a 2x2 map: four tokens for five heads, too few for
the attention to say anything the average does not.

| `TARGET_SIZE` | grid | tokens |
|---|---|---|
| 64 | 2x2 | 4 - too few, Run9 warns |
| 128 | 4x4 | 16 |
| 224 | 7x7 | 49 - ConvNeXt's design point |

One deliberate asymmetry: for `POOL=attn` the LayerNorm is applied per token
*before* pooling, where `POOL=cls` pools then norms. The attention scores tokens
against each other, so they must share a scale first, and it matches the ViT
arm, where the transformer's final LayerNorm has already normed every token. It
does mean a uniform attention is not numerically identical to GAP.

`vit_b16` stays in Run8.

Run9 needs nothing Run8 does not: no packages beyond the `timm` that `convnext`
already required, no downloads unless `CONVNEXT_PRETRAINED=1`, and no text.

## Running

Everything runs in Docker; see `cmd.txt` for ready-made commands.

```
docker build -t model .
docker run --rm --gpus all -v "${PWD}\Dataset\IEMOCAP:/app/Dataset2" -v "${PWD}\results:/app/results" -e BATCH_SIZE=8 -e FIXED_SECONDS=3 model python Run7.py
```

The image defaults to `Run5.py`; pass `python Run6.py` / `python Run7.py` /
`python Run8.py` / `python Run9.py` explicitly for the others. `BATCH_SIZE=8 FIXED_SECONDS=3` is
required for Run6/Run7 — the TensorFlow LEAF runs its convolution at full
waveform resolution and OOMs on 12 GB at the defaults.

ConvNeXt run, otherwise identical to a Run8 ViT run:

```
docker run --rm --gpus all -v "${PWD}\Dataset\CREMA-D:/app/Dataset2" -v "${PWD}\results:/app/results" -e BATCH_SIZE=8 -e FIXED_SECONDS=3 -e MODEL=convnext model python Run8.py
```

### The 224 ablation grid

On the cluster, `H100_Set_up/run_224.sh` runs the whole
{`mel`, `leaf`} x {`vit`, `cnn_vit`, `convnext`} grid on Run8 at
`TARGET_SIZE=224`, `NUM_EPOCHS=30`, `SEED=42`, `BATCH_SIZE=32` -- six runs per
corpus, each its own process, so one failure does not take the rest down.

```bash
bash H100_Set_up/run_224.sh iemocap both
```

`$1` is the corpus (`iemocap` | `cremad` | `both`), `$2` the frontend
(`mel` | `leaf` | `both`), and anything after is `VAR=VAL` pairs forwarded to
the run -- `bash H100_Set_up/run_224.sh both both NUM_EPOCHS=2` smoke-tests the
wiring.

It sets `PATCH=16` rather than Run8's default 8. Token count is
`(TARGET_SIZE/PATCH)^2`, so at 224 a patch of 8 gives 784 tokens and a 602 MiB
attention map per layer -- which is what OOMed the ViT arms on a shared GPU.
`PATCH=16` gives 196 tokens and 38 MiB, the standard ViT-at-224 tokenization.
Pass `PATCH=8` to reproduce the older behaviour; ConvNeXt ignores it.

It writes under `results/Ablation Study/` rather than the main `results/` tree,
via `RESULTS_ROOT`, so its rows are not appended to the same `sweep.csv` as the
`TARGET_SIZE=64` sweep. The ablation table is then three rows per file -- one
per classifier:

```
results/Ablation Study/MelTorch_ViT/<corpus>/sweep.csv
results/Ablation Study/LeafTorch_ViT/<corpus>/sweep.csv
```

`run_iemocap_224.sh` and `run_cremad_224.sh` are the older per-corpus form of
the same idea, without the `vit` arm and writing into the main `results/`.

### The same grid, cross-validated

`H100_Set_up/run_cv_224.sh` runs the same six combinations under Run9's
5-fold CV, so each comes back as a mean +- std instead of one number. It runs
`POOL=cls`, which is bit-identical to the Run8 model, plus `PATCH=16`,
`TARGET_SIZE=224`, `NUM_EPOCHS=30`, `SEED=42` as above.

**On IEMOCAP the two grids are not comparable row-for-row.** Run8 splits happy
from excited (5 classes); Run9 defaults to `IEMOCAP_CLASSES=4`, the standard
neutral / sad / angry / happy+excited protocol the paper and the wider
literature report on. Different label sets, different chance levels. Each grid
is internally consistent, so both ablations are valid on their own terms --
treat the CV numbers as the result and the single-split ones as the screen.
`IEMOCAP_CLASSES=5` matches Run8 instead, at the cost of the
literature-standard number.

Submit **one frontend per job** -- 3 models x 5 folds = 15 trainings, roughly
8-15h, where both frontends together would blow a 24h wall clock:

```bash
bash H100_Set_up/run_cv_224.sh iemocap mel
```

Killed jobs are cheap to resume: Run9 appends to `folds.csv` per finished fold
and skips folds already recorded, so resubmitting the same command picks up
where the scheduler cut it off. Results land beside the Run8 grid, in
`results/Ablation Study/{Mel,Leaf}Torch_CV/<corpus>/`.

`run_cv.sh` is the different question -- `cls` vs `attn` pooling on identical
folds. This script holds pooling fixed and varies frontend and classifier.

Run9 runs a whole 5-fold CV per invocation:

```
docker run --rm --gpus all -v "${PWD}\Dataset\IEMOCAP:/app/Dataset2" -v "${PWD}\results:/app/results" -e BATCH_SIZE=32 -e FIXED_SECONDS=3 model python Run9.py
```

On the cluster the entry point is `H100_Set_up/run_cv.sh`, which drives the
pooling arms through `run_server.sh`:

```bash
bash H100_Set_up/run_cv.sh iemocap
```

```bash
bash H100_Set_up/run_cv.sh both all NUM_EPOCHS=2
```

The second is a smoke test — two epochs across both arms and both corpora, to
prove the wiring before spending a real allocation.

Rebuild after editing any `Run*.py`, `Model/`, `leaf-audio/`, `Dockerfile` or
`requirements.txt`. Environment variables alone need no rebuild.

### Resuming an interrupted fold

Run9 checkpoints **every epoch** to `last_<tag>p<patch>_fold<k>.pth` -- model,
frontend, optimizer, scheduler and RNG state -- and a fold killed part-way
picks up at the next epoch on the following run instead of starting over. At
most one epoch is ever lost. The file is deleted once the fold's row reaches
`folds.csv`, and `RESUME_FOLD=0` turns the whole mechanism off.

This is what makes a capped job workable: a 3h limit that lands mid-fold used
to throw away the whole hour, because a fold cut off at epoch 25 of 30 wrote
nothing. Now resubmitting the same command continues it.

The checkpoint is keyed by config tag *and* patch size, since `TAG` does not
carry `PATCH` and two patch sizes give different position-embedding shapes.
A checkpoint that cannot be loaded is reported and ignored rather than
crashing the job, and the fold trains from scratch.

### Fitting a capped job (`TIME_BUDGET_MIN`)

Each submission gets a fixed slice -- a wall clock of a few hours and, on this
node, often a MIG partition rather than a whole H100. A `leaf` fold is ~62 min
there, so a 3h job fits two, and the scheduler's kill is not graceful: a fold
cut off at epoch 25 of 30 writes nothing and that hour is gone.

`TIME_BUDGET_MIN` makes `run_cv_224.sh` (and the 64/32 wrappers) run folds one
at a time and stop before starting one it cannot finish:

```bash
bash H100_Set_up/run_cv_64.sh iemocap leaf MODELS=vit TIME_BUDGET_MIN=170
```

Set it a little under the real limit so the last fold can write its row. The
estimate is the longest fold seen so far, not the last -- a fold already in
`folds.csv` returns in seconds and would otherwise make the script start a
real fold with minutes left. Resubmit the identical command to continue;
finished folds are skipped and a `sweep.csv` row appears once all five are in.

It costs one corpus reload per fold, about a minute against sixty. Left unset
(the default), behaviour is exactly as before: Run9 loops the folds itself.

### Running one arm at a time

All four grid scripts take `MODELS` as an argument, so a grid can be cut down
to the arm you actually need -- useful for resuming after a failure, or for
fitting one model into a short job:

```bash
bash H100_Set_up/run_cv_224.sh iemocap mel MODELS=cnn_vit,convnext
```

Comma-separated, not space-separated: arguments are word-split, so
`MODELS="vit convnext"` cannot survive being passed as one. Spaces still work
when set in the environment. Unknown names are rejected before the first run
starts rather than after the corpus has loaded.

Combined with the frontend and corpus parameters, one submission can be as
narrow as a single 5-fold arm.

### The 64 variants

`run_64.sh` and `run_cv_64.sh` are the same two grids at `TARGET_SIZE=64`, for
when 224 will not fit a job's wall clock. They are thin wrappers that pass
every argument through, pinning only the pair that has to move together:
`TARGET_SIZE=64` with `PATCH=8`, giving 64 tokens. Dropping to 64 while
leaving `PATCH=16` would leave 16 tokens, which is not a configuration any
earlier run used.

```bash
bash H100_Set_up/run_cv_64.sh iemocap mel GPU_PICK=force
```

The saving is in the classifier, not the frontend: 64 tokens against 196 is
roughly a tenth of the attention work, but LEAF still convolves the full
48000-sample waveform whatever the target size, so `leaf` runs have a floor
this does not lower. ConvNeXt sees a 2x2 map at 64 against 7x7 at 224 --
valid under `POOL=cls`, which global-average-pools it, but read the number as
a 64-input number.

Rows carry `target_size` and `patch`, so both sizes coexist in one
`sweep.csv`. They are separate experiments -- do not average across them.

### Which GPU a job lands on

The compute node has more than one H100 and the portal jobs are not launched
with `--gres`, so every process sees all of them and torch takes `cuda:0`.
Everyone queues on GPU 0 while the others idle -- the 2026-08-25 run died with
674 MiB free on GPU 0 and two cards completely empty beside it.

Asking the portal for **GPU Per Node: 1** does not change this. It is a
scheduling request; on this cluster it comes with no device cgroup, so every
job still sees every card and nothing stops several sharing one.

`run_server.sh` now reads `nvidia-smi` and pins `CUDA_VISIBLE_DEVICES` to the
emptiest card, printing the table and its choice. It defers to whatever is
already set, so a job that does request `--gres` keeps Slurm's allocation. It
warns when even the best card has under `MIN_FREE_MIB` (8000) free, which is
the signal that an OOM is other tenants rather than the configuration.

`GPU_PICK` controls it: `auto` (default) picks only when
`CUDA_VISIBLE_DEVICES` is unset, `force` picks regardless -- use this when the
inherited allocation points at a card someone else has filled -- and `off`
never picks. The inherited value is echoed on every run, so the log records
which case the node was in:

```bash
bash H100_Set_up/run_cv_224.sh iemocap leaf GPU_PICK=force
```

## What lands in the results folder

One folder per frontend and corpus, e.g.
`results/Ablation Study/LeafTorch_ViT/iemocap/`, shared by every model in a
grid. `sweep.csv` is the table to read: it accumulates one row per completed
run, carrying `frontend`, `model`, `patch`, `target_size`, `seed` and the
scores. Run9 adds `folds.csv`, the five per-fold rows behind each mean, keyed
by the same `tag`.

`training_log.csv` holds one row per epoch, appended across runs. Every row is
stamped with `run`, `frontend`, `model`, `dataset`, `target_size`, `patch` and
`seed` (Run9 also `run_id`, `pool`, `loss_fn`, `fold`), so one arm's curve can
be selected out of a file holding several. Epoch numbers cannot do that job --
they restart at 1 for every run in the file. Rows written before this stamp
existed carry blanks in those columns.

Checkpoints differ between the two scripts. Run9 tags them
(`best_<tag>_fold<k>.pth`, `confusion_<tag>_fold<k>.csv`), so every arm keeps
its own. Run8 does not: `best.pth`, `last.pth`, `leaf_weights.pth` and
`confusion_matrix.csv` are overwritten by each successive model in a grid, so
after a six-run sweep they belong to whichever ran last. The metrics in
`sweep.csv` are unaffected.

## Key environment variables

| var | default | notes |
|---|---|---|
| `NUM_EPOCHS` | 30 | |
| `BATCH_SIZE` | 32 | use 8 for Run6/Run7 |
| `FIXED_SECONDS` | 0 = auto (p75, capped 6s) | one fixed input window for every batch |
| `LR` / `LEAF_LR` | 3e-4 / 1e-5 | frontend learns much slower than the head |
| `MODEL` | `cnn_vit` | see table above (Run6/Run7/Run8) |
| `TARGET_SIZE` | 64 | feature map fed to the classifier; `convnext` needs ≥ 32 |
| `COORD_CHANNELS` `SPEC_AUGMENT` | 1, 1 | Run7 methods; 0 disables |
| `SPLIT_MODE` | `speaker` | `random` leaks speakers — for measuring that leak only |
| `TRAIN_FRAC` | 1.0 | <1 subsamples train only, so val/test stay comparable |
| `DATASET` | `auto` | inferred from the label vocabulary |
| `RESUME` | 0 | 1 continues from the previous run's LEAF weights |
| `RESULTS_ROOT` | `./results` | Run8 and Run9; moves the result tree, keeping the frontend/dataset folders underneath |
| `RESUME_FOLD` | 1 | Run9; 0 restarts an interrupted fold instead of resuming it mid-fold |
| `TIME_BUDGET_MIN` | 0 | `run_cv_*.sh`; minutes allowed, runs folds one at a time and stops before overrunning |

## Data and evaluation

Both corpora are split **speaker-independently** — CREMA-D by actor id (91
actors), IEMOCAP by session+gender (10 speakers) — so no speaker appears in both
train and test.

| dataset | classes | usable clips |
|---|---|---|
| CREMA-D | 6: anger disgust fear neutral sad happy | 7442 |
| IEMOCAP | 5: neutral sad anger happy excited | 6896 of 10039 |

Run9 is the exception to both rows: it is scored by 5-fold CV rather than one
split, and `IEMOCAP_CLASSES` changes the class set (4-way by default, or
`5others`, which keeps all 10039 clips).

IEMOCAP drops frustration/other/surprise/disgust/fear — not in the class set.

Each run appends a row to `sweep.csv` in its results directory with every
parameter and its `test_wa` / `test_uar`. **Compare on `test_uar`** (unweighted
average recall) — that is what the SER literature reports, and weighted accuracy
flatters a model that ignores a small class.
