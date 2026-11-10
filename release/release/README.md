# Compact TSA-HC Reproduction

This directory is the compact MOSI/MOSEI training release. Runtime training
reads normalized PT packages directly; it does not parse action JSONL files or
run a text encoder.

## Files

- `build_data.py`: one-time server conversion from fixed features/actions to PT.
- `data.py`: validates and loads the normalized PT package.
- `model.py`: Predictor, Semantic Projector, and action Selector.
- `train.py`: training, checkpoint selection, evaluation, and inspection.
- `run.sh`: the six paper pipelines and their fixed seed-2026 settings.

`build_data.py` is used only once on the server to create `data/mosi.pt` and
`data/mosei.pt`. It is not imported by runtime training. The exact commands and
fixed paper data sources are documented in `BUILD_DATA.md`.

## PT package

Each dataset has one PT file:

```text
meta
features/{train,test}
  ids, video_ids, sequence_offsets
  audio, vision, text, labels
action_vocab/embeddings
actions/{train,test}/{missing_condition}
  ids, action_ids, anchors
```

The action vocabulary removes duplicate copies of identical text embeddings.
`action_ids` indexes that vocabulary, while sample IDs and anchor indices
remain available for alignment checks.

The submission root contains the sibling directories `release/` and `data/`.
Default locations are `data/mosei.pt` and `data/mosi.pt`. Override them with
`MOSEI_PT` and `MOSI_PT`.

## Training objectives

The Predictor uses non-neutral MSE plus BCE and sign-margin supervision. The
Semantic Projector uses only task MSE and direction alignment:

```text
L_projector = lambda_task * L_task + 0.2 * L_direction
```

The frozen-projector Selector uses only Smooth-L1 TD value loss. Its action
reward is:

```text
R = task_MSE_gain + 0.2 * semantic_direction_gain - 0.03
```

The discount is `0.9`; STOP reward is exactly zero. Policy, behavior-cloning,
oracle, MAE, harm, sign, no-harm, delta, classification, and composition
auxiliary terms are absent.

## Commands

From the project root:

```bash
# Inspect the prepared packages.
bash release/run.sh inspect mosei
bash release/run.sh inspect mosi

# Predictor lineage: MOSEI from scratch, then MOSI from MOSEI epoch 25.
bash release/run.sh predictor mosei
bash release/run.sh predictor mosi

# Semantic Projector and Selector pipelines.
bash release/run.sh semantic mosei single
bash release/run.sh semantic mosei pair
bash release/run.sh semantic mosi single
bash release/run.sh semantic mosi pair

# Run the complete lineage sequentially.
bash release/run.sh all
```

Use `CUDA_VISIBLE_DEVICES` to choose a GPU. `DEVICE`, `OUT_ROOT`, and `PYTHON`
are also configurable environment variables.

## Fixed experiment structure

- Single-missing conditions use five candidates.
- Pair-missing conditions use ten candidates.
- Each dataset and missingness group trains one joint Projector, saves the best
  Projector epoch separately for every target missing condition, and trains a
  target-specific Selector from each corresponding checkpoint.
- MOSI single-missing fixes the Projector task weights to
  `audio=2,text=1,video=2`; missing text uses the clean AV actions embedded in
  `mosi.pt`.
- MOSI pair-missing fixes the Projector task weights to
  `audio_video=2.0,audio_text=0.9,video_text=0.7`.
- Every run uses seed `2026`.



