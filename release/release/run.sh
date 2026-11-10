#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT="$HERE"
PYTHON=${PYTHON:-python3}
DEVICE=${DEVICE:-cuda}
SEED=2026

DATA_DIR=${DATA_DIR:-"$PROJECT/data"}
OUT_ROOT=${OUT_ROOT:-"$PROJECT/outputs"}
MOSEI_PT=${MOSEI_PT:-"$DATA_DIR/mosei.pt"}
MOSI_PT=${MOSI_PT:-"$DATA_DIR/mosi.pt"}

MOSEI_ROOT="$OUT_ROOT/mosei"
MOSI_ROOT="$OUT_ROOT/mosi"

run_train() {
  "$PYTHON" -u "$HERE/train.py" "$@"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

train_mosei_predictor() {
  require_file "$MOSEI_PT"
  run_train predictor \
    --data "$MOSEI_PT" \
    --out-dir "$MOSEI_ROOT/predictor" \
    --conditions A,V,T,AV,AT,VT,AVT \
    --condition-weights A=1.5,V=1.5,T=1,AV=1.5,AT=1,VT=1,AVT=1 \
    --d-model 1024 \
    --context-layers 1 \
    --dropout 0.3 \
    --lambda-cls 0.3 \
    --lambda-sign 0.2 \
    --sign-margin 0.2 \
    --epochs 80 \
    --batch-size 64 \
    --eval-batch-size 64 \
    --lr 5e-5 \
    --weight-decay 1e-4 \
    --save-epochs 13,25,40,60,80 \
    --seed "$SEED" \
    --device "$DEVICE"
}

train_mosi_predictor() {
  require_file "$MOSI_PT"
  require_file "$MOSEI_ROOT/predictor/epoch_025.pt"
  run_train predictor \
    --data "$MOSI_PT" \
    --out-dir "$MOSI_ROOT/predictor" \
    --init-checkpoint "$MOSEI_ROOT/predictor/epoch_025.pt" \
    --conditions A,V,T,AV,AT,VT,AVT \
    --condition-weights A=1,V=1,T=1,AV=1.5,AT=1,VT=1,AVT=1 \
    --d-model 1024 \
    --context-layers 1 \
    --dropout 0.3 \
    --lambda-cls 0.3 \
    --lambda-sign 0.2 \
    --sign-margin 0.2 \
    --epochs 80 \
    --batch-size 64 \
    --eval-batch-size 64 \
    --lr 1e-5 \
    --weight-decay 1e-4 \
    --save-epochs 10,20,30,40,50,60 \
    --seed "$SEED" \
    --device "$DEVICE"
}

train_mosei_single() {
  require_file "$MOSEI_PT"
  require_file "$MOSEI_ROOT/predictor/best.pt"
  local root="$MOSEI_ROOT/single"
  local projector="$root/projector"

  run_train projector \
    --data "$MOSEI_PT" \
    --predictor-checkpoint "$MOSEI_ROOT/predictor/best.pt" \
    --out-dir "$projector" \
    --missing audio,text,video \
    --select-target audio \
    --lambda-task 1 \
    --lambda-direction 0.2 \
    --epochs 60 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --device "$DEVICE"

  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_audio.pt" "$root/VT" audio \
    "audio=1,text=0,video=0"
  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_text.pt" "$root/AV" text \
    "audio=0,text=1,video=0"
  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_video.pt" "$root/AT" video \
    "audio=0,text=0,video=1"
}

train_mosei_pair() {
  require_file "$MOSEI_PT"
  require_file "$MOSEI_ROOT/predictor/best.pt"
  local root="$MOSEI_ROOT/pair"
  local projector="$root/projector"

  run_train projector \
    --data "$MOSEI_PT" \
    --predictor-checkpoint "$MOSEI_ROOT/predictor/best.pt" \
    --out-dir "$projector" \
    --missing audio_video,audio_text,video_text \
    --select-target audio_video \
    --lambda-task 1 \
    --lambda-direction 0.2 \
    --epochs 60 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --device "$DEVICE"

  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_audio_video.pt" "$root/T" audio_video \
    "audio_video=1,audio_text=0,video_text=0"
  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_audio_text.pt" "$root/V" audio_text \
    "audio_video=0,audio_text=1,video_text=0"
  run_target_selector "$MOSEI_PT" "$MOSEI_ROOT/predictor/best.pt" \
    "$projector/best_video_text.pt" "$root/A" video_text \
    "audio_video=0,audio_text=0,video_text=1"
}

run_target_selector() {
  local data=$1
  local predictor=$2
  local projector=$3
  local root=$4
  local target=$5
  local weights=$6
  local epochs=${7:-60}
  local train_missing=${8:-"$(selector_group "$target")"}

  run_train selector \
    --data "$data" \
    --predictor-checkpoint "$predictor" \
    --projector-checkpoint "$projector" \
    --out-dir "$root/selector" \
    --missing "$train_missing" \
    --select-target "$target" \
    --task-weights "$weights" \
    --semantic-weights "$weights" \
    --max-steps 5 \
    --rollout-steps 5 \
    --gamma 0.9 \
    --lambda-semantic-reward 0.2 \
    --step-cost 0.03 \
    --epochs "$epochs" \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --device "$DEVICE"

  run_train evaluate \
    --data "$data" \
    --checkpoint "$root/selector/best.pt" \
    --missing "$target" \
    --split test \
    --policy q \
    --rollout-steps 5 \
    --eval-batch-size 32 \
    --metrics-json "$root/metrics.json" \
    --seed "$SEED" \
    --device "$DEVICE"
}

selector_group() {
  case "$1" in
    audio|text|video)
      printf '%s' "audio,text,video"
      ;;
    audio_video|audio_text|video_text)
      printf '%s' "audio_video,audio_text,video_text"
      ;;
    *)
      echo "unknown target condition: $1" >&2
      exit 1
      ;;
  esac
}

train_mosi_single() {
  require_file "$MOSI_PT"
  require_file "$MOSI_ROOT/predictor/best.pt"
  local root="$MOSI_ROOT/single"
  local projector="$root/projector"

  run_train projector \
    --data "$MOSI_PT" \
    --predictor-checkpoint "$MOSI_ROOT/predictor/best.pt" \
    --out-dir "$projector" \
    --missing audio,text,video \
    --select-target audio \
    --task-weights audio=2.2,text=1.0,video=1.0 \
    --lambda-task 1 \
    --lambda-direction 0.2 \
    --epochs 60 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --device "$DEVICE"

  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$projector/best_audio.pt" "$root/VT" audio \
    "audio=1,text=0,video=0"
  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$projector/best_text.pt" "$root/AV" text \
    "audio=0,text=1,video=0"
  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$projector/best_video.pt" "$root/AT" video \
    "audio=0,text=0,video=1"
}

train_mosi_pair() {
  require_file "$MOSI_PT"
  require_file "$MOSI_ROOT/predictor/best.pt"
  local root="$MOSI_ROOT/pair"

  run_train projector \
    --data "$MOSI_PT" \
    --predictor-checkpoint "$MOSI_ROOT/predictor/best.pt" \
    --out-dir "$root/projector" \
    --missing audio_video,audio_text,video_text \
    --select-target audio_video \
    --task-weights audio_video=1.2,audio_text=0.8,video_text=1.0 \
    --lambda-task 1 \
    --lambda-direction 0.2 \
    --epochs 60 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --seed "$SEED" \
    --device "$DEVICE"

  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$root/projector/best_audio_video.pt" "$root/T" audio_video \
    "audio_video=1,audio_text=0,video_text=0" 60
  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$root/projector/best_audio_text.pt" "$root/V" audio_text \
    "audio_video=0,audio_text=1,video_text=0" 60
  run_target_selector "$MOSI_PT" "$MOSI_ROOT/predictor/best.pt" \
    "$root/projector/best_video_text.pt" "$root/A" video_text \
    "audio_video=0,audio_text=0,video_text=1" 60
}

usage() {
  cat <<'EOF'
Usage:
  bash release/run.sh
  bash release/run.sh predictor mosei|mosi
  bash release/run.sh semantic mosei|mosi single|pair
  bash release/run.sh inspect mosei|mosi

Environment:
  DATA_DIR, MOSEI_PT, MOSI_PT, OUT_ROOT, DEVICE, PYTHON
EOF
}

command=${1:-all}
dataset=${2:-}
group=${3:-}

case "$command:$dataset:$group" in
  all::)
    train_mosei_predictor
    train_mosei_single
    train_mosei_pair
    train_mosi_predictor
    train_mosi_single
    train_mosi_pair
    ;;
  predictor:mosei:)
    train_mosei_predictor
    ;;
  predictor:mosi:)
    train_mosi_predictor
    ;;
  semantic:mosei:single)
    train_mosei_single
    ;;
  semantic:mosei:pair)
    train_mosei_pair
    ;;
  semantic:mosi:single)
    train_mosi_single
    ;;
  semantic:mosi:pair)
    train_mosi_pair
    ;;
  inspect:mosei:)
    require_file "$MOSEI_PT"
    run_train inspect --data "$MOSEI_PT"
    ;;
  inspect:mosi:)
    require_file "$MOSI_PT"
    run_train inspect --data "$MOSI_PT"
    ;;
  *)
    usage
    exit 2
    ;;
esac
