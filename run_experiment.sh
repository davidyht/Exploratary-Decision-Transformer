#!/bin/bash
set -e  # Stop if any command fails

# ========== Configuration ==========
ENV_NAME="bandit"
H=100
DIM=3
VAR=0.1
N_ENVS=100000
N_ENVS_EVAL=1000
EMBD=32
HEAD=1
LAYER=4
N_EVAL=1000
EXPLORATION=100.0
LOG_DIR="logs"

mkdir -p $LOG_DIR

# ========== Step 1: Create dataset ==========
echo "🚀 [1/4] Collecting dataset..."
python collect_data.py \
    --envs $N_ENVS \
    --envs_eval $N_ENVS_EVAL \
    --env "$ENV_NAME" \
    --H $H \
    --dim $DIM \
    --var $VAR \
    | tee $LOG_DIR/data_collection.log

# ========== Step 2: Pretrain DPT ==========
echo "🎯 [2/4] Pretraining DPT..."
python train.py \
    --class "dpt" \
    --embd $EMBD \
    --head $HEAD \
    --layer $LAYER \
    | tee $LOG_DIR/train_dpt.log

# ========== Step 3: Pretrain PPT ==========
echo "🧠 [3/4] Pretraining PPT..."
python train.py \
    --class "ppt" \
    --exploration_rate $EXPLORATION \
    --embd $EMBD \
    --head $HEAD \
    --layer $LAYER \
    | tee $LOG_DIR/train_ppt.log

# ========== Step 4: Evaluation ==========
echo "📊 [4/4] Evaluating models..."
python eval.py --n_eval $N_EVAL \
    | tee $LOG_DIR/eval.log

echo "✅ All done!"
