#!/bin/bash
# エラーが発生しても次のコマンドを続けて実行するようにします。
# もし「前のコマンドが失敗したらそこで止めたい」場合は、直下に set -e を記述してください。

# ==============================================================================
# 実験共通設定:
# - grpo.beta=0.04
# - grpo.epochs=1000
# - grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt"
# ==============================================================================

# ① ucb1、tssあり(訓練段階と盤面収集時両方)
echo "[$(date)] Starting Experiment 1: ucb1, tss_both..."
uv run renju-grpo.py \
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" \
  grpo.epochs=1000 \
  grpo.temperature=1.0 \
  grpo.beta=0.04 \
  grpo.mcts_simulations=1000 \
  grpo.use_tss_collection=true \
  grpo.use_tss_training=true \
  grpo.use_puct_collection=false \
  grpo.use_puct_training=false \
  train.output_root="./artifacts/exp_ucb1_tss_both"

# ② puct、tssあり(訓練段階と盤面収集時両方)
echo "[$(date)] Starting Experiment 2: puct, tss_both..."
uv run renju-grpo.py \
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" \
  grpo.epochs=1000 \
  grpo.temperature=1.0 \
  grpo.beta=0.04 \
  grpo.mcts_simulations=1000 \
  grpo.use_tss_collection=true \
  grpo.use_tss_training=true \
  grpo.use_puct_collection=true \
  grpo.use_puct_training=true \
  train.output_root="./artifacts/exp_puct_tss_both"

# ③ ucb1、tssあり(訓練段階のみ)
echo "[$(date)] Starting Experiment 3: ucb1, tss_train_only..."
uv run renju-grpo.py \
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" \
  grpo.epochs=1000 \
  grpo.temperature=1.0 \
  grpo.beta=0.04 \
  grpo.mcts_simulations=1000 \
  grpo.use_tss_collection=false \
  grpo.use_tss_training=true \
  grpo.use_puct_collection=false \
  grpo.use_puct_training=false \
  train.output_root="./artifacts/exp_ucb1_tss_train_only"

# ④ puct、tssあり(訓練段階のみ)
echo "[$(date)] Starting Experiment 4: puct, tss_train_only..."
uv run renju-grpo.py \
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" \
  grpo.epochs=1000 \
  grpo.temperature=1.0 \
  grpo.beta=0.04 \
  grpo.mcts_simulations=1000 \
  grpo.use_tss_collection=false \
  grpo.use_tss_training=true \
  grpo.use_puct_collection=true \
  grpo.use_puct_training=true \
  train.output_root="./artifacts/exp_puct_tss_train_only"

echo "[$(date)] All experiments completed!"
