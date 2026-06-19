#!/bin/bash
# エラーが発生しても次のコマンドを続けて実行するようにします。
# もし「前のコマンドが失敗したらそこで止めたい」場合は、直下に set -e を記述してください。

echo "[$(date)] Starting beta=0.01..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.01 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_001"

echo "[$(date)] Starting beta=0.03..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.03 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_003"

echo "[$(date)] Starting beta=0.05..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.05 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_005"

echo "[$(date)] Starting beta=0.10..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.1 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_010"

echo "[$(date)] All training completed!"