Write-Output "Starting beta=0.01..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.01 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_001"

Write-Output "Starting beta=0.03..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.03 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_003"

Write-Output "Starting beta=0.05..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.05 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_005"

Write-Output "Starting beta=0.10..."
uv run renju-grpo.py grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" grpo.epochs=5000 grpo.temperature=1.0 grpo.beta=0.1 grpo.mcts_simulations=1000 grpo.use_tss_training=true train.output_root="./artifacts/beta_010"