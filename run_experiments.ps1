# run_experiments.ps1
# PowerShell script for running 4 GRPO experiments

# ① ucb1、tssあり(訓練段階と盤面収集時両方)
Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Starting Experiment 1: ucb1, tss_both..."
uv run renju-grpo.py `
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" `
  grpo.epochs=1000 `
  grpo.temperature=1.0 `
  grpo.beta=0.04 `
  grpo.mcts_simulations=1000 `
  grpo.use_tss_collection=true `
  grpo.use_tss_training=true `
  grpo.use_puct_collection=false `
  grpo.use_puct_training=false `
  train.output_root="./artifacts/exp_ucb1_tss_both"

# ② puct、tssあり(訓練段階と盤面収集時両方)
Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Starting Experiment 2: puct, tss_both..."
uv run renju-grpo.py `
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" `
  grpo.epochs=1000 `
  grpo.temperature=1.0 `
  grpo.beta=0.04 `
  grpo.mcts_simulations=1000 `
  grpo.use_tss_collection=true `
  grpo.use_tss_training=true `
  grpo.use_puct_collection=true `
  grpo.use_puct_training=true `
  train.output_root="./artifacts/exp_puct_tss_both"

# ③ ucb1、tssあり(訓練段階のみ)
Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Starting Experiment 3: ucb1, tss_train_only..."
uv run renju-grpo.py `
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" `
  grpo.epochs=1000 `
  grpo.temperature=1.0 `
  grpo.beta=0.04 `
  grpo.mcts_simulations=1000 `
  grpo.use_tss_collection=false `
  grpo.use_tss_training=true `
  grpo.use_puct_collection=false `
  grpo.use_puct_training=false `
  train.output_root="./artifacts/exp_ucb1_tss_train_only"

# ④ puct、tssあり(訓練段階のみ)
Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] Starting Experiment 4: puct, tss_train_only..."
uv run renju-grpo.py `
  grpo.checkpoint_path="./artifacts/checkpoints/pretrained.pt" `
  grpo.epochs=1000 `
  grpo.temperature=1.0 `
  grpo.beta=0.04 `
  grpo.mcts_simulations=1000 `
  grpo.use_tss_collection=false `
  grpo.use_tss_training=true `
  grpo.use_puct_collection=true `
  grpo.use_puct_training=true `
  train.output_root="./artifacts/exp_puct_tss_train_only"

Write-Output "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] All experiments completed!"
