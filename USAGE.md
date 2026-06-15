# versus
uv run python scripts/play_versus.py versus.model_a_path=<一つ目のモデルのパス> versus.model_b_path=<二つ目のモデルのパス> versus.num_games=2 versus.temperature=0

# 初めからrun
uv run renju-grpo.py grpo.checkpoint_path="./artifacts\checkpoints\pretrained.pt" grpo.epochs=3000 grpo.temperature=1.0 grpo.beta=0.01 grpo.mcts_simulations=1000     

# 続きからrun
uv run renju-grpo.py grpo.checkpoint_path="./artifacts\grpo_checkpoint_600.pt" grpo.epochs=1000 grpo.temperature=1.0 grpo.beta=0.01 grpo.mcts_simulations=1000  +mlflow.run_id="05f67dc4ff15471088cff0d59fbb8774"

# onnxファイルの作り方
uv run .\scripts\export_onnx.py --checkpoint artifacts\grpo_checkpoint_4750.pt --output docs/renju_transformer_4750.onnx    

# 対戦シミュレーション
uv run -m http.server 8000 --directory docs                                                                    
# mlflow
uv run mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5002