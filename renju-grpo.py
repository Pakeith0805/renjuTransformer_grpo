# renju-grpo.py
from __future__ import annotations
import sys
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig
import mlflow

# src ディレクトリを Python の検索パスに追加
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 自作した GRPO 関連のクラス・関数をインポート
from grpo.agent import GRPOAgent
from grpo.trainer import GRPOTrainer
from grpo.load_model import load_policy_and_reference

# 既存の tokenizer と便利関数をインポート
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.utils import (
    ensure_mlflow_experiment,
    select_device,
    set_seed,
)

@hydra.main(version_base="1.3", config_path="config", config_name="config_grpo")
def main(cfg: DictConfig) -> None:
    # 1. 乱数シードの設定
    set_seed(cfg.seed)
    
    # 2. デバイス（CUDA または CPU）の自動決定
    device = select_device(cfg.train.device)
    
    # 3. トークナイザの準備
    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    
    # 4. 事前学習モデル (Policy & Reference) のロード
    print(f"Loading checkpoint from: {cfg.grpo.checkpoint_path}")
    policy_model, ref_model = load_policy_and_reference(cfg.grpo.checkpoint_path, device)
    
    # 4.5. チェックポイントから開始イテレーションを推測
    start_iteration = 1
    try:
        checkpoint = torch.load(cfg.grpo.checkpoint_path, map_location="cpu", weights_only=False)
        if "iteration" in checkpoint:
            start_iteration = checkpoint["iteration"] + 1
            print(f"Resuming training from iteration: {start_iteration}")
    except Exception:
        pass
    
    # 5. オプティマイザの初期化
    # 【重要】更新対象として policy_model のパラメータ「のみ」を渡します
    # (ref_model のパラメータは渡さないことで、確実に固定させます)
    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        betas=tuple(cfg.optimizer.betas),
        eps=cfg.optimizer.eps,
    )
    
    # 6. Agent と Trainer のインスタンス化
    mcts_simulations = cfg.grpo.get("mcts_simulations", 200) # cfgからシミュレーション回数をとってくる。あればその値を、なければ200を。
    agent = GRPOAgent(
        policy_model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        device=device,
        mcts_simulations=mcts_simulations
    )
    
    trainer = GRPOTrainer(
        agent=agent,
        optimizer=optimizer,
        cfg=cfg
    )
    
    # 7. MLflow の初期設定 (データベースや実験名の確認)
    ensure_mlflow_experiment(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=cfg.mlflow.experiment_name,
        artifact_root=cfg.mlflow.artifact_root,
    )
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    
    # 8. 強化学習 (GRPO) のループをスタート！
    # (繰り返し回数は config_grpo.yaml で指定した epochs 数になります)
    trainer.train(
        num_iterations=cfg.grpo.epochs,
        save_every=50,
        run_id=cfg.mlflow.get("run_id", None),
        start_iteration=start_iteration
    )

if __name__ == "__main__":
    main()