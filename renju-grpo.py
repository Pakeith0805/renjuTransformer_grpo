# renju-grpo.py
from __future__ import annotations
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# src ディレクトリを検索パスに追加
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# 作成した grpo モジュールからロード関数をインポート
from grpo.load_model import load_policy_and_reference
from renju_transformer.utils import select_device

@hydra.main(version_base="1.3", config_path="config", config_name="config_grpo")
def main(cfg: DictConfig) -> None:
    # 1. 読み込まれた設定をコンソールに表示して確認
    print("--- Loaded GRPO Configuration ---")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    
    # 2. デバイスの決定
    device = select_device(cfg.train.device)
    
    # 3. 事前学習モデルのロード (Policy と Reference)
    print(f"Initializing models from checkpoint: {cfg.grpo.checkpoint_path}...")
    policy_model, ref_model = load_policy_and_reference(cfg.grpo.checkpoint_path, device)
    print("Models loaded successfully!")
    
    # TODO: ここに強化学習ループ (データ収集 -> 損失計算 -> 更新) を後から追加していく

if __name__ == "__main__":
    main()