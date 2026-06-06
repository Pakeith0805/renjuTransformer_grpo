import copy
from pathlib import Path
import torch
from omegaconf import DictConfig

# 既存のモデルクラスとデバイス選択ヘルパーをインポート
from renju_transformer.model import RenjuTransformerModel
from renju_transformer.utils import select_device

def load_policy_and_reference(checkpoint_path: str | Path, device: torch.device):
    """
    事前学習済みのチェックポイントから、更新用の Policy モデルと
    固定用（KL制御用）の Reference モデルをロードして作成します。
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. チェックポイントのロード
    # weights_only=False は、configなどの辞書オブジェクトが含まれているため必要です
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # チェックポイントに保存されている訓練時のconfigを取得
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError("Checkpoint does not contain 'config' field.")
    
    model_cfg = checkpoint_config["model"]

    # 2. Policy モデルの作成と重みのロード
    policy_model = RenjuTransformerModel(
        vocab_size=model_cfg["token_vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        norm_first=model_cfg["norm_first"],
        num_move_labels=model_cfg["num_move_labels"],
    )
    policy_model.load_state_dict(checkpoint["model_state_dict"])
    policy_model.to(device)

    # 3. Reference モデルの作成 (Policy モデルを複製)
    # copy.deepcopy を使うことで、同じ重みを持った独立したモデルインスタンスを作成できます
    ref_model = copy.deepcopy(policy_model)
    ref_model.to(device)

    # 4. Reference モデルのパラメータを固定 (勾配計算を無効化)
    # これにより、誤差逆伝播の計算対象から外れ、メモリ消費と計算量を節約できます
    for param in ref_model.parameters():
        param.requires_grad = False

    # 5. 各モデルのモード設定
    policy_model.train()  # Policy は更新するため学習モード
    ref_model.eval()      # Reference は推論のみのため評価モード (Dropout等を無効化)

    return policy_model, ref_model


# 動作確認用のメイン処理の例
if __name__ == "__main__":
    # デバイスの自動選択 (CUDAがあればGPU、無ければCPU)
    device = select_device("auto")
    
    checkpoint_file = "artifacts/checkpoints/best_model.pt"
    
    print(f"Loading models from {checkpoint_file} on {device}...")
    policy, ref = load_policy_and_reference(checkpoint_file, device)
    
    print("Policy Model parameters requiring grad:")
    # Policyはパラメータ更新が必要なので True
    print(any(p.requires_grad for p in policy.parameters()))  # -> True になるべき
    
    print("Reference Model parameters requiring grad:")
    # Referenceはパラメータ固定なので False
    print(any(p.requires_grad for p in ref.parameters()))    # -> False になるべき