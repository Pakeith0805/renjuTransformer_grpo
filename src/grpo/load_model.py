import copy
from pathlib import Path
import torch
from torch.distributions import Categorical  # 追加
from omegaconf import DictConfig

# 既存のモデルクラスとデバイス選択ヘルパーをインポート
from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer  # 追加
from grpo.agent import GRPOAgent
from renju_transformer.utils import select_device

def load_policy_and_reference(policy_checkpoint_path: str | Path, ref_checkpoint_path: str | Path, device: torch.device):
    """
    更新用の Policy モデルを policy_checkpoint_path からロードし、
    固定用（KL制御用）の Reference モデルを ref_checkpoint_path からロードして作成します。
    """
    policy_checkpoint_path = Path(policy_checkpoint_path)
    ref_checkpoint_path = Path(ref_checkpoint_path)
    
    if not policy_checkpoint_path.exists():
        raise FileNotFoundError(f"Policy checkpoint not found at: {policy_checkpoint_path}")
    if not ref_checkpoint_path.exists():
        raise FileNotFoundError(f"Reference checkpoint not found at: {ref_checkpoint_path}")

    # 1. Policy チェックポイントのロード
    policy_checkpoint = torch.load(policy_checkpoint_path, map_location=device, weights_only=False)
    policy_config = policy_checkpoint.get("config")
    if policy_config is None:
        raise ValueError("Policy checkpoint does not contain 'config' field.")
    
    model_cfg = policy_config["model"]

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
    policy_model.load_state_dict(policy_checkpoint["model_state_dict"])
    policy_model.to(device)

    # 3. Reference チェックポイントのロードとモデルの作成
    ref_checkpoint = torch.load(ref_checkpoint_path, map_location=device, weights_only=False)
    ref_config = ref_checkpoint.get("config")
    if ref_config is None:
        raise ValueError("Reference checkpoint does not contain 'config' field.")
        
    ref_model_cfg = ref_config["model"]
    ref_model = RenjuTransformerModel(
        vocab_size=ref_model_cfg["token_vocab_size"],
        max_seq_len=ref_model_cfg["max_seq_len"],
        d_model=ref_model_cfg["d_model"],
        nhead=ref_model_cfg["nhead"],
        num_layers=ref_model_cfg["num_layers"],
        dim_feedforward=ref_model_cfg["dim_feedforward"],
        dropout=ref_model_cfg["dropout"],
        activation=ref_model_cfg["activation"],
        norm_first=ref_model_cfg["norm_first"],
        num_move_labels=ref_model_cfg["num_move_labels"],
    )
    ref_model.load_state_dict(ref_checkpoint["model_state_dict"])
    ref_model.to(device)

    # 4. Reference モデルのパラメータを固定 (勾配計算を無効化)
    for param in ref_model.parameters():
        param.requires_grad = False

    # 5. 各モデルのモード設定
    policy_model.train()  # Policy は更新するため学習モード
    ref_model.eval()      # Reference は推論のみのため評価モード (Dropout等を無効化)

    return policy_model, ref_model

def load_value_model(value_checkpoint_path: str | Path, device: torch.device):
    """value ヘッド付きの判定者モデルをロードして固定(eval, requires_grad=False)で返す。"""
    value_checkpoint_path = Path(value_checkpoint_path)
    if not value_checkpoint_path.exists():
        raise FileNotFoundError(f"Value checkpoint not found at: {value_checkpoint_path}")
    ckpt = torch.load(value_checkpoint_path, map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    model = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_layers=mc["num_layers"], dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"], activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"], with_value_head=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device).eval()
    return model


def print_board(board_state: list[int]):
    """
    15x15 の五目並べ盤面をターミナルに綺麗なテキストグリッドとして描画します。
    """
    # 盤面記号のマッピング (空マス: '.', 黒石: '●', 白石: '○')
    symbols = {0: " . ", 1: " ● ", 2: " ○ "}
    
    print("\n   " + " ".join(f"{col:2d}" for col in range(15)))  # 列番号ヘッダー
    print("  " + "-" * 46)
    
    for row in range(15):
        row_str = f"{row:2d} |"
        for col in range(15):
            cell_idx = row * 15 + col
            row_str += symbols[board_state[cell_idx]]
        print(row_str)
    print("  " + "-" * 46 + "\n")


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

    # トークナイザの準備
    # デフォルトの引数値 (sep_token_id=228, move_id_offset=3) を指定
    tokenizer = RenjuTokenizer(sep_token_id=228, move_id_offset=3)
    
    # テスト用の空盤面 (225個の0) を用意
    board = [0] * 225
    
    # 8個の手をサンプリング (動作テストのため温度を少し高めの 1.2 にして手をばらけさせます)
    print("\nSampling 8 actions from the empty board...")

    agent = GRPOAgent(policy_model=policy, ref_model=ref, tokenizer=tokenizer, device=device)

    actions, log_pi, log_ref = agent.get_group_actions(
        board_state=board,
        group_size=8,
        temperature=1.2
    )
    
    # 5. 【新規追加】サンプリングされた 8 個の手から、それぞれ対局を最後まで走らせる
    print("\nRunning self-play rollouts for the 8 actions...")
    rewards = []
    
    for i in range(8):
        action_idx = actions[i].item()
        move_id = tokenizer.index_to_move_id(action_idx)
        
        # 進行状況がわかるように print しながら実行します
        print(f"  - Playing Game {i+1}/8 (First Move ID: {move_id})... ", end="", flush=True)
        
        # ロールアウトメソッドの実行
        reward, final_board = agent.rollout_single_game(
            initial_board_state=board,
            first_move_idx=action_idx,
            temperature=1.0  # 対局中の行動の多様性 (1.0 が標準です)
        )

        print_board(final_board)
        
        rewards.append(reward)
        print(f"Finished. Reward: {reward}")
        
    print("-" * 50)
    print(f"All Rollouts Completed! Rewards: {rewards}")
    print("-" * 50) 