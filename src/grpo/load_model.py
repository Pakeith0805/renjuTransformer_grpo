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