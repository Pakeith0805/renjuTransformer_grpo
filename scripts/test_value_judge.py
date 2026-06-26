#!/usr/bin/env python
"""value_judge_rewards の dry-run。

value checkpoint があればそれを、無ければ pretrained に value ヘッド(ランダム)を付けて
コードパスを検証する。TSS 上書き(即勝ち→+1)は value ヘッドに依らず決定論なので確認できる。

  uv run python scripts/test_value_judge.py --value models/pretrained_value.pt
  uv run python scripts/test_value_judge.py --pretrained models/pretrained.pt   # ダミー検証
"""
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.utils import select_device
from grpo.agent import GRPOAgent


def build_value_model(path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    model = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_layers=mc["num_layers"], dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"], activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"], with_value_head=True,
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)  # value 無ければランダムのまま
    return model.to(device).eval()


def idx(r, c):
    return r * 15 + c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default=None, help="value checkpoint (あれば)")
    ap.add_argument("--pretrained", default="models/pretrained.pt", help="value 無いとき胴体に使う")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = select_device(args.device)
    tok = RenjuTokenizer(sep_token_id=228, move_id_offset=3)
    src = args.value or args.pretrained
    print(f"loading {src} (value head {'real' if args.value else 'random dummy'})")
    vm = build_value_model(src, device)
    agent = GRPOAgent(policy_model=vm, ref_model=vm, tokenizer=tok, device=device, value_model=vm)

    # --- テストA: 黒の即勝ち手がある盤面 (parity 黒=白=4 → 黒手番) ---
    board = [0] * 225
    for p in [(7, 4), (7, 5), (7, 6), (7, 7)]:
        board[idx(*p)] = 1                       # 黒の四
    board[idx(7, 3)] = 2                          # 白ブロッカー
    for p in [(0, 0), (0, 14), (14, 0)]:
        board[idx(*p)] = 2                        # 白フィラー (孤立)
    win = idx(7, 8)                               # ここで五完成
    moves = [win, idx(0, 7), idx(10, 10), win]    # 勝ち手 + 普通の手2つ + 重複
    rewards, _ = agent.value_judge_rewards(board, moves)
    print("テストA rewards:", [round(r, 3) for r in rewards])
    assert rewards[0] == 1.0 and rewards[3] == 1.0, "即勝ち手が +1 になっていない(TSS上書き失敗)"
    assert all(-1.0 <= r <= 1.0 for r in rewards), "報酬が [-1,1] を外れている"
    print("  -> 即勝ち手=+1 (TSS上書きOK), 重複も同値, 全て[-1,1]内")

    # --- テストB: 空盤(戦術なし)で value 経路が動くか ---
    empty = [0] * 225
    mv2 = [idx(7, 7), idx(7, 8), idx(8, 8)]
    r2, _ = agent.value_judge_rewards(empty, mv2)
    print("テストB rewards(空盤,value経路):", [round(r, 3) for r in r2])
    assert len(r2) == 3 and all(-1.0 <= r <= 1.0 for r in r2)
    print("  -> value 一括評価が動作, 長さ一致, [-1,1]内")

    print("\nOK: value_judge_rewards のコードパス検証 完了")


if __name__ == "__main__":
    main()
