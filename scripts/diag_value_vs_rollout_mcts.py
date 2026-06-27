#!/usr/bin/env python
"""
診断: MCTS の葉評価 rollout vs value-net (同一 C++ 木で葉だけ差し替え)
====================================================================
同じ run_mcts(C++) を使い、葉評価だけを「ロールアウト」↔「value net」で切替えて
value-MCTS vs rollout-MCTS を head-to-head 対戦させる。

  value-MCTS が rollout-MCTS に勝ち越す → ロールアウトが MCTS のボトルネックだった、の証拠。

葉以外(木・PUCT・候補生成・(任意)TSS)は完全に同条件なので、leaf 評価の差だけを切り分けられる。

前提: mcts.cpp を再ビルドして set_value_fn_c_api / clear_value_fn_c_api を含む mcts.so/dll が必要。
  (Linux 例)  g++ -O3 -shared -fPIC -static-libgcc -static-libstdc++ -o mcts.so mcts.cpp

使い方:
  uv run python scripts/diag_value_vs_rollout_mcts.py \
      --value models/pretrained_value.pt --policy models/pretrained.pt \
      --games 100 --sims 200 --use-tss
"""
import argparse
import ctypes
import sys
import random
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.rules import infer_player, winner_after_move, board_with_move  # noqa: F401
from renju_transformer.utils import select_device

N = 225
BLACK, WHITE = 1, 2
VALUE_FN = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.POINTER(ctypes.c_int), ctypes.c_int)


def load_lib():
    name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
    lib = ctypes.CDLL(str(PROJECT_ROOT / name))
    lib.run_mcts_c_api_with_policy_and_visits.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_uint64,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.c_int,
    ]
    lib.run_mcts_c_api_with_policy_and_visits.restype = ctypes.c_double
    lib.solve_vcf_c_api.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    lib.solve_vcf_c_api.restype = ctypes.c_int
    lib.set_value_fn_c_api.argtypes = [VALUE_FN]
    lib.set_value_fn_c_api.restype = None
    lib.clear_value_fn_c_api.argtypes = []
    lib.clear_value_fn_c_api.restype = None
    return lib


def build_model(path, device, with_value):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    mc = ck["config"]["model"]
    m = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_layers=mc["num_layers"], dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"], activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"], with_value_head=with_value,
    )
    m.load_state_dict(ck["model_state_dict"], strict=False)
    return m.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="models/pretrained_value.pt")
    ap.add_argument("--policy", default="models/pretrained.pt", help="PUCT prior 用(両者共通)")
    ap.add_argument("--games", type=int, default=100)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--use-puct", action="store_true", default=True)
    ap.add_argument("--use-tss", action="store_true", help="root で VCF(必勝/受け)を両者同条件で適用")
    ap.add_argument("--max-vcf-depth", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    device = select_device(args.device)
    tok = RenjuTokenizer(228, 3)
    lib = load_lib()
    policy = build_model(args.policy, device, with_value=False)
    value = build_model(args.value, device, with_value=True)

    # --- value 葉評価コールバック (C++ から1葉ごとに呼ばれる。GC されないよう参照保持) ---
    @VALUE_FN
    def value_cb(board_ptr, player):
        board = [int(board_ptr[i]) for i in range(N)]
        with torch.no_grad():
            ids = tok.encode_input(board).unsqueeze(0).to(device)
            _, v = value(ids, return_value=True)
        return float(v.item())

    def prior_of(board):
        legal = tok.legal_move_mask(board).to(device)
        ids = tok.encode_input(board).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(ids).squeeze(0).masked_fill(~legal, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, legal

    def mcts_move(board, use_value):
        """同一 MCTS で葉だけ切替えて着手を1つ返す。"""
        cur = infer_player(board)
        opp = WHITE if cur == BLACK else BLACK
        arr = (ctypes.c_int * N)(*board)
        if args.use_tss:
            mv = lib.solve_vcf_c_api(arr, cur, args.max_vcf_depth)
            if mv >= 0:
                return mv
            mv = lib.solve_vcf_c_api(arr, opp, args.max_vcf_depth)
            if mv >= 0:
                return mv
        probs, legal = prior_of(board)
        if use_value:
            lib.set_value_fn_c_api(value_cb)
        else:
            lib.clear_value_fn_c_api()
        visits = (ctypes.c_int * N)()
        probs_arr = (ctypes.c_double * N)(*probs.tolist())
        lib.run_mcts_c_api_with_policy_and_visits(
            arr, args.sims, random.randint(0, 2**63 - 1), probs_arr, visits, 1 if args.use_puct else 0)
        lib.clear_value_fn_c_api()
        legal_idx = [i for i, ok in enumerate(legal.tolist()) if ok]
        return max(legal_idx, key=lambda i: visits[i])

    def play_game(value_is_black):
        board = [0] * N
        for ply in range(1, N + 1):
            player = infer_player(board)
            cur_uses_value = (player == BLACK) == value_is_black
            mv = mcts_move(board, cur_uses_value)
            board[mv] = player
            w = winner_after_move(board, mv, player)
            if w is not None:
                return "value" if ((w == BLACK) == value_is_black) else "rollout"
            if all(c != 0 for c in board):
                return "draw"
        return "draw"

    print(f"diag: value-leaf MCTS vs rollout-leaf MCTS  (sims={args.sims}, games={args.games}, "
          f"use_tss={args.use_tss}, device={device})", file=sys.stderr)
    res = {"value": 0, "rollout": 0, "draw": 0}
    for g in range(args.games):
        winner = play_game(value_is_black=(g % 2 == 0))   # 先後を交互に
        res[winner] += 1
        if (g + 1) % 10 == 0:
            print(f"  {g+1}/{args.games}  value={res['value']} rollout={res['rollout']} draw={res['draw']}",
                  file=sys.stderr)

    tot = args.games
    print("\n==== 結果 ====")
    print(f"value-leaf MCTS 勝ち:   {res['value']}/{tot} ({res['value']/tot*100:.1f}%)")
    print(f"rollout-leaf MCTS 勝ち: {res['rollout']}/{tot} ({res['rollout']/tot*100:.1f}%)")
    print(f"draw: {res['draw']}")
    print("\n>50% で value 葉が rollout 葉より強い = ロールアウトが MCTS のボトルネックだった")


if __name__ == "__main__":
    main()
