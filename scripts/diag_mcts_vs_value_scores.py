#!/usr/bin/env python
"""
診断: 1戦を通して MCTS スコアと value-net スコアを並べて見る
============================================================
1局を rollout-leaf MCTS(PUCT prior + 任意 TSS) で実際に打たせ、各手番で「着手の直前」に
  - MCTS の root 勝率   = run_mcts の返り値 root.wins/root.visits  (手番側視点 [0,1])
  - value net のスコア  = (tanh value + 1)/2                       (手番側視点 [0,1])
を **同じ盤面に対して対で** 記録する。どちらも「手番側がどれくらい勝ちそうか」を表すので直接並べられる。

出力:
  - 標準出力に表 (ply / 手番 / mcts / value / 差 / 着手)
  - CSV  (--out-csv)
  - PNG 折れ線 (--out-png, matplotlib があれば。2本: mcts と value を ply 軸で重ねる)
  - 相関(Pearson) と 平均絶対差

使うNNは 2層ベースライン value (models/pretrained_value.pt が既定)。

使い方:
  uv run python scripts/diag_mcts_vs_value_scores.py \
      --value models/pretrained_value.pt --policy models/pretrained.pt \
      --sims 200 --use-tss --seed 0
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
from renju_transformer.rules import infer_player, winner_after_move
from renju_transformer.utils import select_device

N = 225
SIDE = 15
BLACK, WHITE = 1, 2


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
    ap.add_argument("--value", default="models/pretrained_value.pt", help="2層ベースライン value")
    ap.add_argument("--policy", default="models/pretrained.pt", help="PUCT prior 用")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--use-puct", action="store_true", default=True)
    ap.add_argument("--use-tss", action="store_true", help="root で VCF(必勝/受け)を着手に適用")
    ap.add_argument("--max-vcf-depth", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default="scratchpad/mcts_vs_value_scores.csv")
    ap.add_argument("--out-png", default="scratchpad/mcts_vs_value_scores.png")
    args = ap.parse_args()

    random.seed(args.seed)
    device = select_device(args.device)
    tok = RenjuTokenizer(228, 3)
    lib = load_lib()
    policy = build_model(args.policy, device, with_value=False)
    value = build_model(args.value, device, with_value=True)

    def value_score(board):
        """手番側視点の (tanh value + 1)/2 を [0,1] で返す。"""
        with torch.no_grad():
            ids = tok.encode_input(board).unsqueeze(0).to(device)
            _, v = value(ids, return_value=True)
        return (float(v.item()) + 1.0) * 0.5

    def prior_of(board):
        legal = tok.legal_move_mask(board).to(device)
        ids = tok.encode_input(board).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(ids).squeeze(0).masked_fill(~legal, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs, legal

    def mcts_score_and_move(board):
        """着手の直前に呼ぶ。(mcts_winrate, move, tss_flag) を返す。
        mcts_winrate は TSS の有無に関わらず常に MCTS を回して取得(value と対にするため)。"""
        cur = infer_player(board)
        opp = WHITE if cur == BLACK else BLACK
        arr = (ctypes.c_int * N)(*board)

        probs, legal = prior_of(board)
        visits = (ctypes.c_int * N)()
        probs_arr = (ctypes.c_double * N)(*probs.tolist())
        winrate = lib.run_mcts_c_api_with_policy_and_visits(
            arr, args.sims, random.randint(0, 2**63 - 1), probs_arr, visits, 1 if args.use_puct else 0)
        legal_idx = [i for i, ok in enumerate(legal.tolist()) if ok]
        mcts_move = max(legal_idx, key=lambda i: visits[i])

        # 着手は TSS 優先(あれば)。スコア(winrate)はあくまで MCTS の値を残す。
        move, tss = mcts_move, ""
        if args.use_tss:
            mv = lib.solve_vcf_c_api(arr, cur, args.max_vcf_depth)
            if mv >= 0:
                move, tss = mv, "win"
            else:
                mv = lib.solve_vcf_c_api(arr, opp, args.max_vcf_depth)
                if mv >= 0:
                    move, tss = mv, "block"
        return winrate, move, tss

    rows = []  # (ply, player, mcts, value, move, tss)
    board = [0] * N
    print(f"1戦診断: sims={args.sims} use_tss={args.use_tss} seed={args.seed} device={device}",
          file=sys.stderr)
    print(f"value={args.value}", file=sys.stderr)
    result = "draw"
    for ply in range(1, N + 1):
        player = infer_player(board)
        mcts, move, tss = mcts_score_and_move(board)
        val = value_score(board)
        rows.append((ply, player, mcts, val, move, tss))
        board[move] = player
        w = winner_after_move(board, move, player)
        if w is not None and w != 0:
            result = "black" if w == BLACK else "white"
            break
        if all(c != 0 for c in board):
            break

    # --- 表 ---
    print(f"\n{'ply':>3} {'手番':>4} {'MCTS':>6} {'value':>6} {'差':>6}  {'着手(r,c)':>9} {'TSS':>5}")
    diffs = []
    for ply, player, mcts, val, move, tss in rows:
        diffs.append(abs(mcts - val))
        side = "黒" if player == BLACK else "白"
        r, c = divmod(move, SIDE)
        print(f"{ply:>3} {side:>4} {mcts:6.3f} {val:6.3f} {mcts-val:+6.3f}  ({r:2d},{c:2d}){'':>2} {tss:>5}")

    # --- 統計 ---
    mae = sum(diffs) / max(len(diffs), 1)
    mc = [r[2] for r in rows]
    vl = [r[3] for r in rows]
    n = len(rows)
    if n >= 2:
        mm = sum(mc) / n
        mv = sum(vl) / n
        cov = sum((a - mm) * (b - mv) for a, b in zip(mc, vl))
        sm = sum((a - mm) ** 2 for a in mc) ** 0.5
        sv = sum((b - mv) ** 2 for b in vl) ** 0.5
        pear = cov / (sm * sv) if sm > 0 and sv > 0 else float("nan")
    else:
        pear = float("nan")
    print(f"\n手数={n}  決着={result}  平均絶対差(MCTS-value)={mae:.3f}  Pearson相関={pear:.3f}")

    # --- CSV ---
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ply,player,mcts_winrate,value_score,move,row,col,tss\n")
        for ply, player, mcts, val, move, tss in rows:
            r, c = divmod(move, SIDE)
            f.write(f"{ply},{player},{mcts:.4f},{val:.4f},{move},{r},{c},{tss}\n")
    print(f"CSV -> {out_csv}", file=sys.stderr)

    # --- PNG (matplotlib があれば) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, mc, "-o", ms=3, label="MCTS root win-rate", color="#1f77b4")
        ax.plot(xs, vl, "-s", ms=3, label="value net (v+1)/2", color="#d62728")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("ply (着手番号)")
        ax.set_ylabel("手番側の勝ちやすさ [0,1]")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"MCTS vs value-net score (1 game, sims={args.sims}, "
                     f"Pearson={pear:.2f}, MAE={mae:.2f})")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        out_png = PROJECT_ROOT / args.out_png
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        print(f"PNG -> {out_png}", file=sys.stderr)
    except ImportError:
        print("matplotlib 無し → PNG はスキップ(CSV は出力済み)", file=sys.stderr)


if __name__ == "__main__":
    main()
