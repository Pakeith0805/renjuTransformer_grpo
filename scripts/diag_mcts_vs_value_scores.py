#!/usr/bin/env python
"""
診断: 1戦を通して MCTS / MCTS+TSS / value-net スコアを並べて見る
================================================================
1局を rollout-leaf MCTS(PUCT prior + 任意 TSS) で実際に打たせ、各手番で「着手の直前」に
同じ盤面へ 3 つの評価を **対で** 記録する。どれも「手番側がどれくらい勝ちそうか」を表す [0,1]。

  - MCTS      = run_mcts の返り値 root.wins/root.visits        (TSS を反映しない生の rollout 勝率)
  - MCTS+TSS  = 自分に VCF 必勝があれば 1.0 に上書き、無ければ MCTS と同じ
                (相手 VCF は "block" 注記のみ。1手受けで防げるので敗北断定はしない=誇張しない)
  - value net = (tanh value + 1)/2                             (実戦勝敗で学習=TSS 的必勝を暗黙に反映)

狙い: NN(value) が「生 MCTS 寄り」か「TSS 込み MCTS 寄り」かを可視化する。
NN は VCF 局面の勝敗も学習しているので、必勝直前で 1.0 に跳ねるなら TSS 寄り、鈍いなら生 MCTS 寄り。

出力:
  - 標準出力に表 (ply / 手番 / mcts / mcts+tss / value / 着手)
  - CSV  (--out-csv)
  - PNG 折れ線 (--out-png, matplotlib があれば。3本を ply 軸で重ねる)
  - value と各 MCTS の 相関(Pearson) / 平均絶対差

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
        """着手の直前に呼ぶ。(mcts, mcts_tss, move, tss_flag) を返す。
        mcts は TSS の有無に関わらず常に MCTS を回して取得(value と対にするため)。
        mcts_tss は自分の VCF 必勝が見つかれば 1.0 に上書きした版。"""
        cur = infer_player(board)
        opp = WHITE if cur == BLACK else BLACK
        arr = (ctypes.c_int * N)(*board)

        # --- 生 MCTS (TSS 無視) ---
        probs, legal = prior_of(board)
        visits = (ctypes.c_int * N)()
        probs_arr = (ctypes.c_double * N)(*probs.tolist())
        mcts = lib.run_mcts_c_api_with_policy_and_visits(
            arr, args.sims, random.randint(0, 2**63 - 1), probs_arr, visits, 1 if args.use_puct else 0)
        legal_idx = [i for i, ok in enumerate(legal.tolist()) if ok]
        mcts_move = max(legal_idx, key=lambda i: visits[i])

        # --- TSS: 自分の必勝 / 相手の脅威。着手選択にもスコア上書きにも使う ---
        own_win = lib.solve_vcf_c_api(arr, cur, args.max_vcf_depth)        # >=0 で自分に VCF 必勝
        opp_threat = -1
        if own_win < 0:
            opp_threat = lib.solve_vcf_c_api(arr, opp, args.max_vcf_depth)  # >=0 で相手 VCF を要受け

        # MCTS+TSS スコア(列): 自分に VCF 必勝 → 1.0、無ければ MCTS のまま。常に算出。
        mcts_tss = 1.0 if own_win >= 0 else mcts

        # 着手: --use-tss のときのみ TSS で打つ(自分の必勝 > 相手脅威の受け > MCTS 最多訪問)
        if args.use_tss and own_win >= 0:
            move, tss = own_win, "win"
        elif args.use_tss and opp_threat >= 0:
            move, tss = opp_threat, "block"
        else:
            move, tss = mcts_move, ("win" if own_win >= 0 else ("block" if opp_threat >= 0 else ""))
        return mcts, mcts_tss, move, tss

    rows = []  # (ply, player, mcts, mcts_tss, value, move, tss)
    board = [0] * N
    print(f"1戦診断: sims={args.sims} use_tss={args.use_tss} seed={args.seed} device={device}",
          file=sys.stderr)
    print(f"value={args.value}", file=sys.stderr)
    result = "draw"
    for ply in range(1, N + 1):
        player = infer_player(board)
        mcts, mcts_tss, move, tss = mcts_score_and_move(board)
        val = value_score(board)
        rows.append((ply, player, mcts, mcts_tss, val, move, tss))
        board[move] = player
        w = winner_after_move(board, move, player)
        if w is not None and w != 0:
            result = "black" if w == BLACK else "white"
            break
        if all(c != 0 for c in board):
            break

    # --- 表 ---
    print(f"\n{'ply':>3} {'手番':>4} {'MCTS':>7} {'MCTS+TSS':>9} {'value':>7}  {'着手(r,c)':>9} {'TSS':>5}")
    for ply, player, mcts, mcts_tss, val, move, tss in rows:
        side = "黒" if player == BLACK else "白"
        r, c = divmod(move, SIDE)
        print(f"{ply:>3} {side:>4} {mcts:7.3f} {mcts_tss:9.3f} {val:7.3f}  ({r:2d},{c:2d}){'':>2} {tss:>5}")

    # --- 統計: value と各 MCTS の一致度 ---
    mc = [r[2] for r in rows]
    mct = [r[3] for r in rows]
    vl = [r[4] for r in rows]
    n = len(rows)

    def stats(a, b):
        mae = sum(abs(x - y) for x, y in zip(a, b)) / max(len(a), 1)
        if len(a) < 2:
            return mae, float("nan")
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        cov = sum((x - ma) * (y - mb) for x, y in zip(a, b))
        sa = sum((x - ma) ** 2 for x in a) ** 0.5
        sb = sum((y - mb) ** 2 for y in b) ** 0.5
        return mae, (cov / (sa * sb) if sa > 0 and sb > 0 else float("nan"))

    mae_v_m, p_v_m = stats(vl, mc)
    mae_v_t, p_v_t = stats(vl, mct)
    print(f"\n手数={n}  決着={result}")
    print(f"value vs MCTS     : MAE={mae_v_m:.3f}  Pearson={p_v_m:.3f}")
    print(f"value vs MCTS+TSS : MAE={mae_v_t:.3f}  Pearson={p_v_t:.3f}  "
          f"(NN が TSS 寄りなら此方が一致)")

    # --- CSV ---
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ply,player,mcts,mcts_tss,value,move,row,col,tss\n")
        for ply, player, mcts, mcts_tss, val, move, tss in rows:
            r, c = divmod(move, SIDE)
            f.write(f"{ply},{player},{mcts:.4f},{mcts_tss:.4f},{val:.4f},{move},{r},{c},{tss}\n")
    print(f"CSV -> {out_csv}", file=sys.stderr)

    # --- PNG (matplotlib があれば) ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [r[0] for r in rows]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(xs, mc, "-o", ms=3, label="MCTS (rollout, no TSS)", color="#1f77b4")
        ax.plot(xs, mct, "-^", ms=3, label="MCTS+TSS (own VCF -> 1.0)", color="#2ca02c")
        ax.plot(xs, vl, "-s", ms=3, label="value net (v+1)/2", color="#d62728")
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        ax.set_xlabel("ply (着手番号)")
        ax.set_ylabel("手番側の勝ちやすさ [0,1]")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"MCTS / MCTS+TSS / value-net (1 game, sims={args.sims}, "
                     f"r(v,mcts)={p_v_m:.2f}, r(v,tss)={p_v_t:.2f})")
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
