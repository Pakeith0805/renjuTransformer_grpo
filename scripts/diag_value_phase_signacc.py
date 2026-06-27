#!/usr/bin/env python
"""
診断: value-net の sign_acc を「フェーズ(手数)別」に集計する
============================================================
N 戦を rollout-MCTS(+TSS, 開幕ランダム K 手) で自己対戦させ、各局面で
  - value net のスコア (手番側視点 [0,1])
  - (任意) MCTS の root 勝率 (手番側視点 [0,1])
を記録。終局後に勝者を確定し、各局面について

    予測正解 = (score > 0.5) == (手番 == 最終勝者)

を判定して、**手数ビン別**に sign_acc を集計する。狙いは「80% という value の天井が
どのフェーズに居るか」を切り分けること:

  - 終盤(残り手数小)が既に ~95% → 80%は **序盤コインフリップ床** が占有 = CNN 余地小
  - 終盤・中盤も伸びない          → **CNN 余地あり**(静的評価が中盤の鋭い局面で外す)

2 軸で集計する:
  (1) 開始からの手数ビン   … 序盤の床を見る
  (2) 終局までの残り手数ビン … 終盤の天井を見る(ゲーム長のばらつきを吸収)

引分は除外。MCTS を同じ枠で並べるので value↔MCTS のどちらが各フェーズで効くかも判る。

使い方:
  uv run python scripts/diag_value_phase_signacc.py \
      --value models/pretrained_value.pt --policy models/pretrained.pt \
      --games 50 --sims 200 --use-tss --random-open 4
  高速(valueのみ, MCTS無し): --no-mcts  (着手は policy prior 貪欲 + TSS)
"""
import argparse
import ctypes
import sys
import random
from collections import defaultdict
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


def bin_label(edges, x):
    """edges=[1,6,11,...] に対し x が入る区間ラベル文字列を返す。"""
    for i in range(len(edges) - 1):
        if edges[i] <= x < edges[i + 1]:
            hi = edges[i + 1] - 1
            return f"{edges[i]:>2}-{hi:<2}"
    return f"{edges[-1]}+  "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="models/pretrained_value.pt", help="2層ベースライン value")
    ap.add_argument("--policy", default="models/pretrained.pt", help="PUCT prior 用")
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--use-puct", action="store_true", default=True)
    ap.add_argument("--use-tss", action="store_true", help="着手に VCF(必勝/受け)を適用")
    ap.add_argument("--max-vcf-depth", type=int, default=12)
    ap.add_argument("--random-open", type=int, default=4, help="開幕にランダム合法手を何手置くか(局面多様化)")
    ap.add_argument("--no-mcts", action="store_true", help="MCTS を回さず policy 貪欲で打つ(高速, valueのみ計測)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default="scratchpad/value_phase_signacc_positions.csv")
    ap.add_argument("--out-png", default="scratchpad/value_phase_signacc.png")
    args = ap.parse_args()

    random.seed(args.seed)
    device = select_device(args.device)
    tok = RenjuTokenizer(228, 3)
    lib = load_lib()
    policy = build_model(args.policy, device, with_value=False)
    value = build_model(args.value, device, with_value=True)
    use_mcts = not args.no_mcts

    def value_score(board):
        with torch.no_grad():
            ids = tok.encode_input(board).unsqueeze(0).to(device)
            _, v = value(ids, return_value=True)
        return (float(v.item()) + 1.0) * 0.5

    def prior_of(board):
        legal = tok.legal_move_mask(board).to(device)
        ids = tok.encode_input(board).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy(ids).squeeze(0).masked_fill(~legal, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
        return probs.cpu().numpy(), legal

    def step(board):
        """(mcts_score or None, move) を返す。着手は TSS優先 → (MCTS最多訪問 or policy貪欲)。"""
        cur = infer_player(board)
        opp = WHITE if cur == BLACK else BLACK
        arr = (ctypes.c_int * N)(*board)
        probs, legal = prior_of(board)
        legal_idx = [i for i, ok in enumerate(legal.tolist()) if ok]

        mcts = None
        base_move = max(legal_idx, key=lambda i: probs[i])  # policy 貪欲 (no-mcts 用)
        if use_mcts:
            visits = (ctypes.c_int * N)()
            probs_arr = (ctypes.c_double * N)(*probs.tolist())
            mcts = lib.run_mcts_c_api_with_policy_and_visits(
                arr, args.sims, random.randint(0, 2**63 - 1), probs_arr, visits, 1 if args.use_puct else 0)
            base_move = max(legal_idx, key=lambda i: visits[i])

        move = base_move
        if args.use_tss:
            mv = lib.solve_vcf_c_api(arr, cur, args.max_vcf_depth)
            if mv >= 0:
                move = mv
            else:
                mv = lib.solve_vcf_c_api(arr, opp, args.max_vcf_depth)
                if mv >= 0:
                    move = mv
        return mcts, move

    def random_open(board, k):
        for _ in range(k):
            legal = tok.legal_move_mask(board)
            cand = [i for i, ok in enumerate(legal.tolist()) if ok]
            if not cand:
                return None
            player = infer_player(board)
            mv = random.choice(cand)
            board[mv] = player
            w = winner_after_move(board, mv, player)
            if w is not None and w != 0:
                return None  # 開幕で決着 → この局はやり直し
        return board

    # positions: (ply, plies_to_end, player, value_score, mcts_score, winner) を貯める
    positions = []
    games_used = draws = 0
    print(f"phase sign_acc: games={args.games} sims={args.sims} use_tss={args.use_tss} "
          f"mcts={use_mcts} open={args.random_open} device={device}", file=sys.stderr)

    g = 0
    attempts = 0
    while games_used < args.games and attempts < args.games * 5:
        attempts += 1
        board = [0] * N
        if args.random_open > 0 and random_open(board, args.random_open) is None:
            continue
        rec = []  # この局の (ply, player, value, mcts)
        winner = None
        for ply in range(1, N + 1):
            player = infer_player(board)
            mcts, move = step(board)
            val = value_score(board)
            rec.append((ply, player, val, mcts))
            board[move] = player
            w = winner_after_move(board, move, player)
            if w is not None and w != 0:
                winner = w
                break
            if all(c != 0 for c in board):
                break
        if winner is None:
            draws += 1
            continue
        games_used += 1
        g += 1
        total = rec[-1][0]
        for (ply, player, val, mcts) in rec:
            positions.append((ply, total - ply, player, val, mcts, winner))
        if games_used % 10 == 0:
            print(f"  {games_used}/{args.games} 局 (draw={draws})  累計局面={len(positions)}", file=sys.stderr)

    if not positions:
        print("有効局面が集まりませんでした。", file=sys.stderr)
        return

    # --- 集計ヘルパ ---
    def acc_by(keyfn, edges):
        """ビン -> (n, value_acc, mcts_acc, value_conf_winner, mcts_conf_winner)"""
        agg = defaultdict(lambda: [0, 0, 0, 0.0, 0.0, 0])  # n, v_ok, m_ok, v_conf, m_conf, m_n
        for (ply, t2e, player, val, mcts, winner) in positions:
            lab = bin_label(edges, keyfn(ply, t2e))
            a = agg[lab]
            a[0] += 1
            v_pred_win = val > 0.5
            v_true = (player == winner)
            a[1] += int(v_pred_win == v_true)
            # winner 視点の確信度 (手番が勝者なら val、そうでなければ 1-val)
            a[3] += val if player == winner else (1.0 - val)
            if mcts is not None:
                a[2] += int((mcts > 0.5) == v_true)
                a[4] += mcts if player == winner else (1.0 - mcts)
                a[5] += 1
        return agg

    def print_table(title, agg, edges):
        labels = [bin_label(edges, edges[i]) for i in range(len(edges) - 1)] + [f"{edges[-1]}+  "]
        print(f"\n=== {title} ===")
        print(f"{'bin':>8} {'n':>6} {'value_acc':>10} {'mcts_acc':>10} {'v_conf':>8} {'m_conf':>8}")
        for lab in labels:
            if lab not in agg:
                continue
            n, vok, mok, vconf, mconf, mn = agg[lab]
            macc = f"{mok/mn*100:6.1f}%" if mn else "    -  "
            mcf = f"{mconf/mn:6.3f}" if mn else "   -  "
            print(f"{lab:>8} {n:>6} {vok/n*100:8.1f}%  {macc:>10} {vconf/n:8.3f} {mcf:>8}")

    # overall
    vok = sum(int((p[3] > 0.5) == (p[2] == p[5])) for p in positions)
    tot = len(positions)
    mvalid = [p for p in positions if p[4] is not None]
    mok = sum(int((p[4] > 0.5) == (p[2] == p[5])) for p in mvalid)
    print(f"\n使用局={games_used} 引分除外={draws} 総局面={tot}")
    print(f"overall value sign_acc = {vok/tot*100:.1f}%  "
          + (f"(MCTS {mok/len(mvalid)*100:.1f}%)" if mvalid else ""))

    start_edges = [1, 6, 11, 16, 21, 26, 31]
    end_edges = [0, 1, 2, 3, 5, 8, 12]  # 残り手数: 0(最終手),1,2,3-4,5-7,8-11,12+
    agg_start = acc_by(lambda ply, t2e: ply, start_edges)
    agg_end = acc_by(lambda ply, t2e: t2e, end_edges)
    print_table("開始からの手数ビン (序盤の床)", agg_start, start_edges)
    print_table("終局までの残り手数ビン (終盤の天井)", agg_end, end_edges)
    print("\n読み方: 残り手数小ビンが ~95% なら 80%は序盤床が占有=CNN余地小 / "
          "中盤・終盤も伸びないなら CNN余地あり")

    # --- 局面 CSV ---
    out_csv = PROJECT_ROOT / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("ply,plies_to_end,player,value,mcts,winner\n")
        for (ply, t2e, player, val, mcts, winner) in positions:
            f.write(f"{ply},{t2e},{player},{val:.4f},{'' if mcts is None else f'{mcts:.4f}'},{winner}\n")
    print(f"CSV -> {out_csv}", file=sys.stderr)

    # --- PNG: 2パネル(手数別 / 残り手数別) の sign_acc ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def series(agg, edges):
            labels = [bin_label(edges, edges[i]) for i in range(len(edges) - 1)] + [f"{edges[-1]}+  "]
            xs, vacc, macc = [], [], []
            for lab in labels:
                if lab not in agg:
                    continue
                n, vok, mok, vconf, mconf, mn = agg[lab]
                xs.append(lab.strip())
                vacc.append(vok / n * 100)
                macc.append(mok / mn * 100 if mn else float("nan"))
            return xs, vacc, macc

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
        for ax, (agg, edges, title, xlab) in zip(axes, [
            (agg_start, start_edges, "開始からの手数 (序盤の床)", "ply from start"),
            (agg_end, end_edges, "終局までの残り手数 (終盤の天井)", "plies to end"),
        ]):
            xs, vacc, macc = series(agg, edges)
            ax.plot(xs, vacc, "-s", color="#d62728", label="value sign_acc")
            if any(v == v for v in macc):  # NaN でない値がある
                ax.plot(xs, macc, "-o", color="#1f77b4", label="MCTS sign_acc")
            ax.axhline(80, color="gray", lw=0.8, ls="--", label="80% 天井")
            ax.axhline(50, color="lightgray", lw=0.8, ls=":")
            ax.set_ylim(40, 102)
            ax.set_title(title)
            ax.set_xlabel(xlab)
            ax.set_ylabel("sign_acc (%)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"value-net phase sign_acc  (games={games_used}, sims={args.sims})")
        fig.tight_layout()
        out_png = PROJECT_ROOT / args.out_png
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=120)
        print(f"PNG -> {out_png}", file=sys.stderr)
    except ImportError:
        print("matplotlib 無し → PNG はスキップ", file=sys.stderr)


if __name__ == "__main__":
    main()
