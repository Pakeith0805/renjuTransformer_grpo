#!/usr/bin/env python
"""
TSS(VCF) 模倣テスト
===================
policy が「探索なしの一発推論」でどこまで TSS らしい手を選べるかを、VCF ソルバーを
正解オラクルにして自動採点する。

考え方:
  - 型(テンプレ)で戦術モチーフを作る (四・四三・受け・優先順位)。
  - 位置/向き/色をランダムに振り、パリティを安全なフィラー石で合わせる。
  - 盤面を `solve_vcf_c_api` に通し、**オラクルが意図通りの答えを返す盤面だけ採用**
    (= 自己検算。ソルバーの細かい挙動を予測しなくても、正解が確定した盤面だけ残る)。
  - 各モデルの masked-argmax がオラクル手と一致するか、正解手への確率質量はどれだけか、を集計。

このコードベースの TSS は VCF(四の連続強制) のみ。活三だけの攻防(②③)は射程外なので
ここでは扱わない(v2 で VCT/手ラベルを追加予定)。

使い方:
  uv run python scripts/test_tss_imitation.py \
      --models pretrained=artifacts/checkpoints/pretrained.pt \
               g32_80=artifacts/exp_minimax_topk_g32/grpo_checkpoint_80.pt \
      --per-category 200 --seed 0
  オプション: --depth 12  --device cpu  --out results.csv  --show 3
"""
from __future__ import annotations
import sys
import ctypes
import argparse
import random
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.rules import infer_player, legal_move_mask, winner_after_move  # noqa: F401
from renju_transformer.utils import select_device, set_seed

N = 15
EMPTY, BLACK, WHITE = 0, 1, 2
PERP = {(0, 1): (1, 0), (1, 0): (0, 1), (1, 1): (1, -1), (1, -1): (1, 1)}
DIRS = list(PERP.keys())


# --------------------------------------------------------------------------- #
# C ライブラリ (VCF ソルバー = オラクル)
# --------------------------------------------------------------------------- #
_lib = None


def get_lib():
    global _lib
    if _lib is None:
        name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
        path = PROJECT_ROOT / name
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}. ビルドしてください。")
        _lib = ctypes.CDLL(str(path))
        _lib.solve_vcf_c_api.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        _lib.solve_vcf_c_api.restype = ctypes.c_int
    return _lib


def vcf(board, player, depth):
    """player の VCF 勝ち初手 index (>=0) / 無ければ -1。"""
    arr = (ctypes.c_int * 225)(*board)
    return get_lib().solve_vcf_c_api(arr, player, depth)


def oracle(board, depth):
    """コードベースの TSS と同じ判断: 自分VCF→attack / 相手VCF→block / どちらも無→(None)。
    返り値: ("attack"|"block"|None, move_index)。"""
    me = infer_player(board)
    opp = WHITE if me == BLACK else BLACK
    my = vcf(board, me, depth)
    if my >= 0:
        return "attack", my
    op = vcf(board, opp, depth)
    if op >= 0:
        return "block", op
    return None, -1


# --------------------------------------------------------------------------- #
# 盤面ヘルパ
# --------------------------------------------------------------------------- #
def idx(r, c):
    return r * N + c


def inb(r, c):
    return 0 <= r < N and 0 <= c < N


def counts(board):
    b = sum(1 for x in board if x == BLACK)
    w = sum(1 for x in board if x == WHITE)
    return b, w


def line(r, c, d, k):
    return [(r + i * d[0], c + i * d[1]) for i in range(k)]


def place(board, cells, color):
    """cells を color で置く。範囲外/非空なら False。"""
    for (r, c) in cells:
        if not inb(r, c) or board[idx(r, c)] != EMPTY:
            return False
    for (r, c) in cells:
        board[idx(r, c)] = color
    return True


def isolated_cell(board, rng):
    """周囲5x5が全て空なマス(他の石と必ず2マス以上離れる→新たな連を作らない)。"""
    cands = []
    for i in range(225):
        if board[i] != EMPTY:
            continue
        r, c = divmod(i, N)
        if all(
            board[idx(rr, cc)] == EMPTY
            for rr in range(max(0, r - 2), min(N, r + 3))
            for cc in range(max(0, c - 2), min(N, c + 3))
        ):
            cands.append(i)
    return rng.choice(cands) if cands else None


def balance_parity(board, to_move, rng):
    """to_move が手番になるよう、孤立フィラー石で石数を調整 (BLACK→b==w / WHITE→b==w+1)。"""
    b, w = counts(board)
    need_black = (w + (1 if to_move == WHITE else 0)) - b  # 追加すべき黒の数 (負なら白を追加)
    add_color, n_add = (BLACK, need_black) if need_black > 0 else (WHITE, -need_black)
    for _ in range(n_add):
        cell = isolated_cell(board, rng)
        if cell is None:
            return False
        board[cell] = add_color
    b, w = counts(board)
    return (to_move == BLACK and b == w) or (to_move == WHITE and b == w + 1)


# --------------------------------------------------------------------------- #
# モチーフ生成 (空盤に置く → 後でパリティ調整)
# --------------------------------------------------------------------------- #
def gen_own_four(rng):
    """① 自分の四(片端開き)。手番=color が p4 で五を完成。"""
    color = rng.choice([BLACK, WHITE])
    opp = WHITE if color == BLACK else BLACK
    r, c = rng.randrange(N), rng.randrange(N)
    d = rng.choice(DIRS)
    four = line(r, c, d, 4)
    back = (r - d[0], c - d[1])
    openend = (r + 4 * d[0], c + 4 * d[1])
    if not (inb(*back) and inb(*openend)):
        return None
    board = [EMPTY] * 225
    if not place(board, four, color) or not place(board, [back], opp):
        return None
    return dict(board=board, to_move=color, want="attack", distractor=set())


def gen_block_four(rng):
    """④ 相手の四(片端開き)。手番=color は p4 で受けるしかない。"""
    color = rng.choice([BLACK, WHITE])
    opp = WHITE if color == BLACK else BLACK
    r, c = rng.randrange(N), rng.randrange(N)
    d = rng.choice(DIRS)
    four = line(r, c, d, 4)
    back = (r - d[0], c - d[1])
    openend = (r + 4 * d[0], c + 4 * d[1])
    if not (inb(*back) and inb(*openend)):
        return None
    board = [EMPTY] * 225
    if not place(board, four, opp) or not place(board, [back], color):
        return None
    return dict(board=board, to_move=color, want="block", distractor=set())


def gen_four_three(rng):
    """⑤ 四三フォーク。f を打つと四(d方向)と活三(e方向)を同時生成 = 強制勝ち。"""
    color = rng.choice([BLACK, WHITE])
    r, c = rng.randrange(N), rng.randrange(N)
    d = rng.choice(DIRS)
    e = PERP[d]
    three_d = line(r, c, d, 3)                 # d方向の三
    f = (r + 3 * d[0], c + 3 * d[1])           # フォーク手(これで四)
    perp_two = [(f[0] - 2 * e[0], f[1] - 2 * e[1]), (f[0] - e[0], f[1] - e[1])]  # e方向の二
    back_d = (r - d[0], c - d[1])              # 四の後端(開き)
    ends_e = [(f[0] - 3 * e[0], f[1] - 3 * e[1]), (f[0] + e[0], f[1] + e[1])]   # 三の両端(開き)
    cells = three_d + perp_two
    need_empty = [f, back_d] + ends_e
    if not all(inb(*p) for p in cells + need_empty):
        return None
    board = [EMPTY] * 225
    if not place(board, cells, color):
        return None
    if any(board[idx(*p)] != EMPTY for p in need_empty):
        return None
    return dict(board=board, to_move=color, want="attack", distractor=set())


def gen_priority(rng):
    """⑥ 優先順位。相手の四(受け必須) + 自分の活三(誘惑)。正解は四を受ける手。"""
    base = gen_block_four(rng)
    if base is None:
        return None
    board = base["board"]
    color = base["to_move"]
    # 自分の活三を追加 (両端開き)。正しさ(オラクルが block を返すか)は make_case が再検算するので、
    # ここでは「三石+両端+その1マス周囲が空」程度の軽い隔離で十分(偶発的な連を避ける用)。
    for _ in range(60):
        r, c = rng.randrange(N), rng.randrange(N)
        d = rng.choice(DIRS)
        three = line(r, c, d, 3)
        e0 = (r - d[0], c - d[1])
        e1 = (r + 3 * d[0], c + 3 * d[1])
        if not all(inb(*p) for p in three + [e0, e1]):
            continue
        ring = set()
        for (rr, cc) in three + [e0, e1]:
            for ar in range(rr - 1, rr + 2):
                for ac in range(cc - 1, cc + 2):
                    if inb(ar, ac):
                        ring.add(idx(ar, ac))
        if any(board[i] != EMPTY for i in ring):
            continue
        if place(board, three, color):
            base["distractor"] = {idx(*e0), idx(*e1)}
            return base
    return None


GENERATORS = {
    "own_four": gen_own_four,
    "block_four": gen_block_four,
    "four_three": gen_four_three,
    "priority": gen_priority,
}


def make_case(name, rng, depth, tries=200):
    """型生成→パリティ調整→オラクル検算。意図通りに正解が確定した盤面のみ返す。"""
    gen = GENERATORS[name]
    for _ in range(tries):
        case = gen(rng)
        if case is None:
            continue
        board = case["board"]
        if not balance_parity(board, case["to_move"], rng):
            continue
        try:
            kind, mv = oracle(board, depth)
        except ValueError:
            continue  # パリティ不正など
        if kind != case["want"] or mv < 0:
            continue  # オラクルが意図と違う→破棄(フィラーが壊した/構成ミス)
        # 正解手は合法か(黒禁手でないか)も担保
        if not legal_move_mask(board)[mv]:
            continue
        case["answer"] = mv
        case["oracle_kind"] = kind
        return case
    return None


# --------------------------------------------------------------------------- #
# ケース収集: template(型) と random(自然局面+オラクル選別)
# --------------------------------------------------------------------------- #
def collect_template_cases(rng, depth, per_category):
    cases = []
    for cat in GENERATORS:
        made, fails = 0, 0
        while made < per_category and fails < per_category * 3 + 50:
            c = make_case(cat, rng, depth)
            if c is None:
                fails += 1
                continue
            cases.append(dict(cat=cat, board=c["board"], answer=c["answer"],
                              distractor=c["distractor"], kind=c["oracle_kind"]))
            made += 1
        if made < per_category:
            print(f"[warn] {cat}: {made}/{per_category} 盤面のみ生成", file=sys.stderr)
    return cases


def gen_random_position(rng, kmin, kmax):
    """空盤からランダムに合法手を K 手打って自然な中盤局面を作る。途中で決着したら破棄。"""
    k = rng.randint(kmin, kmax)
    board = [EMPTY] * 225
    for _ in range(k):
        legal = [i for i, ok in enumerate(legal_move_mask(board)) if ok]
        if not legal:
            return None
        player = infer_player(board)
        mv = rng.choice(legal)
        board[mv] = player
        if winner_after_move(board, mv, player) is not None:
            return None  # 決着済みは捨てる
    return board


def collect_random_cases(rng, depth, per_category, kmin, kmax):
    """自然局面を量産し、オラクルが attack/block と判定したものだけ各 per_category まで採用。"""
    buckets = {"attack": [], "block": []}
    cap = per_category * 600 + 3000
    for _ in range(cap):
        if all(len(buckets[k]) >= per_category for k in buckets):
            break
        board = gen_random_position(rng, kmin, kmax)
        if board is None:
            continue
        try:
            kind, mv = oracle(board, depth)
        except ValueError:
            continue
        if kind is None or mv < 0 or not legal_move_mask(board)[mv]:
            continue
        if len(buckets[kind]) < per_category:
            buckets[kind].append(dict(cat=kind, board=board, answer=mv,
                                      distractor=set(), kind=kind))
    for k in buckets:
        if len(buckets[k]) < per_category:
            print(f"[warn] random {k}: {len(buckets[k])}/{per_category} のみ採用(自然局面では稀)",
                  file=sys.stderr)
    return buckets["attack"] + buckets["block"]


# --------------------------------------------------------------------------- #
# モデル
# --------------------------------------------------------------------------- #
def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    model = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_layers=mc["num_layers"], dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"], activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict(model, tokenizer, board, device):
    """masked-argmax の手と、各マスの確率(softmax)を返す。"""
    legal = tokenizer.legal_move_mask(board).to(device)
    input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).squeeze(0)
        masked = logits.masked_fill(~legal, float("-inf"))
        probs = torch.softmax(masked, dim=-1)
        pred = int(masked.argmax().item())
    return pred, probs


def print_board(board):
    sym = {0: " .", 1: " ●", 2: " ○"}
    print("    " + "".join(f"{c:2d}" for c in range(N)))
    for r in range(N):
        print(f"{r:2d} " + "".join(sym[board[idx(r, c)]] for c in range(N)))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path の並び。例: pretrained=models/pretrained.pt g32=.../ckpt_80.pt")
    ap.add_argument("--per-category", type=int, default=200)
    ap.add_argument("--source", choices=["template", "random"], default="template",
                    help="template=型生成 / random=ランダム対局K手→オラクル選別の自然局面")
    ap.add_argument("--kmin", type=int, default=8, help="random: 進める手数の下限")
    ap.add_argument("--kmax", type=int, default=30, help="random: 進める手数の上限")
    ap.add_argument("--depth", type=int, default=12, help="VCF 探索深さ(オラクル)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="サンプル毎の結果を書く CSV")
    ap.add_argument("--show", type=int, default=0, help="各カテゴリ先頭 N 盤面を表示")
    args = ap.parse_args()

    set_seed(args.seed)
    rng = random.Random(args.seed)
    device = select_device(args.device)
    tokenizer = RenjuTokenizer(sep_token_id=228, move_id_offset=3)

    models = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        models[label] = load_model(path, device)
        print(f"loaded {label}: {path}")

    # ---- ケース収集 (ソース別) ----
    if args.source == "random":
        cases = collect_random_cases(rng, args.depth, args.per_category, args.kmin, args.kmax)
    else:
        cases = collect_template_cases(rng, args.depth, args.per_category)
    print(f"collected {len(cases)} cases (source={args.source})")

    categories = []
    for c in cases:
        if c["cat"] not in categories:
            categories.append(c["cat"])
    # 集計: agg[label][cat] = [n, top1_hits, prob_sum, distractor_hits]
    agg = {lab: {cat: [0, 0, 0.0, 0] for cat in categories} for lab in models}
    shown = {cat: 0 for cat in categories}
    rows = []

    for case in cases:
        cat, board, ans, distr = case["cat"], case["board"], case["answer"], case["distractor"]
        if shown[cat] < args.show:
            print(f"\n--- {cat} (手番={'黒' if infer_player(board)==BLACK else '白'}, "
                  f"オラクル={case['kind']} @ {divmod(ans, N)}) ---")
            print_board(board)
            shown[cat] += 1
        for lab, model in models.items():
            pred, probs = predict(model, tokenizer, board, device)
            a = agg[lab][cat]
            a[0] += 1
            a[1] += int(pred == ans)
            a[2] += float(probs[ans].item())
            a[3] += int(pred in distr)
            rows.append((lab, cat, ans, pred, int(pred == ans),
                         round(float(probs[ans].item()), 4), int(pred in distr)))

    # ---- レポート ----
    print("\n" + "=" * 72)
    print(" TSS(VCF) 模倣テスト結果  (top1=オラクル一致率, p=正解手への平均確率)")
    print("=" * 72)
    for lab in models:
        print(f"\n# {lab}")
        print(f"  {'category':12} {'n':>4} {'top1':>7} {'mean_p':>8}  {'distractor':>10}")
        print("  " + "-" * 46)
        tot_n = tot_h = 0
        for cat in categories:
            n, h, ps, dh = agg[lab][cat]
            if n == 0:
                continue
            tot_n += n
            tot_h += h
            dz = f"{dh / n * 100:9.1f}%" if cat == "priority" else " " * 10
            print(f"  {cat:12} {n:>4} {h / n * 100:6.1f}% {ps / n:8.3f}  {dz}")
        if tot_n:
            print("  " + "-" * 46)
            print(f"  {'OVERALL':12} {tot_n:>4} {tot_h / tot_n * 100:6.1f}%")

    if args.out:
        import csv
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["model", "category", "oracle_move", "pred", "correct", "prob_correct", "is_distractor"])
            wr.writerows(rows)
        print(f"\nper-sample CSV -> {args.out}")


if __name__ == "__main__":
    main()
