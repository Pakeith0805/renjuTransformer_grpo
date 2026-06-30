#!/usr/bin/env python
"""
固定盤面 (fixed_positions.json) に対する TSS 模倣率を計測する。

extract_fixed_positions.py で作った JSON をそのまま読み、
指定したモデルチェックポイントが各カテゴリのオラクル手を
どれだけ再現できるかを採点する。

使い方:
  uv run python scripts/eval_fixed_positions.py \
      --models pretrained=models/pretrained.pt \
               ckpt80=artifacts/grpo_checkpoint_80.pt \
      --json scripts/fixed_positions.json

オプション:
  --show N   各カテゴリ先頭 N 盤面を ASCII で表示
  --out CSV  サンプル毎の結果を CSV に書き出す
"""
from __future__ import annotations
import sys, json, argparse, csv as csv_mod
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.utils import select_device

N = 15
EMPTY, BLACK, WHITE = 0, 1, 2

ATTACK_CATS = {"own_four", "own_three_win"}
BLOCK_CATS  = {"block_four", "block_three_win"}
CAT_ORDER   = ["own_four", "block_four", "own_three_win", "block_three_win"]


# ---- モデル ----------------------------------------------------------------

def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    mc = ckpt["config"]["model"]
    model = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"],
        d_model=mc["d_model"], nhead=mc["nhead"], num_layers=mc["num_layers"],
        dim_feedforward=mc["dim_feedforward"], dropout=mc["dropout"],
        activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model


def predict(model, tokenizer, board, device):
    """masked-argmax の手と、各マスの確率 (softmax) を返す。"""
    legal = tokenizer.legal_move_mask(board).to(device)
    input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_ids).squeeze(0)
        masked = logits.masked_fill(~legal, float("-inf"))
        probs  = torch.softmax(masked, dim=-1)
        pred   = int(masked.argmax().item())
    return pred, probs


# ---- 採点 ------------------------------------------------------------------

def score_position(pos, pred, probs):
    """(correct: int, prob_mass: float) を返す。
    attack 系: oracle_move と単一一致。
    block  系: correct_set のどれかなら正解 + prob_mass は集合全体の合計。"""
    cat = pos["cat"]
    if cat in ATTACK_CATS:
        ans = pos["oracle_move"]
        ok  = int(pred == ans)
        pm  = float(probs[ans].item())
    else:
        cs  = set(pos.get("correct_set", [pos["oracle_move"]]))
        ok  = int(pred in cs)
        pm  = float(sum(probs[m].item() for m in cs))
    return ok, pm


# ---- 表示 ------------------------------------------------------------------

def print_board(board, highlight=None):
    """highlight: {idx: label} で特定マスに記号を上書き表示。"""
    sym = {EMPTY: " .", BLACK: " ●", WHITE: " ○"}
    hl  = highlight or {}
    print("    " + "".join(f"{c:2d}" for c in range(N)))
    for r in range(N):
        row_str = ""
        for c in range(N):
            i = r * N + c
            if i in hl:
                row_str += f" {hl[i]}"
            else:
                row_str += sym[board[i]]
        print(f"{r:2d} {row_str}")


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="label=path の並び。例: pretrained=models/pretrained.pt")
    ap.add_argument("--json", default="scripts/fixed_positions.json",
                    help="extract_fixed_positions.py で作った JSON ファイル")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--show", type=int, default=0,
                    help="各カテゴリ先頭 N 盤面を ASCII 表示")
    ap.add_argument("--out", default=None, help="サンプル毎の結果を書く CSV")
    args = ap.parse_args()

    device    = select_device(args.device)
    tokenizer = RenjuTokenizer(sep_token_id=228, move_id_offset=3)

    # JSON 読み込み
    json_path = PROJECT_ROOT / args.json
    with open(json_path, encoding="utf-8") as f:
        positions = json.load(f)
    print(f"loaded {len(positions)} positions from {json_path}")

    cats_in_json = []
    for p in positions:
        if p["cat"] not in cats_in_json:
            cats_in_json.append(p["cat"])
    cats = [c for c in CAT_ORDER if c in cats_in_json]  # 順序を固定

    # カテゴリ別件数を表示
    from collections import Counter
    cnt = Counter(p["cat"] for p in positions)
    for c in cats:
        print(f"  {c:20s}: {cnt[c]} 件")

    # モデル読み込み
    models = {}
    for spec in args.models:
        label, path = spec.split("=", 1)
        models[label] = load_model(path, device)
        print(f"loaded model [{label}]: {path}")

    # 採点
    # agg[label][cat] = [n, hits, prob_sum]
    agg  = {lab: {cat: [0, 0, 0.0] for cat in cats} for lab in models}
    rows = []
    shown = {cat: 0 for cat in cats}

    for pos in positions:
        cat   = pos["cat"]
        board = pos["board"]
        omv   = pos["oracle_move"]

        if shown[cat] < args.show:
            kind_label = "attack" if cat in ATTACK_CATS else "block"
            hl = {omv: "★"}
            if cat in BLOCK_CATS:
                for m in pos.get("correct_set", []):
                    hl[m] = "◆"
            print(f"\n--- {cat}  手番={'黒' if pos['to_move']=='black' else '白'}  {pos['stones']}石"
                  f"  oracle=({omv//N},{omv%N}) ---")
            print_board(board, hl)
            shown[cat] += 1

        for lab, model in models.items():
            pred, probs = predict(model, tokenizer, board, device)
            ok, pm = score_position(pos, pred, probs)
            a = agg[lab][cat]
            a[0] += 1
            a[1] += ok
            a[2] += pm
            rows.append((lab, cat, pos["id"], omv, pred, ok, round(pm, 4)))

    # レポート
    print("\n" + "=" * 64)
    print(f" TSS 模倣率  (固定 {len(positions)} 局面 / {json_path.name})")
    print("=" * 64)
    print(f"  ※ attack 系 (own_four / own_three_win): oracle 手と単一一致")
    print(f"  ※ block  系 (block_four / block_three_win): 正解手集合メンバーシップ")
    for lab in models:
        print(f"\n# {lab}")
        print(f"  {'category':22s} {'n':>4} {'top1':>7} {'mean_p':>8}")
        print("  " + "-" * 44)
        tot_n = tot_h = 0
        for cat in cats:
            n, h, ps = agg[lab][cat]
            if n == 0:
                continue
            tot_n += n
            tot_h += h
            print(f"  {cat:22s} {n:>4} {h/n*100:6.1f}% {ps/n:8.3f}")
        if tot_n:
            print("  " + "-" * 44)
            print(f"  {'OVERALL':22s} {tot_n:>4} {tot_h/tot_n*100:6.1f}%")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            wr = csv_mod.writer(f)
            wr.writerow(["model", "cat", "pos_id", "oracle_move", "pred", "correct", "prob_mass"])
            wr.writerows(rows)
        print(f"\nper-sample CSV -> {args.out}")


if __name__ == "__main__":
    main()
