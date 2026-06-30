#!/usr/bin/env python
"""
data.csv.gz から「都合の良い盤面」を抽出して JSON に保存し、HTML で可視化する。

都合の良い = 中盤局面 (stones_min〜stones_max) かつ
             VCF/VCT オラクルが attack または block を返す局面。

出力:
  scripts/fixed_positions.json   -- 盤面データ
  scripts/fixed_positions.html   -- 盤面可視化 (ブラウザで開く)
"""
from __future__ import annotations
import sys, ctypes, gzip, csv, json, random, argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.rules import infer_player, legal_move_mask, winner_after_move

N = 15
EMPTY, BLACK, WHITE = 0, 1, 2

# ---- C ライブラリ ----
_lib = None
def get_lib():
    global _lib
    if _lib is None:
        name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
        path = PROJECT_ROOT / name
        _lib = ctypes.CDLL(str(path))
        _lib.solve_vcf_c_api.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        _lib.solve_vcf_c_api.restype = ctypes.c_int
        _lib.solve_vct_c_api.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib.solve_vct_c_api.restype = ctypes.c_int
    return _lib

def solve_vcf(board, player, depth=12):
    arr = (ctypes.c_int * 225)(*board)
    return get_lib().solve_vcf_c_api(arr, player, depth)

def solve_vct(board, player, depth=3):
    arr = (ctypes.c_int * 225)(*board)
    return get_lib().solve_vct_c_api(arr, player, depth, 0)

def oracle(board, use_vct=False, vcf_depth=12, vct_depth=3):
    """(kind, move_idx) を返す。kind: 'attack'|'block'|None"""
    me = infer_player(board)
    opp = WHITE if me == BLACK else BLACK
    fn = (lambda b, p: solve_vct(b, p, vct_depth)) if use_vct else (lambda b, p: solve_vcf(b, p, vcf_depth))
    my = fn(board, me)
    if my >= 0:
        return "attack", my
    op = fn(board, opp)
    if op >= 0:
        return "block", op
    return None, -1

def block_defense_set(board, use_vct=False, vcf_depth=12, vct_depth=3):
    me = infer_player(board)
    opp = WHITE if me == BLACK else BLACK
    fn = (lambda b, p: solve_vct(b, p, vct_depth)) if use_vct else (lambda b, p: solve_vcf(b, p, vcf_depth))
    s = set()
    legal = legal_move_mask(board)
    for mv in range(225):
        if not legal[mv]:
            continue
        nb = list(board)
        nb[mv] = me
        if fn(nb, opp) < 0:
            s.add(mv)
    return s

def is_finished(board):
    for mv in range(225):
        if board[mv] != EMPTY:
            player = board[mv]
            if winner_after_move(board, mv, player) is not None:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv.gz")
    ap.add_argument("--n", type=int, default=20, help="抽出する盤面数")
    ap.add_argument("--stones-min", type=int, default=10)
    ap.add_argument("--stones-max", type=int, default=35)
    ap.add_argument("--use-vct", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-json", default="scripts/fixed_positions.json")
    ap.add_argument("--out-html", default="scripts/fixed_positions.html")
    ap.add_argument("--sample-rate", type=float, default=0.002,
                    help="行をこの確率でサンプリング(速度調整)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    data_path = PROJECT_ROOT / args.data

    # カテゴリ別に最大 n//2 ずつ集める
    buckets: dict[str, list] = {"attack": [], "block": []}
    per_bucket = max(1, args.n // 2)
    n_scanned = 0
    n_candidate = 0

    print(f"スキャン中: {data_path} (sample_rate={args.sample_rate}) ...")
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if rng.random() > args.sample_rate:
                continue
            if all(len(buckets[k]) >= per_bucket for k in buckets):
                break
            n_scanned += 1
            board = [int(x) for x in row[:225]]
            stones = sum(1 for x in board if x != EMPTY)
            if not (args.stones_min <= stones <= args.stones_max):
                continue
            if is_finished(board):
                continue
            try:
                kind, mv = oracle(board, use_vct=args.use_vct)
            except Exception:
                continue
            if kind is None or mv < 0:
                continue
            if not legal_move_mask(board)[mv]:
                continue
            if len(buckets[kind]) >= per_bucket:
                continue
            n_candidate += 1
            entry = {
                "id": n_candidate,
                "board": board,
                "stones": stones,
                "kind": kind,
                "oracle_move": mv,
                "oracle_row": int(mv // N),
                "oracle_col": int(mv % N),
                "to_move": "black" if infer_player(board) == BLACK else "white",
            }
            if kind == "block":
                cs = block_defense_set(board, use_vct=args.use_vct)
                entry["correct_set"] = sorted(cs)
            buckets[kind].append(entry)
            print(f"  [{kind}] #{len(buckets[kind])}/{per_bucket}  stones={stones}  move=({mv//N},{mv%N})")

    positions = buckets["attack"] + buckets["block"]
    positions.sort(key=lambda x: x["stones"])

    # ID を振り直す
    for i, p in enumerate(positions):
        p["id"] = i + 1

    # JSON 保存
    out_json = PROJECT_ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(positions, f, ensure_ascii=False, indent=2)
    print(f"\n保存: {out_json}  ({len(positions)} 局面)")

    # HTML 可視化
    html = build_html(positions)
    out_html = PROJECT_ROOT / args.out_html
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"可視化: {out_html}")


def build_html(positions: list[dict]) -> str:
    stone_color = {"black": "#222", "white": "#eee"}

    def board_svg(pos, size=300):
        board = pos["board"]
        oracle_mv = pos["oracle_move"]
        correct_set = set(pos.get("correct_set", [oracle_mv]))
        kind = pos["kind"]
        to_move = pos["to_move"]

        cell = size / N
        pad = cell * 0.7
        total = size + pad * 2

        lines = []
        lines.append(f'<svg width="{total:.0f}" height="{total:.0f}" xmlns="http://www.w3.org/2000/svg">')
        lines.append(f'<rect width="{total:.0f}" height="{total:.0f}" fill="#d4a843"/>')

        # グリッド線
        for i in range(N):
            x = pad + i * cell
            lines.append(f'<line x1="{x:.1f}" y1="{pad:.1f}" x2="{x:.1f}" y2="{pad + (N-1)*cell:.1f}" stroke="#555" stroke-width="0.8"/>')
            y = pad + i * cell
            lines.append(f'<line x1="{pad:.1f}" y1="{y:.1f}" x2="{pad + (N-1)*cell:.1f}" y2="{y:.1f}" stroke="#555" stroke-width="0.8"/>')

        # 天元・星
        stars = [(3,3),(3,11),(7,7),(11,3),(11,11)]
        for (sr, sc) in stars:
            sx = pad + sc * cell
            sy = pad + sr * cell
            lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="#555"/>')

        r_stone = cell * 0.44

        for idx in range(225):
            r, c = divmod(idx, N)
            cx = pad + c * cell
            cy = pad + r * cell
            stone = board[idx]
            if stone == BLACK:
                lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_stone:.1f}" fill="#222" stroke="#000" stroke-width="0.5"/>')
            elif stone == WHITE:
                lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_stone:.1f}" fill="#f5f5f5" stroke="#555" stroke-width="1"/>')

        # オラクル手をハイライト
        # attack=赤、block=青
        highlight_color = "#e33" if kind == "attack" else "#33e"
        for mv in correct_set:
            r, c = divmod(mv, N)
            cx = pad + c * cell
            cy = pad + r * cell
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r_stone*0.55:.1f}" fill="{highlight_color}" opacity="0.85"/>')

        lines.append('</svg>')
        return "\n".join(lines)

    cards = []
    for pos in positions:
        svg = board_svg(pos)
        to_move_jp = "黒" if pos["to_move"] == "black" else "白"
        kind_jp = "攻め (attack)" if pos["kind"] == "attack" else "受け (block)"
        color_cls = "attack" if pos["kind"] == "attack" else "block"
        oracle_coord = f"({pos['oracle_row']}, {pos['oracle_col']})"
        correct_moves = pos.get("correct_set", [pos["oracle_move"]])
        correct_str = ", ".join(f"({m//N},{m%N})" for m in sorted(correct_moves)[:5])
        if len(correct_moves) > 5:
            correct_str += f" ... (+{len(correct_moves)-5})"

        cards.append(f"""
<div class="card {color_cls}">
  <div class="card-header">
    <span class="id">#{pos['id']}</span>
    <span class="badge {color_cls}">{kind_jp}</span>
    <span class="meta">手番:{to_move_jp} / {pos['stones']}石</span>
  </div>
  {svg}
  <div class="info">
    オラクル手: {oracle_coord}<br>
    正解手集合: {correct_str}
  </div>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>Fixed Positions - TSS 模倣テスト用盤面</title>
<style>
body {{ font-family: sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }}
h1 {{ text-align: center; color: #adf; margin-bottom: 4px; }}
.subtitle {{ text-align: center; color: #88a; margin-bottom: 20px; font-size: 0.9em; }}
.grid {{ display: flex; flex-wrap: wrap; gap: 18px; justify-content: center; }}
.card {{ background: #16213e; border-radius: 10px; padding: 12px; box-shadow: 0 2px 8px #0006; }}
.card.attack {{ border-top: 3px solid #e33; }}
.card.block  {{ border-top: 3px solid #33e; }}
.card-header {{ display: flex; align-items: center; gap: 8px; margin-bottom: 8px; }}
.id {{ font-weight: bold; color: #adf; font-size: 1.1em; }}
.badge {{ padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: bold; }}
.badge.attack {{ background: #e33; color: #fff; }}
.badge.block  {{ background: #33e; color: #fff; }}
.meta {{ font-size: 0.8em; color: #88a; }}
.info {{ font-size: 0.75em; color: #aac; margin-top: 6px; line-height: 1.6; }}
.legend {{ text-align: center; color: #88a; margin-bottom: 16px; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>TSS 模倣テスト用 固定盤面</h1>
<p class="subtitle">data.csv.gz から抽出した {len(positions)} 局面 |
  <span style="color:#e33">■ 赤丸 = 攻め手 (attack)</span> &nbsp;
  <span style="color:#33e">■ 青丸 = 受け手 (block)</span>
</p>
<div class="grid">
{"".join(cards)}
</div>
</body>
</html>"""
    return html


if __name__ == "__main__":
    main()
