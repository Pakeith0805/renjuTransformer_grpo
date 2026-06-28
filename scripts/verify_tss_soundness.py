#!/usr/bin/env python
"""
TSS ソルバーの健全性検証器（偽陽性ハンター）
=============================================
目的: ソルバー（現行 solve_vcf / 将来 solve_vct）が「必勝」と返した手が、本当に
**全ての防御に対して強制勝ちか**を独立に確認する。VCT を入れる前提の安全網。

なぜ要るか: VCF/VCT ソルバーは学習報酬・評価オラクル・推論MCTSが共有する単一実装。
偽陽性（ありもしない必勝を勝ちと返す）が1つでもあると、報酬も指標も推論も同時に汚染される。
現行VCFは「四は受けが一意」ゆえ健全なはず → ここで 100% を確認し、VCT追加時の回帰テストにする。

検証ロジック（攻め手はソルバーに従い、守りは全ローカル合法手を試す AND-OR 確認）:
  attacker_wins(board):
    mv = solver(board, attacker)            # ソルバーの推奨必勝初手
    打って五なら True。
    そうでなければ(=四/三の脅威) → 守り側の "近傍の全合法手" を一つ残らず試し、
    どれに対しても attacker_wins(...) が True なら True。1つでも逃げられたら False。
  → ソルバーが健全なら、必勝と返した全局面で True になるはず。False が出たら偽陽性=バグ。

使い方:
  uv run python scripts/verify_tss_soundness.py --positions 100 --seed 0
  （現行 VCF を検証。--depth は VCF 探索深さ、--verify-depth は確認木の深さ）
"""
import argparse
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from renju_transformer.rules import infer_player, winner_after_move, board_with_move, legal_move_mask
# ソルバーと局面生成は test_tss_imitation のものを再利用
from test_tss_imitation import vcf as solve_vcf, gen_random_position

N = 225
SIDE = 15
BLACK, WHITE = 1, 2


def other(p):
    return WHITE if p == BLACK else BLACK


def local_empty(board):
    """石からチェビシェフ距離2以内の空きマス（防御の候補=この超集合に含まれるはず）。"""
    has_stone = [board[i] != 0 for i in range(N)]
    cands = []
    for i in range(N):
        if board[i] != 0:
            continue
        r, c = divmod(i, SIDE)
        near = False
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                rr, cc = r + dr, c + dc
                if 0 <= rr < SIDE and 0 <= cc < SIDE and has_stone[rr * SIDE + cc]:
                    near = True
                    break
            if near:
                break
        if near:
            cands.append(i)
    return cands


def attacker_wins(board, attacker, solver, vcf_depth, depth, budget):
    """attacker 手番。solver が示す手順が「全防御に対して」強制勝ちか確認できれば True。
    budget 枯渇/depth 切れは False（=確認不能。安全側に倒す）。"""
    if budget[0] <= 0:
        return False
    budget[0] -= 1
    mv = solver(board, attacker, vcf_depth)
    if mv < 0:
        return False                      # ソルバーは必勝と言っていない
    b1 = board_with_move(board, mv, attacker)
    if winner_after_move(b1, mv, attacker) == attacker:
        return True                       # 五完成=勝ち
    if depth <= 0:
        return False
    defender = other(attacker)
    for d in local_empty(b1):
        b2 = board_with_move(b1, d, defender)
        if winner_after_move(b2, d, defender) == defender:
            return False                  # 守りが先に五=攻めの必勝主張は偽
        if not attacker_wins(b2, attacker, solver, vcf_depth, depth - 1, budget):
            return False                  # この防御を切れていない=逃げられた
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=100, help="検証する『必勝』局面の目標数")
    ap.add_argument("--depth", type=int, default=12, help="ソルバー(VCF)の探索深さ")
    ap.add_argument("--verify-depth", type=int, default=14, help="確認木の深さ")
    ap.add_argument("--budget", type=int, default=40000, help="1局面あたりの確認ノード上限")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    solver = solve_vcf

    verified = false_positive = unconfirmed = 0
    checked = 0
    attempts = 0
    fp_examples = []
    print(f"健全性検証: ソルバー=VCF depth={args.depth} verify_depth={args.verify_depth} "
          f"target={args.positions}", file=sys.stderr)

    while checked < args.positions and attempts < args.positions * 200 + 5000:
        attempts += 1
        board = gen_random_position(rng, args.kmin, args.kmax)
        if board is None:
            continue
        player = infer_player(board)
        if solver(board, player, args.depth) < 0:
            continue                       # 必勝主張のある局面だけ検証
        checked += 1
        budget = [args.budget]
        ok = attacker_wins(board, player, solver, args.depth, args.verify_depth, budget)
        if ok:
            verified += 1
        elif budget[0] <= 0:
            unconfirmed += 1               # 予算切れ=判定保留(偽陽性とは断定しない)
        else:
            false_positive += 1            # 予算内で逃げられた=ソルバーの偽陽性=バグ
            if len(fp_examples) < 5:
                fp_examples.append(board)
        if checked % 20 == 0:
            print(f"  {checked}/{args.positions}  verified={verified} "
                  f"false_pos={false_positive} unconfirmed={unconfirmed}", file=sys.stderr)

    print("\n==== 健全性検証 結果 ====")
    print(f"検証した必勝局面: {checked}")
    print(f"  [OK]   強制勝ちを確認 (verified): {verified}")
    print(f"  [BUG]  偽陽性 (false positive=逃げられた): {false_positive}")
    print(f"  [HOLD] 予算切れで保留 (unconfirmed): {unconfirmed}")
    if checked:
        print(f"健全率(verified / (verified+false_pos)) = "
              f"{verified / max(verified + false_positive, 1) * 100:.1f}%")
    print("\n判定: VCF は理論上健全なので false_positive=0 が期待値。")
    print("      VCT 実装後は同じスクリプトを回し、false_positive=0 を必須ゲートにする。")
    def opp_immediate_five(board):
        """手番でない側(相手)がその場で五を作れる空きが在るか=VCFが相手の脅威を無視した偽陽性原因。"""
        me = infer_player(board)
        opp = other(me)
        for i in range(N):
            if board[i] == 0 and winner_after_move(board_with_move(board, i, opp), i, opp) == opp:
                return True
        return False

    if false_positive:
        n_opp5 = sum(1 for b in fp_examples if opp_immediate_five(b))
        print(f"\n[偽陽性 {len(fp_examples)}例の分類] 相手に即五あり(VCFが相手脅威を無視): "
              f"{n_opp5}/{len(fp_examples)}")
        b = fp_examples[0]
        print("[例] 手番=", "黒" if infer_player(b) == BLACK else "白",
              " 相手即五=", opp_immediate_five(b))
        sym = {0: " .", 1: " #", 2: " O"}
        print("    " + "".join(f"{c:2d}" for c in range(SIDE)))
        for r in range(SIDE):
            print(f"{r:2d} " + "".join(sym[b[r * SIDE + c]] for c in range(SIDE)))


if __name__ == "__main__":
    main()
