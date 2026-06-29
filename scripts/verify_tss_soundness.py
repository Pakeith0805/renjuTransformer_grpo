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
import ctypes
import sys
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from renju_transformer.rules import (
    infer_player, winner_after_move, board_with_move, legal_move_mask, count_four_directions,
)
# ソルバーと局面生成は test_tss_imitation のものを再利用
from test_tss_imitation import vcf as solve_vcf, gen_random_position

N = 225
SIDE = 15
BLACK, WHITE = 1, 2


def other(p):
    return WHITE if p == BLACK else BLACK


_path_lib = None


def vcf_line(board, player, depth):
    """solve_vcf_path で solve_vcf が主張する勝ち手順(攻め→受け→攻め…)を [(r,c,'攻/受'),...] で返す。"""
    global _path_lib
    if _path_lib is None:
        name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
        _path_lib = ctypes.CDLL(str(PROJECT_ROOT / name))
        _path_lib.solve_vcf_path_c_api.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
        _path_lib.solve_vcf_path_c_api.restype = ctypes.c_int
    arr = (ctypes.c_int * N)(*board)
    out = (ctypes.c_int * 128)()
    n = _path_lib.solve_vcf_path_c_api(arr, player, depth, out)
    seq = []
    for i in range(n):
        mv = out[i]
        seq.append((mv // SIDE, mv % SIDE, "攻" if i % 2 == 0 else "受"))
    return seq


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


def explain_fp(board, vcf_depth, verify_depth, budget_n):
    """偽陽性局面で『どの防御手で逃げられるか』と、その手の性質を特定する。
    戻り値: 説明文字列。"""
    attacker = infer_player(board)
    defender = other(attacker)
    mv = solve_vcf(board, attacker, vcf_depth)
    if mv < 0:
        return "再現せず(ソルバーが必勝を返さない)"
    b1 = board_with_move(board, mv, attacker)
    if winner_after_move(b1, mv, attacker) == attacker:
        return "初手で即五(本物の勝ち, 再現せず)"
    # 攻めの初手後の『攻め側の五点』(=防御の強制ブロック候補)
    five_pts = set(i for i in range(N) if b1[i] == 0
                   and winner_after_move(board_with_move(b1, i, attacker), i, attacker) == attacker)
    for d in local_empty(b1):
        b2 = board_with_move(b1, d, defender)
        if winner_after_move(b2, d, defender) == defender:
            continue  # 防御の即五(P0で除外済みのはず)はスキップ
        if not attacker_wins(b2, attacker, solve_vcf, vcf_depth, verify_depth - 1, [budget_n]):
            r, c = divmod(d, SIDE)
            is_block = d in five_pts
            dfour = count_four_directions(b2, d, defender)  # d で防御側が作る四の方向数
            mr, mc = divmod(mv, SIDE)
            return (f"逃げ手 d=({r},{c}) [強制ブロック点か={is_block}] "
                    f"d後の防御側四方向数={dfour} / 攻め初手mv=({mr},{mc}) 五点数={len(five_pts)}")
    return "逃げ手を再特定できず(予算/深さ依存=検証側の限界の可能性)"


def _imm_five(board, player):
    for m in local_empty(board):
        if winner_after_move(board_with_move(board, m, player), m, player) == player:
            return True
    return False


def _attacker_threats(board, player):
    """四 or 活三を作る攻め手(=脅威手)。VCTの攻め候補。"""
    from renju_transformer.rules import count_four_directions, count_open_three_directions
    out = []
    for m in local_empty(board):
        nb = board_with_move(board, m, player)
        if count_four_directions(nb, m, player) >= 1 or count_open_three_directions(nb, m, player) >= 1:
            out.append(m)
    return out


def _atk_five_squares(board, attacker):
    """attacker が次に1手で五を作れる空きマス(=四の完成点)の集合。"""
    return [s for s in local_empty(board)
            if winner_after_move(board_with_move(board, s, attacker), s, attacker) == attacker]


def bf_forced_win(board, attacker, depth, budget):
    """総当たり強制勝ちオラクル(独立した真の地面)。攻め手番で強制勝ちが在るか。
    戻り: True/False/None(予算切れ=不明)。
    正しい防御モデルで分岐を抑える(健全性は保つ):
      攻めの各脅威手 m を打った後、攻めの五完成点 W を数える:
        - 守りに即五 → 守り勝ち、m 失敗
        - |W|>=2     → 達四/二重四=勝ち(守りは両方塞げない)
        - |W|==1     → 守りはその1点を**強制ブロック**(単一分岐)→再帰
        - |W|==0(活三)→ 守りは複数。健全のため全ローカル手を試し、全てに勝てれば勝ち
    solve_vcf/vct と独立。これらの健全性(偽陽性)も完全性(取りこぼし)も此処が基準。"""
    if budget[0] <= 0:
        return None
    budget[0] -= 1
    defender = other(attacker)
    if _imm_five(board, attacker):
        return True
    if depth <= 0:
        return False
    saw_unknown = False
    for m in _attacker_threats(board, attacker):
        b1 = board_with_move(board, m, attacker)
        if winner_after_move(b1, m, attacker) == attacker:
            return True
        if _imm_five(b1, defender):
            continue  # 守りが即五で勝つ → この攻めは不成立
        W = _atk_five_squares(b1, attacker)
        if len(W) >= 2:
            return True  # 達四/二重四 → 勝ち
        if len(W) == 1:
            d = W[0]
            b2 = board_with_move(b1, d, defender)
            if winner_after_move(b2, d, defender) == defender:
                continue  # ブロックが守りの五になる(自勝ち) → m 失敗
            r = bf_forced_win(b2, attacker, depth - 1, budget)
            if r is True:
                return True
            if r is None:
                saw_unknown = True
        else:
            replies = local_empty(b1)  # 活三: 守りは複数(健全のため全ローカル)
            all_win = True
            for d in replies:
                b2 = board_with_move(b1, d, defender)
                if winner_after_move(b2, d, defender) == defender:
                    all_win = False
                    break
                r = bf_forced_win(b2, attacker, depth - 1, budget)
                if r is None:
                    all_win = None
                    break
                if not r:
                    all_win = False
                    break
            if all_win is True:
                return True
            if all_win is None:
                saw_unknown = True
    return None if saw_unknown else False


def audit_line(board, depth):
    """solve_vcf の主張手順を1手ずつ監査して、各手の性質を返す（残存FPの正体特定用）。
    - 攻: 即五か / 完成点数(=win_squares) / 終端か
    - 受: 攻の完成点数 / 防御側に即五があるか / 受けが唯一強制か
    全『受』が唯一強制 かつ 終端の『攻』が本物勝ち(完成点>=2 or 即五) なら solve_vcf は正しく、
    検証器のアーティファクト。逆なら solve_vcf の実バグ。"""
    attacker = infer_player(board)
    defender = other(attacker)
    line = vcf_line(board, attacker, depth)
    b = list(board)
    out = []
    for i, (r, c, role) in enumerate(line):
        mv = r * SIDE + c
        if role == "攻":
            b_after = board_with_move(b, mv, attacker)
            five = winner_after_move(b_after, mv, attacker) == attacker
            wsq = sum(1 for s in range(N) if b_after[s] == 0
                      and winner_after_move(board_with_move(b_after, s, attacker), s, attacker) == attacker)
            tag = " <terminal>" if i == len(line) - 1 else ""
            out.append(f"  攻({r},{c}) 即五={five} 完成点数={wsq}{tag}")
            b = b_after
        else:
            comp = [s for s in range(N) if b[s] == 0
                    and winner_after_move(board_with_move(b, s, attacker), s, attacker) == attacker]
            dfive = any(b[s] == 0
                        and winner_after_move(board_with_move(b, s, defender), s, defender) == defender
                        for s in range(N))
            unique = (len(comp) == 1 and not dfive)
            out.append(f"  受({r},{c}) 攻の完成点数={len(comp)} 防御側即五={dfive} 唯一強制={unique}")
            b = board_with_move(b, mv, defender)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=int, default=100, help="検証する『必勝』局面の目標数")
    ap.add_argument("--depth", type=int, default=12, help="ソルバー(VCF)の探索深さ")
    ap.add_argument("--verify-depth", type=int, default=14, help="確認木の深さ")
    ap.add_argument("--budget", type=int, default=40000, help="1局面あたりの確認ノード上限")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bruteforce", action="store_true",
                    help="偽陽性局面を総当たりオラクルで裁定(実バグ/アーティファクト判定)")
    ap.add_argument("--bf-depth", type=int, default=7, help="総当たりオラクルの攻め手深さ")
    ap.add_argument("--bf-budget", type=int, default=400000, help="総当たりオラクルのノード上限")
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
        for k, b in enumerate(fp_examples):
            print(f"  例{k}: " + explain_fp(b, args.depth, args.verify_depth, args.budget))
            line = vcf_line(b, infer_player(b), args.depth)
            print("        solve_vcfの主張手順: " + " ".join(f"{t}({r},{c})" for r, c, t in line))
            print("        [手順監査]")
            for ln in audit_line(b, args.depth):
                print("      " + ln)
            if args.bruteforce:
                res = bf_forced_win(b, infer_player(b), args.bf_depth, [args.bf_budget])
                verdict = {True: "強制勝ち在り → solve_vcfは正/検証器側の問題",
                           False: "強制勝ち無し → solve_vcfの実偽陽性=本物のバグ",
                           None: "不明(予算切れ。--bf-budget/--bf-depth 調整)"}[res]
                print(f"        [総当たり裁定] {verdict}")
        b = fp_examples[0]
        print("[盤面例0] 手番=", "黒" if infer_player(b) == BLACK else "白",
              " 相手即五=", opp_immediate_five(b))
        sym = {0: " .", 1: " #", 2: " O"}
        print("    " + "".join(f"{c:2d}" for c in range(SIDE)))
        for r in range(SIDE):
            print(f"{r:2d} " + "".join(sym[b[r * SIDE + c]] for c in range(SIDE)))


if __name__ == "__main__":
    main()
