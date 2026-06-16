#!/usr/bin/env python3
# scripts/play_human_vs_mcts.py
import sys
import ctypes
import random
from pathlib import Path

# 1. パスの設定と自作ルールのインポート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from renju_transformer.rules import infer_player, winner_after_move, is_forbidden_for_black, board_with_move
except ImportError:
    print("Error: Could not import renju_transformer.rules. Ensure you are running this from the project root.")
    sys.exit(1)

# 2. C++ MCTS ライブラリのロード (OS対応)
lib_name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
dll_path = Path(lib_name).absolute()
if not dll_path.exists():
    dll_path = PROJECT_ROOT / lib_name

if not dll_path.exists():
    print(f"\n[Error] {lib_name} が見つかりません。")
    print("MCTSライブラリをビルドしてから実行してください。")
    print(f"ビルドコマンド (ターミナルで実行):")
    if sys.platform == "win32":
        print(f"  g++ -O3 -shared -fPIC -std=c++17 mcts.cpp -o mcts.dll")
    else:
        print(f"  g++ -O3 -shared -fPIC -std=c++17 mcts.cpp -o mcts.so")
    sys.exit(1)

try:
    lib = ctypes.CDLL(str(dll_path))
    lib.run_mcts_c_api_with_policy_and_visits.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # board_array
        ctypes.c_int,                    # simulations
        ctypes.c_uint64,                 # seed
        ctypes.POINTER(ctypes.c_double), # prior_probs (None for pure MCTS)
        ctypes.POINTER(ctypes.c_int)     # visits_out
    ]
    lib.run_mcts_c_api_with_policy_and_visits.restype = ctypes.c_double

    # VCFソルバーのC-API型定義
    lib.solve_vcf_c_api.argtypes = [
        ctypes.POINTER(ctypes.c_int),    # board_array
        ctypes.c_int,                    # player
        ctypes.c_int                     # max_depth
    ]
    lib.solve_vcf_c_api.restype = ctypes.c_int
except Exception as e:
    print(f"Error loading MCTS library C-API: {e}")
    sys.exit(1)

# 3. 便利関数の定義
COLS = "ABCDEFGHIJKLMNO"

def print_fancy_board(board):
    """盤面を綺麗に描画する関数"""
    print("\n     " + " ".join(COLS) + " ")
    print("   " + "   " + "-" * 29)
    for r in range(15):
        row_str = f"{r+1:2d} | "
        for c in range(15):
            idx = r * 15 + c
            stone = board[idx]
            if stone == 0:
                # 簡易的なドット表示 (天元や星は + にする)
                if idx in (112, 48, 56, 168, 176): # 天元(112), 四隅 of 星
                    row_str += "+ "
                else:
                    row_str += ". "
            elif stone == 1:
                row_str += "● "
            elif stone == 2:
                row_str += "○ "
        row_str += f"| {r+1:2d}"
        print(row_str)
    print("   " + "   " + "-" * 29)
    print("     " + " ".join(COLS) + " \n")

def parse_coordinate(user_input):
    """'H8' や 'h8' などの入力をインデックス(0-224)に変換する。不正なら -1"""
    user_input = user_input.strip().upper()
    if not user_input:
        return -1
    
    try:
        # アルファベット(列)と数字(行)を分離
        col_char = user_input[0]
        if col_char not in COLS:
            return -1
        
        row_num = int(user_input[1:])
        if row_num < 1 or row_num > 15:
            return -1
        
        c = COLS.index(col_char)
        r = row_num - 1
        return r * 15 + c
    except ValueError:
        return -1

def _run_mcts_search(board, simulations=1000):
    """純粋モンテカルロ木探索 (事前確率なし) で最善手を選ぶ"""
    board_array = (ctypes.c_int * 225)(*board)
    visits_array = (ctypes.c_int * 225)()
    seed = random.randint(0, 2**32 - 1)
    
    lib.run_mcts_c_api_with_policy_and_visits(
        board_array,
        simulations,
        seed,
        None,
        visits_array
    )
    
    visits = list(visits_array)
    current_player = infer_player(board)
    
    # 合法手のみをフィルタリング
    legal_moves = []
    for i in range(225):
        if board[i] == 0:
            if current_player == 1 and is_forbidden_for_black(board, i):
                continue
            legal_moves.append(i)
            
    if not legal_moves:
        return -1
        
    # 訪問回数 visits が最大の手を選択
    best_move = max(legal_moves, key=lambda m: visits[m])
    
    if visits[best_move] == 0:
        def dist_to_center(m):
            r, c = divmod(m, 15)
            return (r - 7)**2 + (c - 7)**2
        best_move = min(legal_moves, key=dist_to_center)
        
    return best_move

def get_mcts_move(board, simulations=1000, max_vcf_depth=12):
    """VCF（連続四による勝利）手順の探索を優先し、なければモンテカルロ木探索で最善手を選ぶ"""
    board_array = (ctypes.c_int * 225)(*board)
    current_player = infer_player(board)
    opponent = 2 if current_player == 1 else 1

    # 1. 自身の VCF 勝ち手順があるかチェック (最大12手先の勝ち)
    my_vcf = lib.solve_vcf_c_api(board_array, current_player, max_vcf_depth)
    if my_vcf >= 0:
        print("[*] AIがVCFによる強制勝利手順（連打）を検出しました！勝ち手を打ちます。")
        return my_vcf

    # 2. 相手の VCF 勝ち手順があるかチェック (最大12手先、あれば防御)
    opp_vcf = lib.solve_vcf_c_api(board_array, opponent, max_vcf_depth)
    if opp_vcf >= 0:
        print("[*] AIが相手のVCF脅威を検出しました！防御します。")
        return opp_vcf

    # 3. どちらも無ければ、通常のモンテカルロ木探索
    return _run_mcts_search(board, simulations)

# 4. メイン対局ループ
def play():
    print("==============================================")
    print("      人間 vs ただのモンテカルロ木探索 (MCTS)")
    print("==============================================")
    
    # 先番・後番の選択
    while True:
        choice = input("あなたの手番を選んでください (1: 黒(先手), 2: 白(後手)): ").strip()
        if choice == "1":
            human_player = 1
            ai_player = 2
            break
        elif choice == "2":
            human_player = 2
            ai_player = 1
            break
        print("1 か 2 を入力してください。")
        
    # シミュレーション数の選択
    while True:
        sim_input = input("MCTSのシミュレーション回数を指定してください (推奨: 1000 - 5000): ").strip()
        try:
            simulations = int(sim_input)
            if simulations > 0:
                break
        except ValueError:
            pass
        print("正の整数を入力してください。")

    board = [0] * 225
    print("\nゲーム開始！")
    
    # 黒の第1手は天元(H8)に固定する連珠ルール
    if human_player == 1:
        print("連珠ルールにより、黒の第1手は天元(H8)に自動で配置されます。")
        board[112] = 1
        turn = 2 # 次は白(AI)
    else:
        print("AIが天元(H8)に第1手を打ちました。")
        board[112] = 1
        turn = 2 # 次は白(人間)

    while True:
        print_fancy_board(board)
        
        # 勝者チェック: 盤面全体から5連を探す
        winner = None
        for r in range(15):
            for c in range(15):
                idx = r * 15 + c
                if board[idx] == 0:
                    continue
                p = board[idx]
                # 4方向（右、下、右下、左下）に5連があるか
                directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
                for dr, dc in directions:
                    count = 1
                    for step in range(1, 5):
                        nr, nc = r + dr * step, c + dc * step
                        if 0 <= nr < 15 and 0 <= nc < 15:
                            if board[nr * 15 + nc] == p:
                                count += 1
                            else:
                                break
                        else:
                            break
                    if count >= 5:
                        winner = p
                        break
                if winner:
                    break
            if winner:
                break
                
        if winner:
            if winner == human_player:
                print("★ あなたの勝ちです！おめでとうございます！ ★")
            else:
                print("☆ AIの勝ちです！対局ありがとうございました。 ☆")
            break
            
        # 満局（引き分け）
        if board.count(0) == 0:
            print("引き分け（満局）です。")
            break

        # 手番の処理
        if turn == human_player:
            # 人間のターン
            current_color = "黒 ●" if human_player == 1 else "白 ○"
            print(f"【あなたの手番 ({current_color})】")
            
            while True:
                move_str = input("着手位置を入力してください (例: H8, G9, J10): ").strip()
                move_idx = parse_coordinate(move_str)
                
                if move_idx == -1:
                    print("入力形式が不正です。'H8' や 'G9' のように入力してください。")
                    continue
                    
                if board[move_idx] != 0:
                    print("既に石が置かれています。他の場所を選択してください。")
                    continue
                    
                if human_player == 1 and is_forbidden_for_black(board, move_idx):
                    print("黒番の禁手（三三、四四、長連）です！打つことはできません。")
                    continue
                    
                # 合法手であれば配置
                board[move_idx] = human_player
                break
            
            turn = ai_player
            
        else:
            # AIのターン
            current_color = "黒 ●" if ai_player == 1 else "白 ○"
            print(f"【AIの思考中 ({current_color})... シミュレーション数: {simulations}】")
            
            ai_move = get_mcts_move(board, simulations)
            if ai_move == -1:
                print("AIに合法手が見つかりません。パスします。")
            else:
                col_char = COLS[ai_move % 15]
                row_num = (ai_move // 15) + 1
                print(f"AIは {col_char}{row_num} に着手しました。")
                board[ai_move] = ai_player
                
            turn = human_player

if __name__ == "__main__":
    play()
