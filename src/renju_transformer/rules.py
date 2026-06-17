"""Renju move legality helpers used during inference."""

from __future__ import annotations

import sys
import ctypes
from pathlib import Path

BOARD_SIZE = 15
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE
EMPTY = 0
BLACK = 1
WHITE = 2
DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))
CENTER_INDEX = (BOARD_SIZE // 2) * BOARD_SIZE + (BOARD_SIZE // 2)

_mcts_lib = None

def _get_mcts_lib():
    global _mcts_lib
    if _mcts_lib is None:
        try:
            lib_name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
            dll_path = Path(lib_name).absolute()
            if not dll_path.exists():
                dll_path = Path(__file__).resolve().parent.parent.parent / lib_name
            
            if dll_path.exists():
                _mcts_lib = ctypes.CDLL(str(dll_path))
                _mcts_lib.is_forbidden_for_black_c_api.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int
                ]
                _mcts_lib.is_forbidden_for_black_c_api.restype = ctypes.c_int
                
                _mcts_lib.get_legal_moves_c_api.argtypes = [
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int)
                ]
                _mcts_lib.get_legal_moves_c_api.restype = None
        except Exception:
            pass
    return _mcts_lib

# 0～224を盤面の座標に変換
def idx_to_rc(index: int) -> tuple[int, int]:
    return divmod(index, BOARD_SIZE)

# 逆変換
def rc_to_idx(row: int, col: int) -> int:
    return row * BOARD_SIZE + col

# 入力されたタプルがちゃんと盤面に収まってるか判定
def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

# 今のボードと、位置と、プレイヤー名から、1手進んだ未来の盤面を返す
def board_with_move(board: list[int], index: int, player: int) -> list[int]:
    next_board = board.copy()
    next_board[index] = player
    return next_board

# 盤面上の白石と黒石の数をそれぞれ数えて返す
def stone_counts(board: list[int]) -> tuple[int, int]:
    black_count = sum(1 for cell in board if cell == BLACK)
    white_count = sum(1 for cell in board if cell == WHITE)
    return black_count, white_count

# 盤上の石の数を返す
def move_number(board: list[int]) -> int:
    black_count, white_count = stone_counts(board)
    return black_count + white_count

# drはrowのdirection、dcはcolumnのdirection。特定の方向に、同じ色の石が何個連続で並んでいるかを返す。
def contiguous_count(board: list[int], index: int, player: int, dr: int, dc: int) -> int:
    total = 1
    row, col = idx_to_rc(index)

    step = 1
    while inside(row + dr * step, col + dc * step):
        if board[rc_to_idx(row + dr * step, col + dc * step)] != player:
            break
        total += 1
        step += 1

    step = 1
    while inside(row - dr * step, col - dc * step):
        if board[rc_to_idx(row - dr * step, col - dc * step)] != player:
            break
        total += 1
        step += 1

    return total

# 今持っている石を置いたとき、5個以上石が連続で並ぶかを判定
def has_five_or_more(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 5 for dr, dc in DIRECTIONS)

# 今持っている石を置いたとき、6個以上石が連続で並んでいるかを判定
def is_overline(board: list[int], index: int, player: int) -> bool:
    return any(contiguous_count(board, index, player, dr, dc) >= 6 for dr, dc in DIRECTIONS)

# 盤面全体を探索し、五連以上を達成している箇所があるかを判定
def player_has_five(board: list[int], player: int) -> bool:
    return any(cell == player and has_five_or_more(board, index, player) for index, cell in enumerate(board))

# 盤面全体を探索し、六連以上を達成している箇所があるかを判定
def player_has_overline(board: list[int], player: int) -> bool:
    return any(cell == player and is_overline(board, index, player) for index, cell in enumerate(board))

# 指定した位置を通り、特定の方向に延びる直線状のマスを端から端まですべてリストにして取得
def line_points_through(index: int, dr: int, dc: int) -> list[int]:
    row, col = idx_to_rc(index)
    while inside(row - dr, col - dc):
        row -= dr
        col -= dc

    points: list[int] = []
    while inside(row, col):
        points.append(rc_to_idx(row, col))
        row += dr
        col += dc
    return points

# 特定の直線状で、あと1手で勝ちになる位置の集合を返す
def immediate_wins_in_direction(board: list[int], player: int, line_points: list[int]) -> set[int]:
    wins: set[int] = set()
    for candidate in line_points:
        if board[candidate] != EMPTY:
            continue
        next_board = board_with_move(board, candidate, player)
        if player == BLACK and is_overline(next_board, candidate, BLACK): # 黒の6連以上は反則
            continue
        if has_five_or_more(next_board, candidate, player):
            wins.add(candidate)
    return wins

# 指定した位置に打ったとき、いくつの方向で4連になるかを数える。
def count_four_directions(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points = line_points_through(move, dr, dc)
        if immediate_wins_in_direction(board, player, line_points):
            count += 1
    return count

# いくつの方向で活三が出来るか判定
def count_open_three_directions(board: list[int], move: int, player: int) -> int:
    count = 0
    for dr, dc in DIRECTIONS:
        line_points = line_points_through(move, dr, dc)
        found_open_three = False
        for candidate in line_points:
            if board[candidate] != EMPTY:
                continue
            next_board = board_with_move(board, candidate, player)
            if player == BLACK and is_overline(next_board, candidate, BLACK):
                continue
            winning_points = immediate_wins_in_direction(next_board, player, line_points)
            if len(winning_points) >= 2:
                found_open_three = True
                break
        if found_open_three:
            count += 1
    return count

# 盤面上の黒石と白石の数から、次はどちらの手番かを判定
def infer_player(board: list[int]) -> int:
    black_count, white_count = stone_counts(board)
    if black_count == white_count:
        return BLACK
    if black_count == white_count + 1:
        return WHITE
    raise ValueError(
        f"Invalid board: black_count={black_count}, white_count={white_count}. "
        "Expected black == white or black == white + 1."
    )

# 黒にとって、そのマスが禁じ手かどうか判定
def is_forbidden_for_black(board: list[int], index: int) -> bool:
    try:
        lib = _get_mcts_lib()
        if lib is not None:
            board_array = (ctypes.c_int * 225)(*board)
            return lib.is_forbidden_for_black_c_api(board_array, index) == 1
    except Exception:
        pass
    return _is_forbidden_for_black_python(board, index)

def _is_forbidden_for_black_python(board: list[int], index: int) -> bool:
    if board[index] != EMPTY:
        return True

    black_count, white_count = stone_counts(board)
    move_number = black_count + white_count

    if move_number == 0:
        return index != CENTER_INDEX

    next_board = board_with_move(board, index, BLACK)
    if is_overline(next_board, index, BLACK):
        return True
    if count_four_directions(next_board, index, BLACK) >= 2:
        return True
    if count_open_three_directions(next_board, index, BLACK) >= 2:
        return True
    return False

# 盤面全体の合法手のリストを作る
def legal_move_mask(board: list[int]) -> list[bool]:
    try:
        lib = _get_mcts_lib()
        if lib is not None:
            player = infer_player(board)
            board_array = (ctypes.c_int * 225)(*board)
            mask_array = (ctypes.c_int * 225)()
            lib.get_legal_moves_c_api(board_array, player, mask_array)
            return [val == 1 for val in mask_array]
    except Exception:
        pass
    return _legal_move_mask_python(board)

def _legal_move_mask_python(board: list[int]) -> list[bool]:
    player = infer_player(board)
    mask: list[bool] = []
    for index, cell in enumerate(board):
        if cell != EMPTY:
            mask.append(False)
            continue
        if player == BLACK:
            mask.append(not is_forbidden_for_black(board, index))
        else:
            mask.append(True)
    return mask

# 石を置いた直後に勝敗が決まるか判定
def winner_after_move(board: list[int], index: int, player: int) -> int | None:
    if player == BLACK and is_overline(board, index, BLACK):
        return WHITE
    if has_five_or_more(board, index, player):
        return player
    return None

# 盤面全体をスキャンして、既に勝負がついているかを判定
def board_winner(board: list[int]) -> int | None:
    if player_has_overline(board, BLACK):
        return WHITE
    if player_has_five(board, BLACK):
        return BLACK
    if player_has_five(board, WHITE):
        return WHITE
    return None
