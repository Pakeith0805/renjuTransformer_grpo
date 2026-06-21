import sys
import os
import ctypes
import concurrent.futures
from pathlib import Path
import torch
from torch.distributions import Categorical

from renju_transformer.rules import infer_player, winner_after_move, board_with_move, is_forbidden_for_black

# TSS (VCFソルバーによる先読み支援) の有効・無効切り替え
USE_TSS = False  # Falseに設定するとTSSを完全にオフにしてMCTSのみの探索を行います

# DLLのロードと関数の初期化
_mcts_lib = None

def _get_mcts_lib():
    global _mcts_lib
    if _mcts_lib is None:
        lib_name = "mcts.so" if sys.platform != "win32" else "mcts.dll"
        dll_path = Path(lib_name).absolute()
        if not dll_path.exists():
            # プロジェクトルートからの相対パスも試す
            dll_path = Path(__file__).resolve().parent.parent.parent / lib_name
            
        if not dll_path.exists():
            raise FileNotFoundError(f"{lib_name} not found at {dll_path}. Please build it first.")
        
        _mcts_lib = ctypes.CDLL(str(dll_path))
        _mcts_lib.run_mcts_c_api.argtypes = [
            ctypes.POINTER(ctypes.c_int), # board_array
            ctypes.c_int,                 # move_idx
            ctypes.c_int,                 # simulations
            ctypes.c_uint64               # seed
        ]
        _mcts_lib.run_mcts_c_api.restype = ctypes.c_double

        # 新しいAPI（ポリシーモデルガイド付きMCTS）のロード
        _mcts_lib.run_mcts_c_api_with_policy.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # board_array
            ctypes.c_int,                    # move_idx
            ctypes.c_int,                    # simulations
            ctypes.c_uint64,                 # seed
            ctypes.POINTER(ctypes.c_double), # prior_probs
            ctypes.c_int                     # use_puct
        ]
        _mcts_lib.run_mcts_c_api_with_policy.restype = ctypes.c_double

        # 新API（訪問回数配列の書き戻し付き単一MCTS）のロード
        _mcts_lib.run_mcts_c_api_with_policy_and_visits.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # board_array
            ctypes.c_int,                    # simulations
            ctypes.c_uint64,                 # seed
            ctypes.POINTER(ctypes.c_double), # prior_probs
            ctypes.POINTER(ctypes.c_int),    # visits_out
            ctypes.c_int                     # use_puct
        ]
        _mcts_lib.run_mcts_c_api_with_policy_and_visits.restype = ctypes.c_double

        # VCFソルバーのC-API型定義
        _mcts_lib.solve_vcf_c_api.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # board_array
            ctypes.c_int,                    # player
            ctypes.c_int                     # max_depth
        ]
        _mcts_lib.solve_vcf_c_api.restype = ctypes.c_int

        # VCFソルバーパス付きC-API型定義
        _mcts_lib.solve_vcf_path_c_api.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # board_array
            ctypes.c_int,                    # player
            ctypes.c_int,                    # max_depth
            ctypes.POINTER(ctypes.c_int)     # path_out
        ]
        _mcts_lib.solve_vcf_path_c_api.restype = ctypes.c_int

        # 黒番の禁手判定C-APIの型定義を追加
        _mcts_lib.is_forbidden_for_black_c_api.argtypes = [
            ctypes.POINTER(ctypes.c_int),    # board_array
            ctypes.c_int                     # move_idx
        ]
        _mcts_lib.is_forbidden_for_black_c_api.restype = ctypes.c_int

        # 手数ペナルティ設定用 C-API
        _mcts_lib.set_length_penalty_c_api.argtypes = [
            ctypes.c_int,                    # use_penalty
            ctypes.c_double                  # coef
        ]
        _mcts_lib.set_length_penalty_c_api.restype = None

    return _mcts_lib

def run_mcts_eval(board: list[int], move_idx: int, simulations: int = 200, seed: int = 42, max_vcf_depth: int = 12,
                  use_tss: bool = False, use_puct: bool = False) -> float:
    try:
        lib = _get_mcts_lib()
        player = infer_player(board)
        opponent = 2 if player == 1 else 1

        if board[move_idx] != 0:
            return 0.0
        if player == 1 and is_forbidden_for_black(board, move_idx):
            return 0.0

        next_board = board_with_move(board, move_idx, player)
        winner = winner_after_move(next_board, move_idx, player)
        if winner is not None:
            return 1.0 if winner == player else 0.0

        # VCFチェック
        if use_tss:
            # 1. opponent に即勝ち（五連）の手がある場合は、敗退行為 (勝率0.0)
            opp_immediate_wins = [
                m for m in range(225)
                if next_board[m] == 0
                and winner_after_move(board_with_move(next_board, m, opponent), m, opponent) == opponent
            ]
            if opp_immediate_wins:
                return 0.0

            # 2. 自身 (player) に即勝ち（五連）にできる手があるかチェック
            player_immediate_wins = [
                m for m in range(225)
                if next_board[m] == 0
                and winner_after_move(board_with_move(next_board, m, player), m, player) == player
            ]
            if len(player_immediate_wins) >= 2:
                return 1.0
            elif len(player_immediate_wins) == 1:
                block_idx = player_immediate_wins[0]
                if opponent == 1 and is_forbidden_for_black(next_board, block_idx):
                    return 1.0
                else:
                    # 相手にブロックさせた局面 (player手番) でVCF勝ちがあるか
                    blocked_board = board_with_move(next_board, block_idx, opponent)
                    if winner_after_move(blocked_board, block_idx, opponent) == opponent:
                        return 0.0
                    blocked_board_array = (ctypes.c_int * 225)(*blocked_board)
                    my_vcf = lib.solve_vcf_c_api(blocked_board_array, player, max_vcf_depth)
                    if my_vcf >= 0:
                        return 1.0

            # 3. 相手 (opponent) にとって、次の局面 (next_board) で VCF 勝ち手順があるか
            if not player_immediate_wins:
                next_board_array = (ctypes.c_int * 225)(*next_board)
                opp_vcf = lib.solve_vcf_c_api(next_board_array, opponent, max_vcf_depth)
                if opp_vcf >= 0:
                    return 0.0

        board_array = (ctypes.c_int * 225)(*board)
        win_rate = lib.run_mcts_c_api(board_array, move_idx, simulations, seed)
        return win_rate
    except Exception as e:
        print(f"Error running MCTS eval (DLL): {e}", file=sys.stderr)
        return 0.5

def run_mcts_eval_with_policy(board: list[int], move_idx: int, prior_probs: list[float], 
                              simulations: int = 200, seed: int = 42, max_vcf_depth: int = 12,
                              use_tss: bool = False, use_puct: bool = False,
                              use_length_penalty: bool = False, length_penalty_coef: float = 0.02) -> float:
    try:
        lib = _get_mcts_lib()
        player = infer_player(board)
        opponent = 2 if player == 1 else 1

        if board[move_idx] != 0:
            return 0.0
        if player == 1 and is_forbidden_for_black(board, move_idx):
            return 0.0

        next_board = board_with_move(board, move_idx, player)
        winner = winner_after_move(next_board, move_idx, player)
        
        mcts_coef = length_penalty_coef / 2.0
        
        if winner is not None:
            if winner == player:
                return 1.0 - mcts_coef * 1 if use_length_penalty else 1.0
            else:
                return 0.0 + mcts_coef * 1 if use_length_penalty else 0.0

        # VCFチェック
        if use_tss:
            # 1. opponent に即勝ち（五連）の手がある場合は、敗退行為 (勝率0.0)
            opp_immediate_wins = [
                m for m in range(225)
                if next_board[m] == 0
                and winner_after_move(board_with_move(next_board, m, opponent), m, opponent) == opponent
            ]
            if opp_immediate_wins:
                return 0.0 + mcts_coef * 2 if use_length_penalty else 0.0

            # 2. 自身 (player) に即勝ち（五連）にできる手があるかチェック
            player_immediate_wins = [
                m for m in range(225)
                if next_board[m] == 0
                and winner_after_move(board_with_move(next_board, m, player), m, player) == player
            ]
            if len(player_immediate_wins) >= 2:
                return 1.0 - mcts_coef * 3 if use_length_penalty else 1.0
            elif len(player_immediate_wins) == 1:
                block_idx = player_immediate_wins[0]
                if opponent == 1 and is_forbidden_for_black(next_board, block_idx):
                    return 1.0 - mcts_coef * 3 if use_length_penalty else 1.0
                else:
                    # 相手にブロックさせた局面 (player手番) でVCF勝ちがあるか
                    blocked_board = board_with_move(next_board, block_idx, opponent)
                    if winner_after_move(blocked_board, block_idx, opponent) == opponent:
                        return 0.0 + mcts_coef * 3 if use_length_penalty else 0.0
                    blocked_board_array = (ctypes.c_int * 225)(*blocked_board)
                    my_vcf = lib.solve_vcf_c_api(blocked_board_array, player, max_vcf_depth)
                    if my_vcf >= 0:
                        if use_length_penalty:
                            path_array = (ctypes.c_int * 256)()
                            path_len = lib.solve_vcf_path_c_api(blocked_board_array, player, max_vcf_depth, path_array)
                            total_len = 2 + (path_len if path_len > 0 else 0)
                            return 1.0 - mcts_coef * total_len
                        else:
                            return 1.0

            # 3. 相手 (opponent) にとって、次の局面 (next_board) で VCF 勝ち手順があるか
            if not player_immediate_wins:
                next_board_array = (ctypes.c_int * 225)(*next_board)
                opp_vcf = lib.solve_vcf_c_api(next_board_array, opponent, max_vcf_depth)
                if opp_vcf >= 0:
                    if use_length_penalty:
                        path_array = (ctypes.c_int * 256)()
                        path_len = lib.solve_vcf_path_c_api(next_board_array, opponent, max_vcf_depth, path_array)
                        total_len = 1 + (path_len if path_len > 0 else 0)
                        return 0.0 + mcts_coef * total_len
                    else:
                        return 0.0

        # DLLに手数ペナルティ設定を設定
        lib.set_length_penalty_c_api(1 if use_length_penalty else 0, mcts_coef)

        board_array = (ctypes.c_int * 225)(*board)
        probs_array = (ctypes.c_double * 225)(*prior_probs)
        win_rate = lib.run_mcts_c_api_with_policy(board_array, move_idx, simulations, seed, probs_array, 1 if use_puct else 0)
        return win_rate
    except Exception as e:
        print(f"Error running MCTS eval with policy (DLL): {e}", file=sys.stderr)
        return 0.5

def run_mcts_eval_with_policy_and_visits(board: list[int], simulations: int, seed: int, prior_probs: list[float],
                                         use_puct: bool = False) -> tuple[float, list[int]]:
    try:
        lib = _get_mcts_lib()
        board_array = (ctypes.c_int * 225)(*board)
        probs_array = (ctypes.c_double * 225)(*prior_probs)
        visits_array = (ctypes.c_int * 225)()
        
        win_rate = lib.run_mcts_c_api_with_policy_and_visits(board_array, simulations, seed, probs_array, visits_array, 1 if use_puct else 0)
        
        return win_rate, list(visits_array)
    except Exception as e:
        print(f"Error running MCTS eval with policy and visits (DLL): {e}", file=sys.stderr)
        return 0.5, [0] * 225

def reconstruct_winning_states(start_board: list[int], path: list[int], win_player: int) -> list[list[int]]:
    states = []
    curr_board = start_board.copy()
    curr_player = infer_player(curr_board)
    for move in path:
        if curr_player == win_player:
            states.append(curr_board.copy())
        curr_board = board_with_move(curr_board, move, curr_player)
        curr_player = 2 if curr_player == 1 else 1
    return states

class GRPOAgent:
    def __init__(self, policy_model, ref_model, tokenizer, device, mcts_simulations=200,
                 use_tss_collection=False, use_tss_training=False,
                 use_puct_collection=False, use_puct_training=False):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.device = device
        self.mcts_simulations = mcts_simulations
        self.use_tss_collection = use_tss_collection
        self.use_tss_training = use_tss_training
        self.use_puct_collection = use_puct_collection
        self.use_puct_training = use_puct_training
        # あらかじめメインスレッドでDLLをロードし、スレッド間の初期化競合を防ぐ
        try:
            _get_mcts_lib()
        except Exception as e:
            print(f"Warning: Failed to pre-load MCTS DLL: {e}", file=sys.stderr)

    def get_group_actions(self, board_state, group_size=8, temperature=1.0, vcf_target_prob=0.4):
        """盤面から G 個のアクションと、Policy/Ref それぞれ of 対数確率、および正確なKLダイバージェンスを計算して返す"""
        import math
        input_ids = self.tokenizer.encode_input(board_state).unsqueeze(0).to(self.device)
        legal_mask = self.tokenizer.legal_move_mask(board_state).to(self.device)

        policy_logits = self.policy(input_ids).squeeze(0)
        masked_policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

        policy_probs = torch.softmax(masked_policy_logits / temperature, dim=-1)
        
        # VCF/TSS dynamic bias sampling
        biased_probs = None
        p_raw_val = None
        
        if self.use_tss_training:
            try:
                lib = _get_mcts_lib()
                current_player = infer_player(board_state)
                opponent = 2 if current_player == 1 else 1
                board_array = (ctypes.c_int * 225)(*board_state)
                
                # Check self VCF win
                vcf_move = lib.solve_vcf_c_api(board_array, current_player, 12)  # max_vcf_depth = 12
                if vcf_move < 0:
                    # Check opponent VCF win (block)
                    vcf_move = lib.solve_vcf_c_api(board_array, opponent, 12)
                    
                if vcf_move >= 0:
                    # Ensure the VCF move is marked as legal in our mask
                    if legal_mask[vcf_move]:
                        p_raw = policy_probs[vcf_move].item()
                        p_raw_val = p_raw
                        p_raw_clamped = max(1e-6, min(1.0 - 1e-6, p_raw))
                        
                        if p_raw_clamped < vcf_target_prob:
                            # Compute exact logit bias B to raise VCF probability to vcf_target_prob
                            bias = math.log(vcf_target_prob / (1.0 - vcf_target_prob)) - math.log(p_raw_clamped / (1.0 - p_raw_clamped))
                            
                            # Apply logit bias (scaled by temperature to match the softmax scaling)
                            biased_policy_logits = masked_policy_logits.clone()
                            biased_policy_logits[vcf_move] += bias * temperature
                            biased_probs = torch.softmax(biased_policy_logits / temperature, dim=-1)
            except Exception as e:
                print(f"Warning: VCF dynamic bias failed in get_group_actions: {e}", file=sys.stderr)

        # Sampling Actions: from biased_probs if exists, otherwise raw policy_probs
        sampling_probs = biased_probs if biased_probs is not None else policy_probs
        behavior_dist = Categorical(probs=sampling_probs)
        sample_actions = behavior_dist.sample((group_size,))

        # log_probs_policy (with gradient, unbiased): evaluate under original policy distribution
        policy_dist = Categorical(probs=policy_probs)
        log_probs_policy = policy_dist.log_prob(sample_actions)
        
        # log_probs_old (detached, biased): evaluate under sampling behavior distribution
        log_probs_old = behavior_dist.log_prob(sample_actions).detach()

        # log_probs_ref (detached, unbiased): evaluate under reference distribution
        with torch.no_grad():
            ref_logits = self.ref(input_ids).squeeze(0)
            masked_ref_logits = ref_logits.masked_fill(~legal_mask, float("-inf"))
            ref_probs = torch.softmax(masked_ref_logits / temperature, dim=-1)
            ref_dist = Categorical(probs=ref_probs)
            log_probs_ref = ref_dist.log_prob(sample_actions)

        # 盤面全体（全225手）の正確なKLダイバージェンスを計算 (生の policy vs 生の reference)
        with torch.no_grad():
            policy_log_probs = torch.log_softmax(masked_policy_logits / temperature, dim=-1)
            ref_log_probs = torch.log_softmax(masked_ref_logits / temperature, dim=-1)
            kl_elementwise = policy_probs * (policy_log_probs - ref_log_probs)
            kl_elementwise = torch.nan_to_num(kl_elementwise, nan=0.0, posinf=0.0, neginf=0.0)
            exact_kl = kl_elementwise.sum().item()

        return sample_actions, log_probs_policy, log_probs_old, log_probs_ref, exact_kl, p_raw_val
    
    def rollout_single_game(self, initial_board_state, first_move_idx, max_plies = 225, temperature = 1.0,
                            use_length_penalty: bool = False, length_penalty_coef: float = 0.02) -> tuple[float, list[int]]:
        """互換性のための直列版（実際には rollout_group を推奨）"""
        player = infer_player(initial_board_state)
        next_board = board_with_move(initial_board_state, first_move_idx, player)
        input_ids = self.tokenizer.encode_input(next_board).unsqueeze(0).to(self.device)
        legal_mask = self.tokenizer.legal_move_mask(next_board).to(self.device)
        with torch.no_grad():
            logits = self.policy(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(masked_logits, dim=-1).cpu().numpy().tolist()

        win_rate = run_mcts_eval_with_policy(
            initial_board_state, first_move_idx, probs, self.mcts_simulations,
            use_tss=self.use_tss_training, use_puct=self.use_puct_training,
            use_length_penalty=use_length_penalty, length_penalty_coef=length_penalty_coef
        )
        reward = 2.0 * win_rate - 1.0 # 0～1を-1～1に変換している
        
        # 視覚化のためにボードを返している。本来不要。
        board = initial_board_state.copy()
        first_player = infer_player(board)
        board[first_move_idx] = first_player
        return reward, board

    def rollout_group(self, initial_board_state, move_indices,
                      use_length_penalty: bool = False, length_penalty_coef: float = 0.02) -> tuple[list[float], list[int]]:
        """指定された複数の着手を並列でMCTS評価し、報酬リストを返します (重複排除版)"""
        # move_indices の中からユニークな指し手を特定
        # 順序を保持するために dictionary を使用（Python 3.7+ では順序保存）
        unique_move_to_indices = {}
        for i, move_idx in enumerate(move_indices):
            if move_idx not in unique_move_to_indices:
                unique_move_to_indices[move_idx] = []
            unique_move_to_indices[move_idx].append(i)
            
        unique_moves = list(unique_move_to_indices.keys())
        num_unique = len(unique_moves)
        
        # 1. 各ユニークな手を打った後の盤面を作成し、そこでの次のプレイヤーの手に対するポリシー事前確率を一括バッチ推論する
        player = infer_player(initial_board_state)
        batch_input_ids = []
        batch_legal_masks = []
        
        for move_idx in unique_moves:
            next_board = board_with_move(initial_board_state, move_idx, player)
            input_ids = self.tokenizer.encode_input(next_board)
            batch_input_ids.append(input_ids)
            legal_mask = self.tokenizer.legal_move_mask(next_board)
            batch_legal_masks.append(legal_mask)
            
        if batch_input_ids:
            input_ids_tensor = torch.stack(batch_input_ids).to(self.device)
            legal_masks_tensor = torch.stack(batch_legal_masks).to(self.device)
            with torch.no_grad():
                logits = self.policy(input_ids_tensor) # (num_unique, 225)
                masked_logits = logits.masked_fill(~legal_masks_tensor, float("-inf"))
                probs_tensor = torch.softmax(masked_logits, dim=-1)
                batch_probs = probs_tensor.cpu().numpy().tolist()
        else:
            batch_probs = []
            
        unique_rewards = [0.0] * num_unique
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_unique) as executor:
            futures = {
                executor.submit(
                    run_mcts_eval_with_policy, 
                    initial_board_state, 
                    move_idx, 
                    batch_probs[i],
                    self.mcts_simulations,
                    seed=42 + i * 997, # MCTSの挙動が重ならないようシードをずらす
                    use_tss=self.use_tss_training,
                    use_puct=self.use_puct_training,
                    use_length_penalty=use_length_penalty,
                    length_penalty_coef=length_penalty_coef
                ): i
                for i, move_idx in enumerate(unique_moves)
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    win_rate = future.result()
                    unique_rewards[idx] = 2.0 * win_rate - 1.0
                except Exception as e:
                    unique_rewards[idx] = 0.0  # Draw on failure
                    
        # 2. ユニークな手の報酬を元のインデックスへマッピング
        rewards = [0.0] * len(move_indices)
        for i, move_idx in enumerate(unique_moves):
            reward = unique_rewards[i]
            for orig_idx in unique_move_to_indices[move_idx]:
                rewards[orig_idx] = reward
                
        # 描画用の盤面として、最初の着手を行った盤面を返す
        last_board = initial_board_state.copy()
        if len(move_indices) > 0:
            first_player = infer_player(last_board)
            last_board[move_indices[0]] = first_player
            
        return rewards, last_board

    def select_move_via_mcts(self, board_state, simulations=1000, temperature=1.0, use_noise=True, max_vcf_depth=12,
                             use_tss=None, use_puct=None) -> int:
        """モデルにガイドされた単一のMCTS探索を実行し、
        各候補手の訪問回数に基づいて確率的（あるいは決定論的）に指し手を選択します。
        ディリクレノイズおよび温度パラメータを適用して探索の多様性を確保します。
        """
        import numpy as np
        import random

        if use_tss is None:
            use_tss = self.use_tss_collection
        if use_puct is None:
            use_puct = self.use_puct_collection

        lib = _get_mcts_lib()
        current_player = infer_player(board_state)
        opponent = 2 if current_player == 1 else 1

        # VCF勝ち手順・防御のチェック
        if use_tss:
            # 1. 自身の VCF 勝ち手順があるかチェック
            board_array = (ctypes.c_int * 225)(*board_state)
            my_vcf = lib.solve_vcf_c_api(board_array, current_player, max_vcf_depth)
            if my_vcf >= 0:
                return my_vcf

            # 2. 相手の VCF 勝ち手順があるかチェック (ブロック)
            opp_vcf = lib.solve_vcf_c_api(board_array, opponent, max_vcf_depth)
            if opp_vcf >= 0:
                return opp_vcf

        legal_mask = self.tokenizer.legal_move_mask(board_state).to(self.device)
        legal_moves = [i for i, is_legal in enumerate(legal_mask.tolist()) if is_legal]
        
        if not legal_moves:
            raise RuntimeError("No legal moves available to select.")
            
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # 1. 現在の局面でモデル推論し、事前確率 prior_probs を取得
        input_ids = self.tokenizer.encode_input(board_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            prior_probs_on_current = torch.softmax(masked_logits, dim=-1).cpu().numpy()
            
        # 2. ディリクレノイズの追加 (自己対戦用)
        if use_noise and len(legal_moves) > 0:
            # 連珠用のノイズパラメータ: alpha = 0.03, epsilon = 0.25
            noise = np.random.dirichlet([0.03] * len(legal_moves))
            legal_probs = prior_probs_on_current[legal_moves]
            
            # 念のため正規化
            sum_legal = np.sum(legal_probs)
            if sum_legal > 0:
                legal_probs = legal_probs / sum_legal
            else:
                legal_probs = np.ones_like(legal_moves) / len(legal_moves)
                
            blended_probs = (1.0 - 0.25) * legal_probs + 0.25 * noise
            
            # prior_probs に書き戻す
            prior_probs_blended = np.zeros(225, dtype=np.float32)
            prior_probs_blended[legal_moves] = blended_probs
            prior_probs_list = prior_probs_blended.tolist()
        else:
            prior_probs_list = prior_probs_on_current.tolist()
            
        # 3. 単一 MCTS 探索の実行 (訪問回数 visits_out の取得)
        seed = random.randint(0, 2**32 - 1)
        win_rate, visits = run_mcts_eval_with_policy_and_visits(
            board_state, simulations, seed, prior_probs_list, use_puct=use_puct
        )
        
        visits_np = np.array(visits, dtype=np.float32)
        
        # 4. 温度パラメータに基づくサンプリング
        if temperature == 0.0:
            # 決定論的 (訪問回数最大の手を選択)
            best_moves = np.argwhere(visits_np == np.max(visits_np)).flatten()
            best_legal_moves = [m for m in best_moves if m in legal_moves]
            if not best_legal_moves:
                # 展開されていなかった場合などのフォールバック
                best_legal_moves = [m for m in np.argwhere(prior_probs_on_current == np.max(prior_probs_on_current)).flatten() if m in legal_moves]
            return int(random.choice(best_legal_moves))
        else:
            # 確率的サンプリング
            if np.sum(visits_np) == 0:
                # 訪問数がすべて 0 の場合は、モデルの事前確率分布でフォールバック
                probs = np.zeros(225, dtype=np.float32)
                probs[legal_moves] = prior_probs_on_current[legal_moves]
                sum_p = np.sum(probs)
                if sum_p > 0:
                    probs = probs / sum_p
                else:
                    probs[legal_moves] = 1.0 / len(legal_moves)
            else:
                # 各手の訪問数 N(a) に対し N(a)^(1/T) を計算して確率分布を構築
                power_visits = np.zeros(225, dtype=np.float32)
                for m in legal_moves:
                    power_visits[m] = np.power(visits_np[m], 1.0 / temperature)
                sum_pv = np.sum(power_visits)
                if sum_pv > 0:
                    probs = power_visits / sum_pv
                else:
                    probs = np.zeros(225, dtype=np.float32)
                    probs[legal_moves] = 1.0 / len(legal_moves)
            
            # 指し手をサンプリング
            return int(np.random.choice(225, p=probs))

    def get_vcf_path(self, board: list[int], player: int, max_depth: int = 12) -> list[int]:
        lib = _get_mcts_lib()
        board_array = (ctypes.c_int * 225)(*board)
        path_array = (ctypes.c_int * 256)()
        path_len = lib.solve_vcf_path_c_api(board_array, player, max_depth, path_array)
        if path_len <= 0:
            return []
        return list(path_array)[:path_len]

    def get_vcf_winning_path_and_player(self, board: list[int], move_idx: int) -> tuple[int, list[int], list[int]] | None:
        """
        Given the board before the move and the move_idx played, check if a VCF win is achieved.
        Returns: (winning_player, start_board_for_path, path_moves) or None
        """
        player = infer_player(board)
        opponent = 2 if player == 1 else 1
        next_board = board_with_move(board, move_idx, player)
        
        # We check the same VCF conditions as run_mcts_eval_with_policy
        # 1. Opponent has immediate win?
        opp_immediate_wins = [
            m for m in range(225)
            if next_board[m] == 0
            and winner_after_move(board_with_move(next_board, m, opponent), m, opponent) == opponent
        ]
        if opp_immediate_wins:
            return None

        # 2. Player has immediate win?
        player_immediate_wins = [
            m for m in range(225)
            if next_board[m] == 0
            and winner_after_move(board_with_move(next_board, m, player), m, player) == player
        ]
        
        lib = _get_mcts_lib()
        max_vcf_depth = 12
        
        if len(player_immediate_wins) >= 2:
            return None
        elif len(player_immediate_wins) == 1:
            block_idx = player_immediate_wins[0]
            if opponent == 1 and is_forbidden_for_black(next_board, block_idx):
                return None
            else:
                blocked_board = board_with_move(next_board, block_idx, opponent)
                if winner_after_move(blocked_board, block_idx, opponent) == opponent:
                    return None
                
                path = self.get_vcf_path(blocked_board, player, max_vcf_depth)
                if path:
                    full_path = [block_idx] + path
                    return player, next_board, full_path
        else:
            opp_path = self.get_vcf_path(next_board, opponent, max_vcf_depth)
            if opp_path:
                return opponent, next_board, opp_path
                
        return None

    def get_vcf_path_states(self, board: list[int], move_idx: int) -> list[list[int]]:
        res = self.get_vcf_winning_path_and_player(board, move_idx)
        if res is None:
            return []
        win_player, start_board_for_path, path_moves = res
        return reconstruct_winning_states(start_board_for_path, path_moves, win_player)
