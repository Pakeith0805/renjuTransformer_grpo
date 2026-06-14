import sys
import os
import ctypes
import concurrent.futures
from pathlib import Path
import torch
from torch.distributions import Categorical

from renju_transformer.rules import infer_player, winner_after_move, board_with_move

# DLLのロードと関数の初期化
_mcts_lib = None

def _get_mcts_lib():
    global _mcts_lib
    if _mcts_lib is None:
        dll_path = Path("mcts.dll").absolute()
        if not dll_path.exists():
            raise FileNotFoundError(f"mcts.dll not found at {dll_path}. Please build it first.")
        
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
            ctypes.POINTER(ctypes.c_double)  # prior_probs
        ]
        _mcts_lib.run_mcts_c_api_with_policy.restype = ctypes.c_double

    return _mcts_lib

def run_mcts_eval(board: list[int], move_idx: int, simulations: int = 200, seed: int = 42) -> float:
    try:
        lib = _get_mcts_lib()
        board_array = (ctypes.c_int * 225)(*board)
        win_rate = lib.run_mcts_c_api(board_array, move_idx, simulations, seed)
        return win_rate
    except Exception as e:
        print(f"Error running MCTS eval (DLL): {e}", file=sys.stderr)
        return 0.5

def run_mcts_eval_with_policy(board: list[int], move_idx: int, prior_probs: list[float], simulations: int = 200, seed: int = 42) -> float:
    try:
        lib = _get_mcts_lib()
        board_array = (ctypes.c_int * 225)(*board)
        probs_array = (ctypes.c_double * 225)(*prior_probs)
        win_rate = lib.run_mcts_c_api_with_policy(board_array, move_idx, simulations, seed, probs_array)
        return win_rate
    except Exception as e:
        print(f"Error running MCTS eval with policy (DLL): {e}", file=sys.stderr)
        return 0.5

class GRPOAgent:
    def __init__(self, policy_model, ref_model, tokenizer, device, mcts_simulations=200):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.device = device
        self.mcts_simulations = mcts_simulations
        # あらかじめメインスレッドでDLLをロードし、スレッド間の初期化競合を防ぐ
        try:
            _get_mcts_lib()
        except Exception as e:
            print(f"Warning: Failed to pre-load MCTS DLL: {e}", file=sys.stderr)

    def get_group_actions(self, board_state, group_size=8, temperature=1.0):
        """盤面から G 個のアクションと、Policy/Ref それぞれ of 対数確率を計算して返す"""
        input_ids = self.tokenizer.encode_input(board_state).unsqueeze(0).to(self.device)
        legal_mask = self.tokenizer.legal_move_mask(board_state).to(self.device)

        policy_logits = self.policy(input_ids).squeeze(0)
        masked_policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

        policy_probs = torch.softmax(masked_policy_logits / temperature, dim=-1)
        policy_dist = Categorical(probs = policy_probs)
        sample_actions = policy_dist.sample((group_size, ))

        log_probs_policy = policy_dist.log_prob(sample_actions)
        with torch.no_grad():
            ref_logits = self.ref(input_ids).squeeze(0)
            masked_ref_logits = ref_logits.masked_fill(~legal_mask, float("-inf"))
            ref_probs = torch.softmax(masked_ref_logits / temperature, dim=-1)
            ref_dist = Categorical(probs=ref_probs)
            log_probs_ref = ref_dist.log_prob(sample_actions)

        return sample_actions, log_probs_policy, log_probs_ref
    
    def rollout_single_game(self, initial_board_state, first_move_idx, max_plies = 225, temperature = 1.0) -> tuple[float, list[int]]:
        """互換性のための直列版（実際には rollout_group を推奨）"""
        player = infer_player(initial_board_state)
        next_board = board_with_move(initial_board_state, first_move_idx, player)
        input_ids = self.tokenizer.encode_input(next_board).unsqueeze(0).to(self.device)
        legal_mask = self.tokenizer.legal_move_mask(next_board).to(self.device)
        with torch.no_grad():
            logits = self.policy(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(masked_logits, dim=-1).cpu().numpy().tolist()

        win_rate = run_mcts_eval_with_policy(initial_board_state, first_move_idx, probs, self.mcts_simulations)
        reward = 2.0 * win_rate - 1.0 # 0～1を-1～1に変換している
        
        # 視覚化のためにボードを返している。本来不要。
        board = initial_board_state.copy()
        first_player = infer_player(board)
        board[first_move_idx] = first_player
        return reward, board

    def rollout_group(self, initial_board_state, move_indices) -> tuple[list[float], list[int]]:
        """指定された複数の着手を並列でMCTS評価し、報酬リストを返します"""
        rewards = [0.0] * len(move_indices)
        
        # 1. 各手を打った後の盤面を作成し、そこでの次のプレイヤーの手に対するポリシー事前確率を一括バッチ推論する
        player = infer_player(initial_board_state)
        batch_input_ids = []
        batch_legal_masks = []
        
        for move_idx in move_indices:
            next_board = board_with_move(initial_board_state, move_idx, player)
            input_ids = self.tokenizer.encode_input(next_board)
            batch_input_ids.append(input_ids)
            legal_mask = self.tokenizer.legal_move_mask(next_board)
            batch_legal_masks.append(legal_mask)
            
        if batch_input_ids:
            input_ids_tensor = torch.stack(batch_input_ids).to(self.device)
            legal_masks_tensor = torch.stack(batch_legal_masks).to(self.device)
            with torch.no_grad():
                logits = self.policy(input_ids_tensor) # (batch_size, 225)
                masked_logits = logits.masked_fill(~legal_masks_tensor, float("-inf"))
                probs_tensor = torch.softmax(masked_logits, dim=-1)
                batch_probs = probs_tensor.cpu().numpy().tolist()
        else:
            batch_probs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(move_indices)) as executor: # 並列処理のためのクラス
            futures = {
                executor.submit(
                    run_mcts_eval_with_policy, 
                    initial_board_state, 
                    move_idx, 
                    batch_probs[i],
                    self.mcts_simulations,
                    seed=42 + i * 997 # MCTsが同じ挙動をしないようにシードを散らす
                ): i
                for i, move_idx in enumerate(move_indices)
            }
            
            # 処理が終わった順に報酬(勝率)を回収。
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    win_rate = future.result()
                    rewards[idx] = 2.0 * win_rate - 1.0
                except Exception as e:
                    rewards[idx] = 0.0  # Draw on failure
                    
        # 描画用の盤面として、最初の着手を行った盤面を返す
        last_board = initial_board_state.copy()
        if len(move_indices) > 0:
            first_player = infer_player(last_board)
            last_board[move_indices[0]] = first_player
            
        return rewards, last_board

    def select_move_via_mcts(self, board_state, simulations=1000) -> int:
        """モデルにガイドされたMCTSを実行し、最も評価の高い手（勝率最大の候補手）を選びます。
        速度向上のため、モデルの事前確率の上位10手（Top-10）のみを探索候補に絞り込みます。
        """
        legal_mask = self.tokenizer.legal_move_mask(board_state).to(self.device)
        legal_moves = [i for i, is_legal in enumerate(legal_mask.tolist()) if is_legal]
        
        if not legal_moves:
            raise RuntimeError("No legal moves available to select.")
            
        if len(legal_moves) == 1:
            return legal_moves[0]
            
        # 1. まず現在の局面でモデルを1回推論し、上位10手を決める (Top-k Pruning)
        input_ids = self.tokenizer.encode_input(board_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.policy(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            prior_probs_on_current = torch.softmax(masked_logits, dim=-1)
            
        # 合法手の中からモデル事前確率の高い上位10手を選択する
        top_k = min(10, len(legal_moves))
        top_values, top_indices = torch.topk(prior_probs_on_current, k=top_k)
        candidate_moves = top_indices.cpu().numpy().tolist()
        
        # 2. 厳選された上位10個の候補手を打った後の局面を生成してモデルに入力し、バッチ推論する
        player = infer_player(board_state)
        batch_input_ids = []
        batch_legal_masks = []
        
        for move_idx in candidate_moves:
            next_board = board_with_move(board_state, move_idx, player)
            input_ids = self.tokenizer.encode_input(next_board)
            batch_input_ids.append(input_ids)
            legal_mask = self.tokenizer.legal_move_mask(next_board)
            batch_legal_masks.append(legal_mask)
            
        if batch_input_ids:
            input_ids_tensor = torch.stack(batch_input_ids).to(self.device)
            legal_masks_tensor = torch.stack(batch_legal_masks).to(self.device)
            with torch.no_grad():
                logits = self.policy(input_ids_tensor)
                masked_logits = logits.masked_fill(~legal_masks_tensor, float("-inf"))
                probs_tensor = torch.softmax(masked_logits, dim=-1)
                batch_probs = probs_tensor.cpu().numpy().tolist()
        else:
            batch_probs = []
            
        rewards = [0.0] * len(candidate_moves)
        
        # 絞り込んだ候補手についてのみ並列でMCTSを実行
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(candidate_moves), 16)) as executor:
            futures = {
                executor.submit(
                    run_mcts_eval_with_policy, 
                    board_state, 
                    move_idx, 
                    batch_probs[i],
                    simulations,
                    seed=42 + i * 997
                ): i
                for i, move_idx in enumerate(candidate_moves)
            }
            
            for future in concurrent.futures.as_completed(futures):
                idx = futures[future]
                try:
                    win_rate = future.result()
                    rewards[idx] = win_rate
                except Exception as e:
                    rewards[idx] = 0.5  # Draw on failure
                    
        # 最も評価（勝率）の高い手を選択
        best_idx = int(torch.tensor(rewards).argmax().item())
        return candidate_moves[best_idx]
