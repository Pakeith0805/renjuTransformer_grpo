import sys
import subprocess
import concurrent.futures
from pathlib import Path
import torch
from torch.distributions import Categorical

from renju_transformer.rules import infer_player, winner_after_move

def run_mcts_eval(board: list[int], move_idx: int, simulations: int = 200, seed: int = 42) -> float:
    board_str = ",".join(map(str, board))
    
    exe_path = Path("mcts.exe").absolute()
    cmd = [
        str(exe_path),
        "--eval",
        "--board", board_str,
        "--move", str(move_idx),
        "--simulations", str(simulations),
        "--seed", str(seed)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            if line.startswith("win_rate="):
                return float(line.split("=")[1])
        raise ValueError("MCTS did not output win_rate")
    except Exception as e:
        print(f"Error running MCTS eval: {e}", file=sys.stderr)
        return 0.5

class GRPOAgent:
    def __init__(self, policy_model, ref_model, tokenizer, device, mcts_simulations=200):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.device = device
        self.mcts_simulations = mcts_simulations

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
        win_rate = run_mcts_eval(initial_board_state, first_move_idx, self.mcts_simulations)
        reward = 2.0 * win_rate - 1.0 # 0～1を-1～1に変換している
        
        # 視覚化のためにボードを返している。本来不要。
        board = initial_board_state.copy()
        first_player = infer_player(board)
        board[first_move_idx] = first_player
        return reward, board

    def rollout_group(self, initial_board_state, move_indices) -> tuple[list[float], list[int]]:
        """指定された複数の着手を並列でMCTS評価し、報酬リストを返します"""
        rewards = [0.0] * len(move_indices)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(move_indices)) as executor: # 並列処理のためのクラス
            futures = {
                executor.submit(
                    run_mcts_eval, 
                    initial_board_state, 
                    move_idx, 
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
