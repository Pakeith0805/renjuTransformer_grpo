import sys
import os
import ctypes
import random
from pathlib import Path
import torch
import numpy as np
import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.rules import infer_player, winner_after_move, board_with_move, is_forbidden_for_black
from renju_transformer.utils import select_device, set_seed
from grpo.load_model import print_board
from grpo.agent import _get_mcts_lib

def load_model(checkpoint_path: str | Path, device: torch.device) -> RenjuTransformerModel:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config")
    if checkpoint_config is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain 'config' field.")
    model_cfg = checkpoint_config["model"]
    model = RenjuTransformerModel(
        vocab_size=model_cfg["token_vocab_size"],
        max_seq_len=model_cfg["max_seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        activation=model_cfg["activation"],
        norm_first=model_cfg["norm_first"],
        num_move_labels=model_cfg["num_move_labels"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model

def select_move_via_mcts_standalone(
    board_state: list[int],
    policy_model: RenjuTransformerModel | None,  # None if uniform MCTS
    simulations: int,
    use_tss: bool,
    use_puct: bool,
    tokenizer: RenjuTokenizer,
    device: torch.device,
    temperature: float = 0.1,
    max_vcf_depth: int = 12
) -> int:
    lib = _get_mcts_lib()
    current_player = infer_player(board_state)
    opponent = 2 if current_player == 1 else 1

    # VCF勝ち手順・防御のチェック
    if use_tss:
        board_array = (ctypes.c_int * 225)(*board_state)
        # 1. 自身の VCF 勝ち手順があるかチェック
        my_vcf = lib.solve_vcf_c_api(board_array, current_player, max_vcf_depth)
        if my_vcf >= 0:
            return my_vcf

        # 2. 相手の VCF 勝ち手順があるかチェック (ブロック)
        opp_vcf = lib.solve_vcf_c_api(board_array, opponent, max_vcf_depth)
        if opp_vcf >= 0:
            return opp_vcf

    legal_mask = tokenizer.legal_move_mask(board_state).to(device)
    legal_moves = [i for i, is_legal in enumerate(legal_mask.tolist()) if is_legal]
    
    if not legal_moves:
        raise RuntimeError("No legal moves available to select.")
        
    if len(legal_moves) == 1:
        return legal_moves[0]

    # 事前確率 (Prior probabilities) の取得
    if policy_model is not None:
        input_ids = tokenizer.encode_input(board_state).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = policy_model(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            prior_probs = torch.softmax(masked_logits, dim=-1).cpu().numpy().tolist()
    else:
        # モデルなし (一様分布) の事前確率
        prior_probs = [1.0 / 225] * 225

    board_array = (ctypes.c_int * 225)(*board_state)
    probs_array = (ctypes.c_double * 225)(*prior_probs)
    visits_array = (ctypes.c_int * 225)()

    seed = random.randint(0, 10000000)
    lib.run_mcts_c_api_with_policy_and_visits(
        board_array, simulations, seed, probs_array, visits_array, 1 if use_puct else 0
    )

    visits = list(visits_array)
    legal_visits = {move: visits[move] for move in legal_moves}

    # 温度パラメータを適用して最終的な手を選択
    if temperature == 0.0:
        best_move = max(legal_visits, key=legal_visits.get)
        return best_move
    else:
        moves = list(legal_visits.keys())
        counts = np.array(list(legal_visits.values()), dtype=np.float32)
        if counts.sum() < 1e-6:
            # フォールバック: 事前確率に基づく選択
            return random.choices(moves, weights=[prior_probs[m] for m in moves], k=1)[0]
        
        power_counts = counts ** (1.0 / temperature)
        probs = power_counts / power_counts.sum()
        return random.choices(moves, weights=probs, k=1)[0]

def play_single_game(
    game_idx: int,
    num_games: int,
    is_model_black: bool,
    model: RenjuTransformerModel,
    ref_model: RenjuTransformerModel | None,
    tokenizer: RenjuTokenizer,
    device: torch.device,
    model_temp: float,
    mcts_sims: int,
    use_tss: bool,
    use_puct: bool,
    mcts_temp: float
) -> dict:
    black_name = "Target Model" if is_model_black else "MCTS Opponent"
    white_name = "MCTS Opponent" if is_model_black else "Target Model"
    print(f"Game {game_idx}/{num_games} started: [Black] {black_name} vs [White] {white_name}")
    
    board = [0] * 225
    winner = None
    plies = 0
    
    for ply in range(1, 226):
        current_player = infer_player(board)
        current_is_model = (current_player == 1) if is_model_black else (current_player == 2)
        
        legal_mask = tokenizer.legal_move_mask(board).to(device)
        if not legal_mask.any():
            break
            
        if current_is_model:
            # Target model move selection (Raw Policy)
            input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(input_ids).squeeze(0)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                if model_temp == 0.0:
                    move_idx = masked_logits.argmax().item()
                else:
                    probs = torch.softmax(masked_logits / model_temp, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    move_idx = dist.sample().item()
        else:
            # MCTS opponent move selection
            move_idx = select_move_via_mcts_standalone(
                board_state=board,
                policy_model=ref_model,
                simulations=mcts_sims,
                use_tss=use_tss,
                use_puct=use_puct,
                tokenizer=tokenizer,
                device=device,
                temperature=mcts_temp
            )
            
        board[move_idx] = current_player
        plies += 1
        
        winner = winner_after_move(board, move_idx, current_player)
        if winner is not None:
            break
            
    # Result reporting
    result_str = ""
    winner_is_model = None
    if winner is None:
        result_str = f"  Game {game_idx}/{num_games} finished: Draw in {plies} plies"
    else:
        winner_is_model = (winner == 1) if is_model_black else (winner == 2)
        if winner_is_model:
            result_str = f"  Game {game_idx}/{num_games} finished: Target Model won in {plies} plies ({'Black' if is_model_black else 'White'})"
        else:
            result_str = f"  Game {game_idx}/{num_games} finished: MCTS Opponent won in {plies} plies ({'White' if is_model_black else 'Black'})"
            
    print(result_str)
    
    return {
        "plies": plies,
        "winner": winner,
        "winner_is_model": winner_is_model,
        "is_model_black": is_model_black,
        "board": board
    }

@hydra.main(version_base="1.3", config_path="../config", config_name="config_eval_mcts")
def main(cfg: DictConfig) -> None:
    import concurrent.futures
    set_seed(cfg.seed)
    device = select_device(cfg.eval_mcts.device)
    
    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    
    print(f"Loading Target Model: {cfg.eval_mcts.model_path}")
    model = load_model(cfg.eval_mcts.model_path, device)
    
    # MCTS の事前確率モデルをロード (uniform の場合はロードしない)
    ref_model_path = cfg.eval_mcts.ref_model_path
    if ref_model_path.lower() == "uniform":
        print("MCTS Opponent Prior: Uniform (No model prior)")
        ref_model = None
    else:
        print(f"Loading Reference Model for MCTS: {ref_model_path}")
        ref_model = load_model(ref_model_path, device)
        
    num_games = cfg.eval_mcts.num_games
    model_temp = cfg.eval_mcts.model_temperature
    mcts_sims = cfg.eval_mcts.mcts_simulations
    use_tss = cfg.eval_mcts.use_tss
    use_puct = cfg.eval_mcts.use_puct
    mcts_temp = cfg.eval_mcts.mcts_temperature
    
    print(f"\nStarting {num_games} matches evaluation against MCTS Opponent (8 threads)...")
    print(f"  Target Model Temp:      {model_temp}")
    print(f"  MCTS Simulations:       {mcts_sims}")
    print(f"  MCTS TSS (VCF Solver):  {use_tss}")
    print(f"  MCTS PUCT Rule:         {use_puct}")
    print(f"  MCTS Temp:              {mcts_temp}")
    
    stats = {
        "model_wins_black": 0,
        "model_wins_white": 0,
        "mcts_wins_black": 0,
        "mcts_wins_white": 0,
        "draws": 0,
        "total_plies": 0,
        "model_win_plies": 0,   # モデルが勝った局の手数合計 (攻めの決定力)
        "model_loss_plies": 0,  # モデルが負けた局の手数合計 (守りの崩壊速度)
    }
    
    last_board = None
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for game_idx in range(1, num_games + 1):
            is_model_black = (game_idx % 2 == 1)
            future = executor.submit(
                play_single_game,
                game_idx=game_idx,
                num_games=num_games,
                is_model_black=is_model_black,
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                device=device,
                model_temp=model_temp,
                mcts_sims=mcts_sims,
                use_tss=use_tss,
                use_puct=use_puct,
                mcts_temp=mcts_temp
            )
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            stats["total_plies"] += res["plies"]
            winner = res["winner"]
            winner_is_model = res["winner_is_model"]
            is_model_black = res["is_model_black"]
            last_board = res["board"]
            
            if winner is None:
                stats["draws"] += 1
            else:
                if winner_is_model:
                    stats["model_win_plies"] += res["plies"]
                    if is_model_black:
                        stats["model_wins_black"] += 1
                    else:
                        stats["model_wins_white"] += 1
                else:
                    stats["model_loss_plies"] += res["plies"]
                    if is_model_black:
                        stats["mcts_wins_white"] += 1
                    else:
                        stats["mcts_wins_black"] += 1
                        
    # Print the final board state of the last finished game
    if last_board is not None:
        print("\nFinal Game Board State:")
        print_board(last_board)
            
    # Calculate stats
    model_wins = stats["model_wins_black"] + stats["model_wins_white"]
    mcts_wins = stats["mcts_wins_black"] + stats["mcts_wins_white"]
    draws = stats["draws"]
    avg_plies = stats["total_plies"] / num_games
    avg_win_plies = (stats["model_win_plies"] / model_wins) if model_wins else 0.0
    avg_loss_plies = (stats["model_loss_plies"] / mcts_wins) if mcts_wins else 0.0

    model_win_rate = (model_wins / num_games) * 100
    mcts_win_rate = (mcts_wins / num_games) * 100
    draw_rate = (draws / num_games) * 100
    
    print("=" * 60)
    print(" EVALUATION AGAINST MCTS RESULTS ")
    print("=" * 60)
    print(f"Total Matches Played: {num_games}")
    print(f"Average Game Length:  {avg_plies:.1f} plies")
    print(f"  Avg plies (Model WINS):  {avg_win_plies:.1f} plies ({model_wins} games)")
    print(f"  Avg plies (Model LOSES): {avg_loss_plies:.1f} plies ({mcts_wins} games)")
    print(f"Draws:                {draws} ({draw_rate:.1f}%)")
    print("-" * 60)
    print(f"Target Model (Black Wins: {stats['model_wins_black']}, White Wins: {stats['model_wins_white']})")
    print(f"  Total Wins: {model_wins} / {num_games} ({model_win_rate:.1f}%)")
    print(f"MCTS Opponent (Black Wins: {stats['mcts_wins_black']}, White Wins: {stats['mcts_wins_white']})")
    print(f"  Total Wins: {mcts_wins} / {num_games} ({mcts_win_rate:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()