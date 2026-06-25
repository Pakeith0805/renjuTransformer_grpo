import sys
from pathlib import Path
import torch
import hydra
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.tokenizer import RenjuTokenizer
from renju_transformer.rules import infer_player, winner_after_move
from renju_transformer.utils import select_device, set_seed
from grpo.load_model import print_board

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

@hydra.main(version_base="1.3", config_path="../config", config_name="config_versus")
def main(cfg: DictConfig) -> None:
    set_seed(cfg.seed)
    device = select_device(cfg.versus.device)
    
    tokenizer = RenjuTokenizer(
        sep_token_id=cfg.data.sep_token_id,
        move_id_offset=cfg.data.move_id_offset,
    )
    
    print(f"Loading Model A: {cfg.versus.model_a_path}")
    model_a = load_model(cfg.versus.model_a_path, device)
    print(f"Loading Model B: {cfg.versus.model_b_path}")
    model_b = load_model(cfg.versus.model_b_path, device)
    
    num_games = cfg.versus.num_games
    temperature = cfg.versus.temperature
    # 開幕ランダム化のオンオフ。0 = 従来通り(単一シード・固定開幕)。
    random_opening_plies = int(cfg.versus.get("random_opening_plies", 0))

    msg = f"\nStarting {num_games} matches evaluation (temperature={temperature}"
    if random_opening_plies > 0:
        msg += f", random_opening_plies={random_opening_plies}"
    print(msg + ")...")

    stats = {
        "a_wins_black": 0,
        "a_wins_white": 0,
        "b_wins_black": 0,
        "b_wins_white": 0,
        "draws": 0,
        "total_plies": 0,
    }
    # 棋譜の多様性を測る: 着手列をそのまま署名にしてユニーク数を数える。
    # 単一ラインに頼っているほど unique_games が小さくなる。
    seen_games = set()

    for game_idx in range(1, num_games + 1):
        # 開幕ランダム化時のみ対局ごとに別シードを振り、開幕とサンプリングを散らす。
        # 無効時は冒頭の set_seed(cfg.seed) のまま(従来挙動を完全維持)。
        if random_opening_plies > 0:
            set_seed(cfg.seed + game_idx)

        # Odd games: Model A = Black, Model B = White
        # Even games: Model B = Black, Model A = White
        is_model_a_black = (game_idx % 2 == 1)

        black_name = "Model A" if is_model_a_black else "Model B"
        white_name = "Model B" if is_model_a_black else "Model A"
        print(f"Game {game_idx}/{num_games}: [Black] {black_name} vs [White] {white_name}")

        board = [0] * 225
        winner = None
        plies = 0
        move_history = []

        # --- 開幕ランダム手 (両者ランダム) ----------------------------------
        # <=8手なら連が成立せず開幕中に決着しないので winner 判定は不要。
        for _ in range(random_opening_plies):
            current_player = infer_player(board)
            legal_mask = tokenizer.legal_move_mask(board)
            legal_idx = legal_mask.nonzero(as_tuple=True)[0]
            if legal_idx.numel() == 0:
                break
            pick = torch.randint(legal_idx.numel(), (1,)).item()
            move_idx = int(legal_idx[pick].item())
            board[move_idx] = current_player
            plies += 1
            move_history.append(move_idx)

        for ply in range(1, 226):
            current_player = infer_player(board)
            current_is_a = (current_player == 1) if is_model_a_black else (current_player == 2)
            current_model = model_a if current_is_a else model_b
            
            legal_mask = tokenizer.legal_move_mask(board).to(device)
            if not legal_mask.any():
                break
                
            input_ids = tokenizer.encode_input(board).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = current_model(input_ids).squeeze(0)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                
                if temperature == 0.0:
                    move_idx = masked_logits.argmax().item()
                else:
                    probs = torch.softmax(masked_logits / temperature, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    move_idx = dist.sample().item()

                # current_entropy = torch.distributions.Categorical(probs=probs).entropy().item()
                # print(f"[DEBUG] Current Ply Entropy: {current_entropy:.3f}")
            
            board[move_idx] = current_player
            plies += 1
            move_history.append(move_idx)

            winner = winner_after_move(board, move_idx, current_player)
            if winner is not None:
                break

        stats["total_plies"] += plies
        seen_games.add(tuple(move_history))
        
        if winner is None:
            stats["draws"] += 1
            print(f"  Result: Draw in {plies} plies")
        else:
            winner_is_a = (winner == 1) if is_model_a_black else (winner == 2)
            if winner_is_a:
                if is_model_a_black:
                    stats["a_wins_black"] += 1
                else:
                    stats["a_wins_white"] += 1
                print(f"  Result: Model A won in {plies} plies ({'Black' if is_model_a_black else 'White'})")
            else:
                if is_model_a_black:
                    stats["b_wins_white"] += 1
                else:
                    stats["b_wins_black"] += 1
                print(f"  Result: Model B won in {plies} plies ({'White' if is_model_a_black else 'Black'})")
                
        # Print the final board state of the very last match
        print("\nFinal Game Board State:")
        print_board(board)
            
    # Calculate stats
    a_wins = stats["a_wins_black"] + stats["a_wins_white"]
    b_wins = stats["b_wins_black"] + stats["b_wins_white"]
    draws = stats["draws"]
    avg_plies = stats["total_plies"] / num_games
    
    a_win_rate = (a_wins / num_games) * 100
    b_win_rate = (b_wins / num_games) * 100
    draw_rate = (draws / num_games) * 100
    
    print("=" * 60)
    print(" EVALUATION RESULTS ")
    print("=" * 60)
    print(f"Total Matches Played: {num_games}")
    print(f"Average Game Length:  {avg_plies:.1f} plies")
    print(f"Unique Games:         {len(seen_games)} / {num_games} "
          f"({len(seen_games) / num_games * 100:.0f}% distinct)")
    print(f"Draws:                {draws} ({draw_rate:.1f}%)")
    print("-" * 60)
    print(f"Model A (Black Wins: {stats['a_wins_black']}, White Wins: {stats['a_wins_white']})")
    print(f"  Total Wins: {a_wins} / {num_games} ({a_win_rate:.1f}%)")
    print(f"Model B (Black Wins: {stats['b_wins_black']}, White Wins: {stats['b_wins_white']})")
    print(f"  Total Wins: {b_wins} / {num_games} ({b_win_rate:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    main()
