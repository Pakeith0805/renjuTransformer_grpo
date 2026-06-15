import torch
# 1. 保存されたチェックポイントをロード
checkpoint = torch.load("./artifacts/grpo_checkpoint_600.pt", map_location="cpu")

# 2. 保存されているモデルの重みに nan があるかチェック (可能性Bの検証)
has_nan = any(torch.isnan(p).any() for p in checkpoint["model_state_dict"].values())
print("モデルパラメータにNaNが含まれているか:", has_nan)

# 3. 保存されているプール（局面リスト）の中に、合法手が0個の盤面があるかチェック (可能性Aの検証)
if "trajectory_boards" in checkpoint:
    from renju_transformer.tokenizer import RenjuTokenizer
    tokenizer = RenjuTokenizer()
    
    zero_legal_boards_count = 0
    for idx, board in enumerate(checkpoint["trajectory_boards"]):
        mask = tokenizer.legal_move_mask(board)
        if not mask.any():
            zero_legal_boards_count += 1
            
    print(f"プール内の全 {len(checkpoint['trajectory_boards'])} 局面中、合法手が0の局面数:", zero_legal_boards_count)