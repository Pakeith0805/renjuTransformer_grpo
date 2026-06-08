import torch
from torch.distributions import Categorical

from renju_transformer.rules import infer_player, winner_after_move, is_forbidden_for_black

class GRPOAgent:
    def __init__(self, policy_model, ref_model, tokenizer, device):
        self.policy = policy_model
        self.ref = ref_model
        self.tokenizer = tokenizer
        self.device = device

    def get_group_actions(self, board_state, group_size=8, temperature=1.0):
        """盤面から G 個のアクションと、Policy/Ref それぞれの対数確率を計算して返す"""
        # unsqueezeはバッチサイズを入力するため
        input_ids = self.tokenizer.encode_input(board_state).unsqueeze(0).to(self.device)

        # 簡易マスクの取得 (空きマス 0 の箇所のみを True にする)
        legal_mask = torch.tensor([cell == 0 for cell in board_state], dtype=torch.bool, device=self.device)

        # policyモデルによる予測
        policy_logits = self.policy(input_ids).squeeze(0)

        # 非合法な手のロジットを-無限にして、確率を0にする。ビット反転。なんか変な処理だから直そう
        masked_policy_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))

        # tempuratureでスケーリングして確率分布を作成
        policy_probs = torch.softmax(masked_policy_logits / temperature, dim=-1)

        # ガチャを作る
        policy_dist = Categorical(probs = policy_probs)

        # ガチャを引く
        sample_actions = policy_dist.sample((group_size, ))

        # klダイバージェンスを計算するために、対数確率を計算
        log_probs_policy = policy_dist.log_prob(sample_actions)
        with torch.no_grad():
            ref_logits = self.ref(input_ids).squeeze(0)
            masked_ref_logits = ref_logits.masked_fill(~legal_mask, float("-inf"))
            ref_probs = torch.softmax(masked_ref_logits / temperature, dim=-1)
            ref_dist = Categorical(probs=ref_probs)
            log_probs_ref = ref_dist.log_prob(sample_actions)

        return sample_actions, log_probs_policy, log_probs_ref
    
    def rollout_single_game(self, initial_board_state, first_move_idx, max_plies = 225, temperature = 1.0) -> float:
        board = initial_board_state.copy()
        
        # 初手（黒）が禁じ手（空盤面において天元以外）かどうかチェック
        # 空盤面を渡す
        empty_board = [0] * len(board)
        if is_forbidden_for_black(empty_board, first_move_idx):
            # 禁じ手を打った場合は、即座に黒の反則負け（ペナルティ報酬 -1.5）
            return -1.5, board

        board[first_move_idx] = 1 # プレイヤー1が石を置く
        
        winner = winner_after_move(board, first_move_idx, 1) # 終了判定
        if winner is not None:
            return (1.0 if winner == 1 else -1.0), board # 勝ちなら1、負けなら-1
        
        for ply in range(2, max_plies + 1):
            current_player = infer_player(board)

            # 簡易マスク：空きマス(0)の場所のみTrue
            legal_mask = torch.tensor([cell == 0 for cell in board], dtype=torch.bool, device=self.device)
            if not legal_mask.any():
                return 0.0, board # 打てる場所がない
            
            input_ids = self.tokenizer.encode_input(board).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.policy(input_ids).squeeze(0)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                probs = torch.softmax(masked_logits / temperature, dim = -1)
                dist = Categorical(probs = probs)
                move_idx = dist.sample().item()

            # 黒番（プレイヤー1）が石を置く直前に禁じ手かどうかチェック
            if current_player == 1:
                if is_forbidden_for_black(board, move_idx):
                    # 禁じ手を打った場合は即座に黒の反則負け（ペナルティ報酬 -1.5）
                    return -1.5, board

            board[move_idx] = current_player

            winner = winner_after_move(board, move_idx, current_player)
            if winner is not None:
                return (1.0 if winner == 1 else -1.0), board
            
        return 0.0, board

    def update_policy(self, loss):
        """Policyモデルの勾配更新をトリガーする"""
        # （Trainer から呼ばれて Policy の重みを更新する）
        pass



