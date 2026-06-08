import torch
import random
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import mlflow
from renju_transformer.rules import infer_player, winner_after_move, is_forbidden_for_black
from torch.distributions import Categorical

# 報酬を受け取ってアドバンテージを返す関数
def compute_group_advantages(rewards: list[float] | torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)

    mean = rewards.mean()
    std = rewards.std()

    advantages = (rewards - mean) / (std + eps)

    return advantages



class GRPOTrainer:
    def __init__(self, agent, optimizer, cfg):
        self.agent = agent
        self.optimizer = optimizer
        self.cfg = cfg

    # 
    def train_step(self, board_state, beta: float = 0.04, clip_eps: float = 0.2):
        # 1回目のアクションと対数確率をとってくる
        actions, log_probs_policy, log_probs_ref = self.agent.get_group_actions(
            board_state, 
            group_size=8, 
            temperature=self.cfg.grpo.temperature
        )

        # 報酬を回収
        rewards = []

        for i in range(8):
            reward, _ = self.agent.rollout_single_game(board_state, actions[i].item())
            rewards.append(reward)

        advantages = compute_group_advantages(rewards)

        advantages = advantages.to(self.agent.device)

        log_probs_old = log_probs_policy.detach()

        total_loss, policy_loss, kl_loss = self.compute_grpo_loss(
            log_probs_policy=log_probs_policy,  # 勾配あり (Policyモデルを更新するため)
            log_probs_old=log_probs_old,        # 勾配なし (基準値)
            log_probs_ref=log_probs_ref,        # 勾配なし (Referenceモデル)
            advantages=advantages,
            beta=beta,
            clip_eps=clip_eps
        )

        self.optimizer.zero_grad()

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_norm=1.0)

        self.optimizer.step()

        mean_reward = sum(rewards) / len(rewards)

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": mean_reward,
            "rewards": rewards  # 個々の勝敗ログ
        }
    
    def collect_trajectory_boards(self) -> list[list[int]]:
        board = [0] * 225
        boards = [board.copy()]

        input_ids = self.agent.tokenizer.encode_input(board).unsqueeze(0).to(self.agent.device)
        legal_mask = torch.tensor([cell == 0 for cell in board], dtype = torch.bool, device = self.agent.device)

        with torch.no_grad():
            logits = self.agent.policy(input_ids).squeeze(0)
            masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            probs = torch.softmax(masked_logits / self.cfg.grpo.temperature, dim=-1)
            dist = Categorical(probs=probs)
            move_idx = dist.sample().item()
        
        empty_board = [0] * len(board)
        if is_forbidden_for_black(empty_board, move_idx):
            return boards # 1手目で終わり
        
        board[move_idx] = 1
        boards.append(board.copy())

        # 2手目以降、ゲーム終了まで打つ
        for ply in range(2, 226):
            current_player = infer_player(board)
            legal_mask = torch.tensor([cell == 0 for cell in board], dtype = torch.bool, device=self.agent.device)
            if not legal_mask.any():
                break

            input_ids = self.agent.tokenizer.encode_input(board).unsqueeze(0).to(self.agent.device)

            with torch.no_grad():
                logits = self.agent.policy(input_ids).squeeze(0)
                masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
                probs = torch.softmax(masked_logits / self.cfg.grpo.temperature, dim = -1)
                dist = Categorical(probs=probs)
                move_idx = dist.sample().item()

            if current_player == 1:
                if is_forbidden_for_black(board, move_idx):
                    break

            board[move_idx] = current_player
            boards.append(board.copy())

            winner = winner_after_move(board, move_idx, current_player)
            if winner is not None:
                break

        return boards



    # 損失関数を返す関数
    def compute_grpo_loss(
            self,
            log_probs_policy: torch.Tensor,
            log_probs_old: torch.Tensor,
            log_probs_ref: torch.Tensor,
            advantages: torch.Tensor,
            beta: float = 0.04,
            clip_eps: float = 0.2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # eの肩に対数を載せた
        ratios = torch.exp(log_probs_policy - log_probs_old)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps , 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 以下、klダイバージェンス
        kl_ratio = torch.exp(log_probs_ref - log_probs_policy)
        kl_diff = log_probs_ref - log_probs_policy
        kl_div = kl_ratio - kl_diff - 1.0
        kl_loss = kl_div.mean()

        # 総損失
        total_loss = policy_loss + beta * kl_loss

        return total_loss, policy_loss, kl_loss
    
    # src/grpo/trainer.py 内の GRPOTrainer クラスに追記

    def train(self, num_iterations: int = 1000, save_every: int = 50):
        """
        強化学習（GRPO）のメインループを実行します (初期盤面のみのシンプルテスト版)。
        """
        print(f"Starting simple GRPO training for {num_iterations} iterations...")
        
        # 初期盤面 (225マスの空盤面) を作成
        initial_board = [0] * 225
        trajectory_boards = []

        sample_prob = self.cfg.grpo.get("trajectory_sample_prob, 0.8")
        
        # MLflow の実験コンテキストを開始
        with mlflow.start_run(run_name=self.cfg.mlflow.run_name_prefix + "-grpo", nested=True):
            
            progress = tqdm(range(1, num_iterations + 1), desc="GRPO Iterations")
            for iteration in progress:

                if not trajectory_boards or random.random() > sample_prob:
                    start_board = initial_board

                    if iteration % 5 == 0 or not trajectory_boards:
                        new_boards = self.collect_trajectory_boards()
                        trajectory_boards.extend(new_boards)

                        if len(trajectory_boards) > 300:
                            trajectory_boards = trajectory_boards[-300:]
                    else:
                        # 盤面をランダムに選ぶ
                        start_board = random.choice(trajectory_boards)

                    # 選ばれた局面から1訓練
                    # 初期盤面から 8 通り試して Policy を更新 (1回の学習ステップ)
                    metrics = self.train_step(
                        initial_board, 
                        beta=self.cfg.grpo.beta, 
                        clip_eps=self.cfg.grpo.clip_eps
                    )
                
                    # メトリクス（Loss、KL、平均報酬）を MLflow に記録
                    mlflow.log_metric("grpo_loss", metrics["loss"], step=iteration)
                    mlflow.log_metric("grpo_kl", metrics["kl_loss"], step=iteration)
                    mlflow.log_metric("grpo_mean_reward", metrics["mean_reward"], step=iteration)
                
                    # 画面の進捗バーの表示を更新
                    progress.set_postfix(
                        loss=f"{metrics['loss']:.4f}",
                        kl=f"{metrics['kl_loss']:.4f}",
                        reward=f"{metrics['mean_reward']:+.2f}"
                    )
                
                    # 定期的にモデルのチェックポイントを保存
                    if iteration % save_every == 0:
                        checkpoint_path = Path(self.cfg.train.output_root) / f"grpo_checkpoint_{iteration}.pt"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            "model_state_dict": self.agent.policy.state_dict(),
                            "config": OmegaConf.to_container(self.cfg, resolve=True),
                            "iteration": iteration
                        }, checkpoint_path)
                        print(f"\nSaved checkpoint to {checkpoint_path}")