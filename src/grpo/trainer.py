import torch

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
        actions, log_probs_policy, log_probs_ref = self.agent.get_group_actions(board_state, group_size=8)

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

    # 損失関数を返す関数
    def compute_grpo_loss(
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