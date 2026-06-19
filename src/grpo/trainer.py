import torch
import random
import gzip
import csv
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import mlflow
from renju_transformer.rules import infer_player, winner_after_move, is_forbidden_for_black
from torch.distributions import Categorical
from grpo.load_model import print_board

def load_initial_trajectory_boards(csv_gz_path: Path, num_samples: int = 300) -> list[list[int]]:
    boards = []
    if not csv_gz_path.exists():
        print(f"Warning: {csv_gz_path} not found. Starting with empty trajectory pool.")
        return boards
    print(f"Pre-loading {num_samples} initial boards from {csv_gz_path}...")
    try:
        with gzip.open(csv_gz_path, mode="rt", encoding="utf-8") as f:
            reader = csv.reader(f)
            all_lines = []
            for i, row in enumerate(reader):
                if not row or len(row) < 225:
                    continue
                try:
                    # 最初の225列（盤面）を数値のリストとして抽出
                    board = [int(cell) for cell in row[:225]]
                    all_lines.append(board)
                except ValueError:
                    continue
                if i > 20000:  # メモリと速度の観点から最初の2万行でサンプリングを打ち切る
                    break
            
            if all_lines:
                boards = random.sample(all_lines, min(num_samples, len(all_lines)))
                print(f"Successfully loaded {len(boards)} boards.")
    except Exception as e:
        print(f"Warning: Failed to load initial boards: {e}. Starting with empty trajectory pool.")
        boards = []
    return boards


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

    def train_step(self, board_state, beta: float = 0.04, clip_eps: float = 0.2):
        # 1回目のアクションと対数確率をとってくる
        actions, log_probs_policy, log_probs_ref = self.agent.get_group_actions(
            board_state, 
            group_size=8, 
            temperature=self.cfg.grpo.temperature
        )

        # 報酬を回収 (並列 MCTS 評価)
        move_indices = [actions[i].item() for i in range(len(actions))]
        rewards, last_final_board = self.agent.rollout_group(board_state, move_indices)

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

        # Check if TSS is enabled for training rollouts and extract VCF paths
        new_trajectory_boards = []
        if self.agent.use_tss_training:
            for i, move_idx in enumerate(move_indices):
                if abs(rewards[i] - 1.0) < 1e-5 or abs(rewards[i] + 1.0) < 1e-5:
                    states = self.agent.get_vcf_path_states(board_state, move_idx)
                    new_trajectory_boards.extend(states)

        mean_reward = sum(rewards) / len(rewards)

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": mean_reward,
            "rewards": rewards,  # 個々の勝敗ログ
            "final_board": last_final_board,
            "new_trajectory_boards": new_trajectory_boards
        }
    
    def collect_trajectory_boards(self) -> list[list[int]]:
        board = [0] * 225
        board[112] = 1  # 1手目は天元に固定。ルールだから
        boards = [board.copy()]

        # 2手目以降、ゲーム終了まで打つ
        for ply in range(2, 226):
            current_player = infer_player(board)
            legal_mask = self.agent.tokenizer.legal_move_mask(board).to(self.agent.device)
            if not legal_mask.any():
                break

            # 【変更】従来のモデル直感からのワンショットサンプリングはコメントアウトにします。
            # 理由: より高品質でバグのない学習対局データを生成するため、
            #      モデルにガイドされたMCTS探索（PUCT, シミュレーション回数1000回）によって指し手を決定するように移行。
            # input_ids = self.agent.tokenizer.encode_input(board).unsqueeze(0).to(self.agent.device)
            # with torch.no_grad():
            #     logits = self.agent.policy(input_ids).squeeze(0)
            #     masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
            #     probs = torch.softmax(masked_logits / self.cfg.grpo.temperature, dim = -1)
            #     dist = Categorical(probs=probs)
            #     move_idx = dist.sample().item()

            # 序盤（最初の10手まで）は T=1.0、それ以降は T=0.1 で探索
            temp = 1.0 if len(boards) <= 10 else 0.1
            move_idx = self.agent.select_move_via_mcts(board, simulations=1000, temperature=temp, use_noise=True)

            board[move_idx] = current_player

            # 勝敗が決着した盤面はプールに追加しない (次の手番のプレイヤーが打てないため)
            winner = winner_after_move(board, move_idx, current_player)
            if winner is None:
                boards.append(board.copy())
            else:
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
    
    def train(self, num_iterations: int = 1000, save_every: int = 50, run_id: str = None, start_iteration: int = 1):
        """
        強化学習（GRPO）のメインループを実行します (初期盤面のみのシンプルテスト版)。
        """
        print(f"Starting simple GRPO training for {num_iterations} iterations...")
        
        initial_board = [0] * 225
        trajectory_boards = []
        
        # チェックポイントからの途中再開時（start_iteration > 1）は、プール局面の復元を試みる
        if start_iteration > 1:
            try:
                checkpoint_path = Path(self.cfg.grpo.checkpoint_path)
                if checkpoint_path.exists():
                    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                    if "trajectory_boards" in checkpoint:
                        # 合法手が存在する有効な盤面のみを復元する (クラッシュ防止)
                        trajectory_boards = [
                            b for b in checkpoint["trajectory_boards"]
                            if self.agent.tokenizer.legal_move_mask(b).any()
                        ]
                        print(f"Restored {len(trajectory_boards)} trajectory boards from checkpoint.")
            except Exception as e:
                print(f"Warning: Failed to restore trajectory_boards from checkpoint: {e}")

        # プールが空の場合（新規開始、またはチェックポイントに無かった場合）のみ、data.csv.gz からコールドスタート局面をロード (1000局)
        if not trajectory_boards:
            csv_gz_path = Path("data.csv.gz").absolute()
            trajectory_boards = load_initial_trajectory_boards(csv_gz_path, num_samples=1000)

        sample_prob = self.cfg.grpo.get("trajectory_sample_prob", 0.8)
        
        # MLflow の実験コンテキストを開始
        with mlflow.start_run(run_id=run_id, run_name=self.cfg.mlflow.run_name_prefix + "-grpo", nested=True):
            
            progress = tqdm(range(start_iteration, start_iteration + num_iterations), desc="GRPO Iterations")
            for iteration in progress:
                start_board = initial_board
 
                if not trajectory_boards or random.random() > sample_prob:
                    start_board = initial_board
 
                if iteration % 100 == 0 or not trajectory_boards:
                    new_boards = self.collect_trajectory_boards()
                    trajectory_boards.extend(new_boards)
 
                    if len(trajectory_boards) > 1000:
                        trajectory_boards = trajectory_boards[-1000:]
                else:
                    # 盤面をランダムに選ぶ
                    start_board = random.choice(trajectory_boards)

                # 選ばれた局面から1訓練
                # 初期盤面から 8 通り試して Policy を更新 (1回の学習ステップ)
                metrics = self.train_step(
                    start_board, 
                    beta=self.cfg.grpo.beta, 
                    clip_eps=self.cfg.grpo.clip_eps
                )
                
                # Check for new VCF boards and add them to trajectory pool
                if "new_trajectory_boards" in metrics and metrics["new_trajectory_boards"]:
                    valid_new_boards = [
                        b for b in metrics["new_trajectory_boards"]
                        if self.agent.tokenizer.legal_move_mask(b).any()
                    ]
                    if valid_new_boards:
                        trajectory_boards.extend(valid_new_boards)
                        if len(trajectory_boards) > 1000:
                            trajectory_boards = trajectory_boards[-1000:]
                
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
                
                # 1イテレーション（エポック）終了ごとに、自己対戦の最終盤面を表示
                print(f"\n[Iteration {iteration}] Self-Play Sample Final Board:")
                print_board(metrics["final_board"])
                
                # 定期的にモデルのチェックポイントを保存
                if iteration % save_every == 0:
                    checkpoint_path = Path(self.cfg.train.output_root) / f"grpo_checkpoint_{iteration}.pt"
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        "model_state_dict": self.agent.policy.state_dict(),
                        "config": OmegaConf.to_container(self.cfg, resolve=True),
                        "iteration": iteration,
                        "trajectory_boards": trajectory_boards  # 局面プールも保存して途中再開できるようにする
                    }, checkpoint_path)
                    print(f"\nSaved checkpoint to {checkpoint_path}")