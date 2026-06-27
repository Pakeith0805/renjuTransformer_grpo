import torch
import random
import gzip
import csv
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
import mlflow
from renju_transformer.rules import infer_player, winner_after_move, is_forbidden_for_black, board_with_move
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
def compute_group_advantages(rewards: list[float] | torch.Tensor, eps: float = 1e-8,
                             weights: torch.Tensor | None = None) -> torch.Tensor:
    if not isinstance(rewards, torch.Tensor):
        rewards = torch.tensor(rewards, dtype=torch.float32)

    # 群が1要素以下だと std が定義できない（top-K で合法手1つの局面など）
    if rewards.numel() < 2:
        return torch.zeros_like(rewards)

    std = rewards.std()
    if std < 1e-6:
        return torch.zeros_like(rewards)

    # ベースライン: 重みがあれば π 重み付き平均（期待値版GRPOの分散最小ベースライン）、無ければ単純平均
    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights.detach().to(rewards.device, dtype=rewards.dtype)
        baseline = (weights * rewards).sum()
    else:
        baseline = rewards.mean()

    advantages = (rewards - baseline) / (std + eps)

    return advantages


class GRPOTrainer:
    def __init__(self, agent, optimizer, cfg):
        self.agent = agent
        self.optimizer = optimizer
        self.cfg = cfg
        # value 共進化(v1.5): value_model を自己対戦の勝敗で継続学習し、凍結判定者の陳腐化を防ぐ。
        self.value_cotrain = bool(cfg.grpo.get("value_cotrain", False)) and (agent.value_model is not None)
        if self.value_cotrain:
            from collections import deque
            vlr = cfg.grpo.get("value_cotrain_lr", 1e-4)
            self.value_optimizer = torch.optim.AdamW(
                [p for p in agent.value_model.parameters() if p.requires_grad],
                lr=vlr, weight_decay=0.01,
            )
            self.value_buffer = deque(maxlen=int(cfg.grpo.get("value_buffer_size", 20000)))
            self.value_loss_fn = torch.nn.MSELoss()
            print(f"value co-train ON (lr={vlr}, buffer={self.value_buffer.maxlen})")

        # TSS(VCF)模倣の定点観測: 学習中の policy が探索なしでオラクル手をどれだけ当てるかを反復毎に追う。
        # 固定ケース集合を1回だけ作り反復間で比較可能にする(value_judge+TSS で模倣を狙う実験の成功指標)。
        self.tss_imit_eval = bool(cfg.grpo.get("tss_imitation_eval", False))
        self.tss_imit_cases = None  # 遅延構築(初回 eval 時に作る)
        if self.tss_imit_eval:
            print(f"TSS imitation eval ON (every {int(cfg.grpo.get('tss_imitation_eval_every', 50))} iters)")

    def _tss_imitation_eval(self, iteration):
        """現 policy の TSS(VCF)模倣率(探索なし masked-argmax がオラクル手と一致する率)を測り mlflow へ。"""
        import sys
        scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from test_tss_imitation import build_imitation_cases, score_imitation
        if self.tss_imit_cases is None:
            self.tss_imit_cases = build_imitation_cases(
                source=self.cfg.grpo.get("tss_imitation_source", "template"),
                per_category=int(self.cfg.grpo.get("tss_imitation_per_category", 100)),
                depth=int(self.cfg.grpo.get("tss_imitation_depth", 12)),
                seed=int(self.cfg.grpo.get("tss_imitation_seed", 0)),
            )
            print(f"TSS imitation eval: {len(self.tss_imit_cases)} 固定ケースを構築")
        overall, per_cat = score_imitation(
            self.agent.policy, self.agent.tokenizer, self.tss_imit_cases, self.agent.device)
        mlflow.log_metric("tss_imitation_top1", overall * 100, step=iteration)
        for cat, acc in per_cat.items():
            mlflow.log_metric(f"tss_imitation_{cat}", acc * 100, step=iteration)
        cats = "  ".join(f"{c}={a*100:.1f}%" for c, a in per_cat.items())
        print(f"[Iteration {iteration}] TSS imitation top1={overall*100:.1f}%  ({cats})")

    def train_step(self, board_state, beta: float = 0.04, clip_eps: float = 0.2):
        # 1回目のアクションと対数確率をとってくる
        vcf_target_prob = self.cfg.grpo.get("vcf_target_prob", 0.4)
        group_size = self.cfg.grpo.get("group_size", 8)
        dirichlet_alpha = self.cfg.grpo.get("dirichlet_alpha", 0.3)
        dirichlet_weight = self.cfg.grpo.get("dirichlet_weight", 0.25)
        use_topk = self.cfg.grpo.get("use_topk_weighted", False)
        # aux3/aux4 はモードで意味が変わる:
        #   sampling → (log_probs_old, log_probs_ref) / topk → (weights, None)
        actions, log_probs_policy, aux3, aux4, exact_kl, p_raw_val = self.agent.get_group_actions(
            board_state,
            group_size=group_size,
            temperature=self.cfg.grpo.temperature,
            vcf_target_prob=vcf_target_prob,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_weight=dirichlet_weight,
            use_topk_weighted=use_topk,
        )

        # 報酬を回収 (並列 MCTS 評価)
        move_indices = [actions[i].item() for i in range(len(actions))]
        # サンプリングされた手のユニーク数 (advantage消滅の原因切り分け用)
        n_unique_actions = len(set(move_indices))
        use_penalty = self.cfg.grpo.get("use_length_penalty", False)
        penalty_coef = self.cfg.grpo.get("length_penalty_coef", 0.02)
        if self.cfg.grpo.get("use_value_judge", False):
            # v1: rollout の代わりに value net で報酬(候補手を一括GPU評価+TSS上書き)
            rewards, last_final_board = self.agent.value_judge_rewards(board_state, move_indices)
        else:
            rewards, last_final_board = self.agent.rollout_group(
                board_state,
                move_indices,
                use_length_penalty=use_penalty,
                length_penalty_coef=penalty_coef
            )

        if use_topk:
            advantages = compute_group_advantages(rewards, weights=aux3)
        else:
            advantages = compute_group_advantages(rewards)

        # advantage 消滅の検知: グループ内 reward の std がほぼ 0 だと
        # advantage が全て 0 になり、policy_loss の勾配が消える（学習シグナルなし）
        rewards_std = torch.tensor(rewards, dtype=torch.float32).std()
        adv_collapsed = 1.0 if rewards_std < 1e-6 else 0.0

        advantages = advantages.to(self.agent.device)

        if use_topk:
            total_loss, policy_loss, kl_loss = self.compute_grpo_loss_weighted(
                log_probs_policy=log_probs_policy,        # 勾配あり
                weights=aux3.to(self.agent.device),       # w(a)=π(a) (detach)
                advantages=advantages,
                exact_kl=exact_kl,
                beta=beta,
            )
        else:
            total_loss, policy_loss, kl_loss = self.compute_grpo_loss(
                log_probs_policy=log_probs_policy,  # 勾配あり (Policyモデルを更新するため)
                log_probs_old=aux3,                 # 勾配なし (行動分布)
                log_probs_ref=aux4,                 # 勾配なし (Referenceモデル)
                advantages=advantages,
                exact_kl=exact_kl,
                beta=beta,
                clip_eps=clip_eps
            )

        self.optimizer.zero_grad()

        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(self.agent.policy.parameters(), max_norm=1.0)

        # advantage が消滅しているステップは policy 勾配がゼロなので optimizer.step をスキップ。
        # KL 勾配は残るため ref からの過剰なドリフトを防ぐ効果もある。
        skip_update = self.cfg.grpo.get("skip_collapsed_steps", True) and adv_collapsed == 1.0
        if not skip_update:
            self.optimizer.step()

        # Check if TSS is enabled for training rollouts and extract VCF paths
        new_trajectory_boards = []
        vcf_win_count = 0
        vcf_path_lengths = []
        if self.agent.use_tss_training:
            player = infer_player(board_state)
            for i, move_idx in enumerate(move_indices):
                # 盤面状態から即時勝利またはVCF勝ち手順があるかをチェック
                next_board = board_with_move(board_state, move_idx, player)
                winner = winner_after_move(next_board, move_idx, player)
                res = self.agent.get_vcf_winning_path_and_player(board_state, move_idx)
                
                if winner is not None or res is not None:
                    states = self.agent.get_vcf_path_states(board_state, move_idx)
                    if states:
                        new_trajectory_boards.extend(states)
                        vcf_win_count += 1
                        if res is not None:
                            _, _, path_moves = res
                            vcf_path_lengths.append(len(path_moves))

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = variance ** 0.5

        avg_vcf_path_length = float(sum(vcf_path_lengths)) / len(vcf_path_lengths) if vcf_path_lengths else 0.0
        new_vcf_injections = len(new_trajectory_boards)

        start_player = infer_player(board_state)
        black_reward = mean_reward if start_player == 1 else None
        white_reward = mean_reward if start_player == 2 else None

        return {
            "loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "exact_kl": exact_kl.item() if hasattr(exact_kl, "item") else exact_kl,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "adv_collapsed": adv_collapsed,
            "n_unique_actions": n_unique_actions,
            "grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
            "rewards": rewards,  # 個々の勝敗ログ
            "final_board": last_final_board,
            "new_trajectory_boards": new_trajectory_boards,
            "vcf_win_count": vcf_win_count,
            "avg_vcf_path_length": avg_vcf_path_length,
            "new_vcf_injections": new_vcf_injections,
            "black_reward": black_reward,
            "white_reward": white_reward,
            "vcf_p_raw": p_raw_val
        }
    
    def collect_trajectory_boards(self):
        """全局を MCTS 自己対戦で打ち、非終局局面のリストと勝者(1/2/None)を返す。
        勝者は value 共進化の教師に使う。決着しなければ None。"""
        board = [0] * 225
        board[112] = 1  # 1手目は天元に固定。ルールだから
        boards = [board.copy()]
        game_winner = None

        max_plies = self.cfg.grpo.get("max_plies", 80)
        # 2手目以降、ゲーム終了または最大手数まで打つ
        for ply in range(2, max_plies + 1):
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
            collection_sims = self.cfg.grpo.get("collection_simulations", 1000)
            move_idx = self.agent.select_move_via_mcts(board, simulations=collection_sims, temperature=temp, use_noise=True)

            board[move_idx] = current_player

            # 勝敗が決着した盤面はプールに追加しない (次の手番のプレイヤーが打てないため)
            winner = winner_after_move(board, move_idx, current_player)
            if winner is None:
                boards.append(board.copy())
            else:
                game_winner = winner  # current_player が勝った
                break

        return boards, game_winner

    def update_value_model(self):
        """value 共進化: バッファ(局面,勝敗±1)から数ステップ MSE 更新。平均lossを返す。"""
        if not self.value_cotrain or len(self.value_buffer) < 64:
            return None
        steps = int(self.cfg.grpo.get("value_cotrain_steps", 4))
        bs = int(self.cfg.grpo.get("value_cotrain_batch", 256))
        vm = self.agent.value_model
        tok = self.agent.tokenizer
        dev = self.agent.device
        vm.train()
        total = 0.0
        for _ in range(steps):
            batch = random.sample(self.value_buffer, min(bs, len(self.value_buffer)))
            input_ids = torch.stack([tok.encode_input(list(b)) for b, _ in batch]).to(dev)
            targets = torch.tensor([t for _, t in batch], dtype=torch.float32, device=dev)
            _, value = vm(input_ids, return_value=True)
            loss = self.value_loss_fn(value, targets)
            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()
            total += loss.item()
        vm.eval()   # 判定者として使うときは eval 固定
        return total / steps

    # 損失関数を返す関数
    def compute_grpo_loss(
            self,
            log_probs_policy: torch.Tensor,
            log_probs_old: torch.Tensor,
            log_probs_ref: torch.Tensor,
            advantages: torch.Tensor,
            exact_kl: torch.Tensor,
            beta: float = 0.04,
            clip_eps: float = 0.2
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratios = torch.exp(log_probs_policy - log_probs_old)

        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps , 1.0 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # exact_kl: 全合法手の分布全体で計算した正確な KL(policy || ref)（勾配あり）
        # per-sample 推定量は VCF injection 時に爆発するため使わない。
        # 勾配を保つことで policy を ref 方向へ引き戻す正則化が実際に効く。
        kl_loss = exact_kl
        total_loss = policy_loss + beta * kl_loss

        return total_loss, policy_loss, kl_loss

    def compute_grpo_loss_weighted(
            self,
            log_probs_policy: torch.Tensor,
            weights: torch.Tensor,
            advantages: torch.Tensor,
            exact_kl: torch.Tensor,
            beta: float = 0.04
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 期待値版 GRPO: L = - Σ_a w(a) A(a) log π(a)
        #   選択はノイズ入り分布の top-K、重み w(a)=π(a) と advantage A(a) は detached、log π に勾配。
        #   決定論的列挙なので importance sampling / PPO clip は使わない（ratio=1相当）。
        #   勾配 = -Σ w A ∇log π = 期待advantage勾配（top-K集合上）。
        policy_loss = -(weights * advantages * log_probs_policy).sum()
        kl_loss = exact_kl
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
                if self.cfg.grpo.get("use_full_game_training", False):
                    # 1戦通して盤面を収集し、全盤面を1回ずつ学習するモード
                    game_boards, game_winner = self.collect_trajectory_boards()
                    if not game_boards:
                        game_boards = [initial_board.copy()]

                    # value 共進化(v1.5): 全局の勝敗を教師としてバッファに蓄積し value を更新
                    value_cotrain_loss = None
                    if self.value_cotrain and game_winner is not None:
                        for b in game_boards:
                            tgt = 1.0 if infer_player(b) == game_winner else -1.0
                            self.value_buffer.append((tuple(b), tgt))
                    if self.value_cotrain and iteration % int(self.cfg.grpo.get("value_cotrain_every", 1)) == 0:
                        value_cotrain_loss = self.update_value_model()
                        if value_cotrain_loss is not None:
                            mlflow.log_metric("grpo_value_cotrain_loss", value_cotrain_loss, step=iteration)

                    step_metrics_list = []
                    for board_state in game_boards:
                        step_metrics = self.train_step(
                            board_state,
                            beta=self.cfg.grpo.beta,
                            clip_eps=self.cfg.grpo.clip_eps
                        )
                        step_metrics_list.append(step_metrics)

                    total_steps = len(step_metrics_list)
                    avg_loss = sum(m["loss"] for m in step_metrics_list) / total_steps
                    avg_policy_loss = sum(m["policy_loss"] for m in step_metrics_list) / total_steps
                    avg_kl_loss = sum(m["kl_loss"] for m in step_metrics_list) / total_steps
                    avg_exact_kl = sum(m["exact_kl"] for m in step_metrics_list) / total_steps
                    avg_mean_reward = sum(m["mean_reward"] for m in step_metrics_list) / total_steps
                    avg_std_reward = sum(m["std_reward"] for m in step_metrics_list) / total_steps
                    avg_grad_norm = sum(m["grad_norm"] for m in step_metrics_list) / total_steps
                    avg_adv_collapsed = sum(m["adv_collapsed"] for m in step_metrics_list) / total_steps
                    # 全ステップ平均のユニーク手数 (ノイズが多様性を生んでいるかの確認)
                    avg_unique_actions = sum(m["n_unique_actions"] for m in step_metrics_list) / total_steps
                    # 消滅ステップに限ったユニーク手数 (原因A=少数 / 原因B=多数 の切り分け)
                    collapsed_steps = [m for m in step_metrics_list if m["adv_collapsed"] == 1.0]
                    unique_on_collapse = (
                        sum(m["n_unique_actions"] for m in collapsed_steps) / len(collapsed_steps)
                        if collapsed_steps else 0.0
                    )
                    sum_vcf_win_count = sum(m["vcf_win_count"] for m in step_metrics_list)
                    sum_new_vcf_injections = sum(m["new_vcf_injections"] for m in step_metrics_list)

                    vcf_steps_lengths = [m["avg_vcf_path_length"] for m in step_metrics_list if m["vcf_win_count"] > 0]
                    avg_vcf_path_length = sum(vcf_steps_lengths) / len(vcf_steps_lengths) if vcf_steps_lengths else 0.0

                    black_rewards = [m["black_reward"] for m in step_metrics_list if m["black_reward"] is not None]
                    white_rewards = [m["white_reward"] for m in step_metrics_list if m["white_reward"] is not None]
                    avg_black_reward = sum(black_rewards) / len(black_rewards) if black_rewards else None
                    avg_white_reward = sum(white_rewards) / len(white_rewards) if white_rewards else None

                    vcf_p_raw_list = [m["vcf_p_raw"] for m in step_metrics_list if m["vcf_p_raw"] is not None]
                    avg_vcf_p_raw = sum(vcf_p_raw_list) / len(vcf_p_raw_list) if vcf_p_raw_list else None

                    metrics = {
                        "loss": avg_loss,
                        "policy_loss": avg_policy_loss,
                        "kl_loss": avg_kl_loss,
                        "exact_kl": avg_exact_kl,
                        "mean_reward": avg_mean_reward,
                        "std_reward": avg_std_reward,
                        "grad_norm": avg_grad_norm,
                        "adv_collapse_rate": avg_adv_collapsed,
                        "mean_unique_actions": avg_unique_actions,
                        "unique_on_collapse": unique_on_collapse,
                        "has_collapse": 1.0 if collapsed_steps else 0.0,
                        "vcf_win_count": sum_vcf_win_count,
                        "avg_vcf_path_length": avg_vcf_path_length,
                        "new_vcf_injections": sum_new_vcf_injections,
                        "black_reward": avg_black_reward,
                        "white_reward": avg_white_reward,
                        "vcf_p_raw": avg_vcf_p_raw,
                        "final_board": step_metrics_list[-1]["final_board"]
                    }
                else:
                    start_board = initial_board
 
                    if not trajectory_boards or random.random() > sample_prob:
                        start_board = initial_board
 
                    if iteration % 100 == 0 or not trajectory_boards:
                        new_boards, _ = self.collect_trajectory_boards()
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
                
                # メトリクスを MLflow に記録
                mlflow.log_metric("grpo_loss", metrics["loss"], step=iteration)
                mlflow.log_metric("grpo_policy_loss", metrics["policy_loss"], step=iteration)
                mlflow.log_metric("grpo_kl", metrics["kl_loss"], step=iteration)
                mlflow.log_metric("grpo_exact_kl", metrics["exact_kl"], step=iteration)
                mlflow.log_metric("grpo_mean_reward", metrics["mean_reward"], step=iteration)
                mlflow.log_metric("grpo_std_reward", metrics["std_reward"], step=iteration)
                mlflow.log_metric("grpo_grad_norm", metrics["grad_norm"], step=iteration)
                mlflow.log_metric("grpo_adv_collapse_rate", metrics.get("adv_collapse_rate", metrics.get("adv_collapsed", 0.0)), step=iteration)
                # ユニーク手数の診断 (advantage消滅の原因A/B切り分け)
                mlflow.log_metric("grpo_mean_unique_actions", metrics.get("mean_unique_actions", metrics.get("n_unique_actions", 0.0)), step=iteration)
                # 消滅時のユニーク手数: 消滅が起きたイテレーションのみ記録 (0埋めで平均を歪めない)
                if "unique_on_collapse" in metrics:
                    if metrics.get("has_collapse", 0.0) == 1.0:
                        mlflow.log_metric("grpo_unique_on_collapse", metrics["unique_on_collapse"], step=iteration)
                elif metrics.get("adv_collapsed", 0.0) == 1.0:
                    mlflow.log_metric("grpo_unique_on_collapse", metrics.get("n_unique_actions", 0.0), step=iteration)
                mlflow.log_metric("grpo_vcf_win_count", metrics["vcf_win_count"], step=iteration)
                mlflow.log_metric("grpo_avg_vcf_path_length", metrics["avg_vcf_path_length"], step=iteration)
                mlflow.log_metric("grpo_new_vcf_injections", metrics["new_vcf_injections"], step=iteration)
                
                if metrics["black_reward"] is not None:
                    mlflow.log_metric("grpo_black_reward", metrics["black_reward"], step=iteration)
                if metrics["white_reward"] is not None:
                    mlflow.log_metric("grpo_white_reward", metrics["white_reward"], step=iteration)
                if "vcf_p_raw" in metrics and metrics["vcf_p_raw"] is not None:
                    mlflow.log_metric("grpo_vcf_p_raw", metrics["vcf_p_raw"], step=iteration)
                
                # 画面の進捗バーの表示を更新
                vcf_p_raw_val = metrics.get("vcf_p_raw")
                progress.set_postfix(
                    loss=f"{metrics['loss']:.4f}",
                    kl=f"{metrics['kl_loss']:.4f}",
                    reward=f"{metrics['mean_reward']:+.2f}",
                    vcf=metrics["vcf_win_count"],
                    p_raw=f"{vcf_p_raw_val:.3f}" if vcf_p_raw_val is not None else "N/A"
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

                # 100エポックに一回、あるいは開始エポックに、事前学習済みモデルとの評価対局を実行
                if iteration == start_iteration or iteration % 100 == 0:
                    # temperature=1.0 での対局 (100ゲーム)
                    print(f"\n[Iteration {iteration}] Running versus evaluation against pretrained model (100 games, temperature=1.0)...")
                    eval_results = self.evaluate_versus(num_games=100, temperature=1.0)
                    print(f"  Policy Win Rate (temp=1.0): {eval_results['policy_win_rate']:.1f}%")
                    print(f"  Reference Win Rate (temp=1.0): {eval_results['ref_win_rate']:.1f}%")
                    print(f"  Draw Rate (temp=1.0): {eval_results['draw_rate']:.1f}%")
                    print(f"  Average Game Length (temp=1.0): {eval_results['avg_plies']:.1f} plies")
                    
                    # MLflow に記録
                    mlflow.log_metric("versus_policy_win_rate", eval_results["policy_win_rate"], step=iteration)
                    mlflow.log_metric("versus_ref_win_rate", eval_results["ref_win_rate"], step=iteration)
                    mlflow.log_metric("versus_draw_rate", eval_results["draw_rate"], step=iteration)
                    mlflow.log_metric("versus_avg_plies", eval_results["avg_plies"], step=iteration)

                    # temperature=0.0 での対局 (2ゲーム)
                    print(f"\n[Iteration {iteration}] Running versus evaluation against pretrained model (2 games, temperature=0.0)...")
                    eval_results_temp0 = self.evaluate_versus(num_games=2, temperature=0.0)
                    print(f"  Policy Win Rate (temp=0.0): {eval_results_temp0['policy_win_rate']:.1f}%")
                    print(f"  Reference Win Rate (temp=0.0): {eval_results_temp0['ref_win_rate']:.1f}%")
                    print(f"  Draw Rate (temp=0.0): {eval_results_temp0['draw_rate']:.1f}%")
                    print(f"  Average Game Length (temp=0.0): {eval_results_temp0['avg_plies']:.1f} plies")
                    
                    # MLflow に記録
                    mlflow.log_metric("versus_policy_win_rate_temp0", eval_results_temp0["policy_win_rate"], step=iteration)
                    mlflow.log_metric("versus_ref_win_rate_temp0", eval_results_temp0["ref_win_rate"], step=iteration)
                    mlflow.log_metric("versus_draw_rate_temp0", eval_results_temp0["draw_rate"], step=iteration)
                    mlflow.log_metric("versus_avg_plies_temp0", eval_results_temp0["avg_plies"], step=iteration)

                # TSS(VCF)模倣率の定点観測(有効時のみ。これが value_judge+TSS 実験の成功指標)
                if self.tss_imit_eval and (
                    iteration == start_iteration
                    or iteration % int(self.cfg.grpo.get("tss_imitation_eval_every", 50)) == 0
                ):
                    self._tss_imitation_eval(iteration)

    def evaluate_versus(self, num_games=10, temperature=1.0) -> dict:
        """現在のポリシーと事前学習済み（Reference）モデルとの対局シミュレーションを行い、勝率を測定します"""
        self.agent.policy.eval()
        self.agent.ref.eval()
        
        stats = {
            "policy_wins": 0,
            "ref_wins": 0,
            "draws": 0,
            "total_plies": 0,
        }
        
        device = self.agent.device
        tokenizer = self.agent.tokenizer
        
        for game_idx in range(1, num_games + 1):
            is_policy_black = (game_idx % 2 == 1)
            board = [0] * 225
            winner = None
            plies = 0
            
            for ply in range(1, 226):
                current_player = infer_player(board)
                current_is_policy = (current_player == 1) if is_policy_black else (current_player == 2)
                current_model = self.agent.policy if current_is_policy else self.agent.ref
                
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
                        dist = Categorical(probs=probs)
                        move_idx = dist.sample().item()
                
                board[move_idx] = current_player
                plies += 1
                
                winner = winner_after_move(board, move_idx, current_player)
                if winner is not None:
                    break
            
            stats["total_plies"] += plies
            if winner is None:
                stats["draws"] += 1
            else:
                winner_is_policy = (winner == 1) if is_policy_black else (winner == 2)
                if winner_is_policy:
                    stats["policy_wins"] += 1
                else:
                    stats["ref_wins"] += 1
                    
        policy_win_rate = (stats["policy_wins"] / num_games) * 100.0
        ref_win_rate = (stats["ref_wins"] / num_games) * 100.0
        draw_rate = (stats["draws"] / num_games) * 100.0
        
        # Reset policy model to train mode
        self.agent.policy.train()
        
        return {
            "policy_win_rate": policy_win_rate,
            "ref_win_rate": ref_win_rate,
            "draw_rate": draw_rate,
            "avg_plies": stats["total_plies"] / num_games
        }