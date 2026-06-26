#!/usr/bin/env python
"""
value ヘッドの教師あり pretrain
================================
data.csv.gz（局面→次の手のペアが対局順に並ぶ）を対局に区切り、各対局の勝者を辿って
全局面に「手番側視点の勝敗ラベル(+1/-1)」を貼り、policy 胴体に付けた value ヘッドを回帰学習する。

  - 行フォーマット: [225 盤面(0/1/2)] + [228 sep] + [move_id]   (盤index = move_id - 3)
  - 対局境界: 空盤(石0個)の行で新しい対局が始まる
  - 勝者: 各対局の最後の手が五を作る → その手番が勝者。作らない(途中終了)対局はスキップ
  - value 教師: 局面の手番(infer_player) == 勝者 なら +1、違えば -1

使い方:
  uv run python scripts/pretrain_value.py \
      --pretrained models/pretrained.pt \
      --data data.csv.gz \
      --out models/pretrained_value.pt \
      --epochs 6 --batch-size 512 --device auto
  オプション: --freeze-body (胴体凍結, value ヘッドだけ学習) / --max-games N (動作確認用)
"""
from __future__ import annotations
import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from renju_transformer.model import RenjuTransformerModel
from renju_transformer.rules import infer_player, winner_after_move, board_with_move
from renju_transformer.utils import select_device, set_seed

SEP = 228
MOVE_OFFSET = 3
N = 225


# --------------------------------------------------------------------------- #
# データ: 対局に区切って勝敗ラベルを作る
# --------------------------------------------------------------------------- #
def build_value_dataset(data_path, max_games=None):
    """戻り値: boards (uint8 [M,225]), values (float32 [M])。M=採用局面数。"""
    boards_out = []
    values_out = []
    games = won = skipped = 0

    cur_boards = []      # この対局の (board, move_id) 列
    cur_moves = []

    def flush_game():
        nonlocal games, won, skipped
        if not cur_boards:
            return
        games += 1
        last_board = cur_boards[-1]
        last_move_id = cur_moves[-1]
        idx = last_move_id - MOVE_OFFSET
        if not (0 <= idx < N):
            skipped += 1
            return
        player = infer_player(last_board)
        nb = board_with_move(last_board, idx, player)
        if winner_after_move(nb, idx, player) != player:
            skipped += 1   # 最後の手が五を作らない=途中終了/引分 → 勝者不明なので捨てる
            return
        winner = player
        won += 1
        for b in cur_boards:
            v = 1.0 if infer_player(b) == winner else -1.0
            boards_out.append(np.asarray(b, dtype=np.uint8))
            values_out.append(v)

    with gzip.open(data_path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            board = [int(x) for x in vals[:N]]
            move_id = int(vals[-1])
            if sum(1 for c in board if c != 0) == 0:   # 空盤 = 新しい対局
                flush_game()
                cur_boards, cur_moves = [], []
                if max_games is not None and games >= max_games:
                    break
            cur_boards.append(board)
            cur_moves.append(move_id)
    flush_game()

    if not boards_out:
        raise RuntimeError("採用局面が0。データ形式を確認してください。")
    boards = np.stack(boards_out).astype(np.uint8)
    values = np.asarray(values_out, dtype=np.float32)
    print(f"対局={games} 勝敗確定={won} スキップ={skipped} / 採用局面={len(values)}", file=sys.stderr)
    return boards, values


class ValueDataset(Dataset):
    def __init__(self, boards, values):
        self.boards = boards      # uint8 [M,225]
        self.values = values      # float32 [M]
        self._sep = torch.full((1,), SEP, dtype=torch.long)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        b = torch.from_numpy(self.boards[i].astype(np.int64))
        input_ids = torch.cat([b, self._sep])   # [226]
        return input_ids, torch.tensor(self.values[i], dtype=torch.float32)


# --------------------------------------------------------------------------- #
# モデル
# --------------------------------------------------------------------------- #
def build_model_with_value(pretrained_path, device):
    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    mc = ckpt["config"]["model"]
    model = RenjuTransformerModel(
        vocab_size=mc["token_vocab_size"], max_seq_len=mc["max_seq_len"], d_model=mc["d_model"],
        nhead=mc["nhead"], num_layers=mc["num_layers"], dim_feedforward=mc["dim_feedforward"],
        dropout=mc["dropout"], activation=mc["activation"], norm_first=mc["norm_first"],
        num_move_labels=mc["num_move_labels"], with_value_head=True,
    )
    # policy 重みをロード (value_head は新規なので strict=False で OK)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"loaded policy weights. missing(=新規value等):{missing} unexpected:{unexpected}", file=sys.stderr)
    model.to(device)
    return model, ckpt["config"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained", default="models/pretrained.pt", help="胴体を初期化する policy checkpoint")
    ap.add_argument("--data", default="data.csv.gz")
    ap.add_argument("--out", default="models/pretrained_value.pt")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freeze-body", action="store_true", help="胴体を凍結し value ヘッドだけ学習")
    ap.add_argument("--val-frac", type=float, default=0.02, help="検証に回す割合")
    ap.add_argument("--max-games", type=int, default=None, help="動作確認用に対局数を制限")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)
    print(f"device={device}", file=sys.stderr)

    t0 = time.time()
    boards, values = build_value_dataset(args.data, max_games=args.max_games)
    print(f"データ構築 {time.time()-t0:.1f}s", file=sys.stderr)

    # train/val split
    n = len(values)
    perm = np.random.RandomState(args.seed).permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    tr = ValueDataset(boards[tr_idx], values[tr_idx])
    va = ValueDataset(boards[val_idx], values[val_idx])
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model, config = build_model_with_value(args.pretrained, device)

    if args.freeze_body:
        for name, p in model.named_parameters():
            p.requires_grad = name.startswith("value_head.")
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        te = time.time()
        run = 0.0
        nb = 0
        for input_ids, target in tr_loader:
            input_ids = input_ids.to(device)
            target = target.to(device)
            _, value = model(input_ids, return_value=True)
            loss = loss_fn(value, target)
            opt.zero_grad()
            loss.backward()
            opt.step()
            run += loss.item()
            nb += 1
        # val (符号一致率 = どれだけ勝敗の向きを当てたか)
        model.eval()
        with torch.no_grad():
            agree = tot = vloss = 0
            vbatches = 0
            for input_ids, target in va_loader:
                input_ids = input_ids.to(device)
                target = target.to(device)
                _, value = model(input_ids, return_value=True)
                vloss += loss_fn(value, target).item()
                vbatches += 1
                agree += (torch.sign(value) == torch.sign(target)).sum().item()
                tot += target.numel()
        print(f"epoch {epoch}/{args.epochs}  train_mse={run/max(nb,1):.4f}  "
              f"val_mse={vloss/max(vbatches,1):.4f}  val_sign_acc={agree/max(tot,1)*100:.1f}%  "
              f"({time.time()-te:.1f}s)", file=sys.stderr)

    # 保存 (config に value ヘッド有を明記)
    config = dict(config)
    config["model"] = dict(config["model"])
    config["model"]["with_value_head"] = True
    out = {"model_state_dict": model.state_dict(), "config": config, "value_pretrain": True}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"saved -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
