"""TransformerEncoder model for next-move prediction."""

from __future__ import annotations

import torch
from torch import nn


class RenjuTransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        norm_first: bool,
        num_move_labels: int,
        with_value_head: bool = False,
        value_head_layers: int = 2,
        value_head_hidden: int = 0,
    ) -> None:
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_move_labels)
        # value ヘッドは任意 (デフォルト無効)。無効時は既存 checkpoint/呼び出しと完全互換。
        # 層数(value_head_layers)と幅(value_head_hidden, 0でd_model)を可変にして容量を調整できる。
        # value_head_layers=2, hidden=d_model がデフォルトで従来と同一構成。
        self.with_value_head = with_value_head
        if with_value_head:
            hidden = value_head_hidden if value_head_hidden > 0 else d_model
            layers: list[nn.Module] = []
            in_dim = d_model
            for _ in range(max(1, value_head_layers) - 1):
                layers += [nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)]
                in_dim = hidden
            layers += [nn.Linear(in_dim, 1)]
            self.value_head = nn.Sequential(*layers)
        else:
            self.value_head = None

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if not torch.onnx.is_in_onnx_export() and seq_len > self.max_seq_len:
            raise ValueError(f"Input length {seq_len} exceeds configured max_seq_len {self.max_seq_len}.")
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.embedding_dropout(hidden)
        encoded = self.encoder(hidden)
        return self.final_norm(encoded[:, -1, :])

    def forward(self, input_ids: torch.Tensor, return_value: bool = False):
        pooled = self._encode(input_ids)
        logits = self.head(pooled)
        if return_value:
            if self.value_head is None:
                raise RuntimeError("value head が無いモデルで return_value=True が呼ばれました。")
            value = torch.tanh(self.value_head(pooled)).squeeze(-1)  # [-1,1], 手番側視点の勝率推定
            return logits, value
        return logits
