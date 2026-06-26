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
        # 容量を持たせるため 1層線形でなく小さな MLP にする(凍結胴体からでも value を引き出しやすい)。
        self.with_value_head = with_value_head
        self.value_head = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, 1),
            )
            if with_value_head
            else None
        )

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
