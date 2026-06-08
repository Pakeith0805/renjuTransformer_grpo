"""Tokenizer for Renju board-state CSV rows."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rules import BOARD_CELLS, BOARD_SIZE, legal_move_mask


@dataclass(slots=True)
class RenjuTokenizer:
    sep_token_id: int = 228 # 区切りトークンのid
    move_id_offset: int = 3 # 指し手id-オフセット=置く場所のインデックス

    @property # プロパティは変数みたいなもの。盤面のマス数
    def board_cells(self) -> int:
        return BOARD_CELLS

    @property # 盤面の一辺のサイズ
    def board_size(self) -> int:
        return BOARD_SIZE

    @property # モデルの入力トークンの長さ。盤面の大きさ+セパレータトークン
    def input_length(self) -> int:
        return self.board_cells + 1

    @property # 予測指し手の総数。盤面のマス数と同じ
    def num_labels(self) -> int:
        return self.board_cells

    @property # 語彙サイズ。使う文字の数。
    def vocab_size(self) -> int:
        return self.sep_token_id + 1

    # リストの長さが225であること、かつ含まれる値が0か1か2のどれかであることを確認
    def validate_board(self, board: list[int]) -> None:
        if len(board) != self.board_cells:
            raise ValueError(f"Expected {self.board_cells} cells, got {len(board)}.")
        invalid = [cell for cell in board if cell not in (0, 1, 2)]
        if invalid:
            raise ValueError(f"Board contains invalid tokens: {sorted(set(invalid))}")

    # 盤面リストが与えられたら、セパレータトークンをくっつけ、テンソルにして返す
    def encode_input(self, board: list[int]) -> torch.Tensor:
        self.validate_board(board)
        tokens = board + [self.sep_token_id]
        return torch.tensor(tokens, dtype=torch.long)

    # 指し手idを、インデックスに変換。3引くだけ。
    def encode_label(self, move_id: int) -> int:
        label = move_id - self.move_id_offset
        if not 0 <= label < self.num_labels:
            raise ValueError(f"Move id {move_id} is out of range.")
        return label

    # インデックスを、指し手idに変換。3足すだけ
    def decode_label(self, label: int) -> int:
        if not 0 <= label < self.num_labels:
            raise ValueError(f"Label {label} is out of range.")
        return label + self.move_id_offset
    # 上記と同じ。使う場面が違うだけ。
    def move_id_to_index(self, move_id: int) -> int:
        return self.encode_label(move_id)

    # 上記と同じ。使う場面が違うだけ。
    def index_to_move_id(self, index: int) -> int:
        return self.decode_label(index)

    # 
    def encode_csv_row(self, row: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        expected_length = self.board_cells + 2
        if len(row) != expected_length:
            raise ValueError(f"Expected {expected_length} columns, got {len(row)}.")
        board = row[: self.board_cells]
        sep = row[self.board_cells]
        if sep != self.sep_token_id:
            raise ValueError(f"Expected SEP token {self.sep_token_id}, got {sep}.")
        move_id = row[-1]
        input_ids = self.encode_input(board)
        label = torch.tensor(self.encode_label(move_id), dtype=torch.long)
        return input_ids, label

    def parse_board_csv(self, board_csv: str) -> list[int]:
        values = [item.strip() for item in board_csv.split(",") if item.strip()]
        board = [int(value) for value in values]
        self.validate_board(board)
        return board

    # boolのlegalマスクを作成
    def legal_move_mask(self, board: list[int]) -> torch.Tensor:
        self.validate_board(board)
        mask = legal_move_mask(board)
        return torch.tensor(mask, dtype=torch.bool)
