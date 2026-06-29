# RenjuTransformer — TSS整備と報酬被覆拡大：これまでと今後

最終更新: 2026-06-28 / ブランチ: `feature/tss-faithful`

このドキュメントは「faithful TSS（原論文準拠の脅威空間探索）の整備」と、その動機である
「GRPO報酬の被覆拡大（−value依存を減らす）」について、**やったこと・成果物・今後の動き**を1枚にまとめたもの。
詳細な実装経過は [`tss_implementation_log.md`](tss_implementation_log.md) を参照。

---

## 1. 背景と目的

- **目的**: 連珠AI（Transformer policy + 推論時 MCTS+TSS）を GRPO で強くする。指標は **vs teacher-MCTS 200戦** と
  **TSS模倣率**（`test_tss_imitation.py`）。
- **見つかっていた核心問題**: GRPO の value判定者報酬は、
  - VCF（四追い）で決まる局面だけ ±1 の**正確ラベル**を出すが、それは全局面の少数派。
  - **残り90%以上は `−value`**（80%天井・中盤で取り違える value net）が支配 → **怪しい先生が勾配の大半**を握り、
    pretrained の広い実力を侵食（KLドリフト）。
- **狙い**: TSS の被覆を **VCF（四のみ）→ VCT（四＋活三）** へ広げ、**鋭い中盤の戦術局面を正確ラベルで接地** →
  `−value` への依存を減らし、侵食を抑えつつ模倣を伸ばす。

---

## 2. やったこと（時系列の要点）

### 2.1 報酬とメトリクスの整合
- value判定者報酬に **「自分の VCF 初手 → +1」** を追加。`test_tss_imitation` のオラクル手と**報酬の最良手を一致**させた
  （これが無いと模倣は氷河的にしか進まない）。→ 自然局面の模倣が ~1% 固着から動いた。
- ただし長回しで **peak→erode**（KLドリフト由来、adv collapse ではない）を mlflow で確認。**beta=0.04→0.2** が抑制レバー。

### 2.2 faithful TSS の整備（本セッションの主作業）
1. **健全性ハーネス** `verify_tss_soundness.py` を新規作成（偽陽性=ありもしない必勝を返す、を独立検証）。
2. 現行 VCF に **~3% の偽陽性**を検出 → 追い詰めた結果:
   - 大半は **防御側即五の見落とし**（本物のバグ, P0で修正）。
   - 残り1件は **「黒の唯一の受けが禁手」を検証ツールが無視**していた偽警報＝**ソルバーは正しかった**、と決着。
3. 検証済みの**総当たり強制勝ちオラクル** `bf_forced_win` を参照に、**正しい四VCFをC++移植** → production の
   `solve_vcf_c_api` を新ソルバー `solve_vct_recursive(fours_only)` に**差し替え**（false_positive=0、同被覆）。
4. **活三VCT** を `solve_vct_recursive(fours_only=false)` に実装し、3段で健全かつ高速化:
   - **conservative defender (Allis)**: 守りのブロック分岐を「cost squares を一括占有した最防御盤で1回再帰」に集約。
   - **dependency-based search (Allis)**: 再帰では攻め候補を直前手の近傍に限定し OR分岐を削減。
   - **五完成点の幾何スキャン**: 石を通る4方向×距離4（最大32マス）で取りこぼし0%・高速。
   - 結果: **健全性ゲート false_positive=0**（型生成・depth=3, 38秒/iter で実用速度）。

### 2.3 報酬への配線
- `value_judge_rewards` に **`use_vct`（既定OFF）** を追加。ON で own勝ち/相手脅威を VCF→VCT で判定し**活三を接地**。
- config: `use_vct` / `vct_depth`（既定4）。production は既定OFF＝**無影響**。

---

## 3. 成果物（ファイル・コマンド）

| 種別 | 場所 | 役割 |
|---|---|---|
| TSSソルバー本体 | `mcts.cpp` `solve_vct_recursive` / `solve_vct_c_api` | 正しい四VCF＋活三VCT（fours_only切替） |
| 健全性検証 | `scripts/verify_tss_soundness.py` | 偽陽性ハンター＋総当たりオラクル＋完全性比較 |
| 模倣メトリクス | `scripts/test_tss_imitation.py` | VCFオラクルで policy の模倣率を採点 |
| 報酬 | `src/grpo/agent.py` `value_judge_rewards` | own-VCF/VCT +1, 相手脅威 −1, それ以外 −value |
| 起動 | `renju-grpo.py` / `config/config_grpo.yaml` | `use_vct` / `vct_depth` / `beta` 等 |
| 実装ログ | `docs/tss_implementation_log.md` | 設計・差分・経過の詳細 |

**健全性ゲート（再ビルド後に必須）**:
```bash
g++ -O3 -shared -fPIC -static-libgcc -static-libstdc++ -o mcts.so mcts.cpp
uv run python scripts/verify_tss_soundness.py --positions 80 --solver vct --vct-threes --use-templates --depth 3
# 期待: false_positive=0
```

**VCT被覆拡大の実験起動**:
```bash
uv run renju-grpo.py \
  grpo.checkpoint_path=models/pretrained.pt \
  grpo.use_value_judge=true grpo.value_judge_path=models/pv_L2.pt \
  grpo.use_full_game_training=true grpo.beta=0.2 \
  grpo.use_vct=true grpo.vct_depth=3 \
  grpo.tss_imitation_eval=true grpo.epochs=3000
```

---

## 4. 現在の実験状態（2026-06-28）

- `use_vct=true vct_depth=3 beta=0.2` で起動。**~38秒/iter（許容範囲）**。
- iter1 時点: vs-pretrained 56%、TSS模倣 top1=16.8%（own_four=26 / block_four=17 / **four_three=4** / priority=20）。
- 注目: **`four_three`（活三絡みの四三）カテゴリが VCT接地で動くか**。

---

## 5. 今後の動き

### 短期（この実験の評価）
- [ ] `tss_imitation_top1` と **`four_three` カテゴリ**の推移を追う（VCTで活三が接地され伸びるか）。
- [ ] **侵食チェック**: `vs-pretrained` / 自然局面模倣 / mlflow KL を見て、beta=0.2 で peak→erode が抑えられたか。
- [ ] 速度: 後半 iter の it/s が安定するか（eval は 100iter毎なので平常時はもっと速いはず）。

### 中期（整合と精度）
- [ ] **train/eval 整合**: 報酬がVCTなら `test_tss_imitation` のオラクルも **VCT化**（活三学習を正しく測る）。今はVCFのまま。
- [ ] **深いVCTの健全性**: depth>3 でも false_positive=0 を確認（時間が許せば）。
- [ ] cost squares `C=G` の完全性: 複雑な三形で受けを取りこぼさないか、ランダム局面でも健全性ゲートを広く回す。

### 長期（戦略）
- [ ] VCTが模倣・侵食に効くなら、**推論時 TSS も VCT化**（現状 production VCF）して MCTS+TSS を強化。
- [ ] 別軸: **value net の改善**（CNN胴体・フェーズ別sign_acc）で `−value` 自体を良い先生にする（被覆拡大と相補的）。
- [ ] チャンク/蒸留: VCF/VCT手順（`solve_vct_path` 相当）を密な教師に使う案（action-chunk の連珠版）。

---

## 6. 引き継ぎメモ（落とし穴）

- **健全性は「ソルバー＋検証器」両方が連珠ルール（黒禁手）を完全実装して初めて担保**される。今回の残存FPは検証器の
  黒禁手見落としだった。新しい検証を足すときも禁手を忘れない。
- **C++ はローカルでビルド/検証できない**（コンパイラ無し）。改修→リモート再ビルド→ハーネス、のループが必須。
  `mcts.so` は実行コンテナ(glibc一致)内でビルドすること。
- VCTは「健全・速い・完全」の三立が難しい。今回は **健全最優先＋依存関係検索/幾何スキャンで速度確保**、完全性は
  依存関係検索ぶん多少譲歩（稀に非連結の深い勝ちを取りこぼす）。報酬用途なら許容。
- 強さは **vs teacher 200戦**で測る（50戦は蜃気楼）。模倣率は **型と自然局面で別物**（型は上振れ）。
