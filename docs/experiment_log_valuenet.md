# RenjuTransformer 実験ログ — GRPO侵食調査 → value判定者 → 葉評価診断

期間: 2026-06-24〜06-27 / ブランチ: `feature/value-net-leaf-eval`
目的: GRPO で連珠 policy を強くする。指標は **vs teacher-MCTS（sims200/TSS/PUCT/temp0）の勝率を200戦**で測る。

---

## 全体の流れ（一枚で）

```
[A] GRPO侵食の特性化            … magnitudeレバー(group_size↑/beta↓)は overshoot を助長、向きを直さない
      ↓ 原因は「報酬/判定者」だと特定
[B] value判定者(報酬を value net 化)
   B1 value pretrain (勝敗教師)   … sign_acc 80%
   B2 v1: 固定value判定者         … vs-teacher 12%→18%・超高速 / 凍結ゆえ peak→erode
   B3 v1.5: value共進化          … erode改善・上限20% / 但し振動
   B4 診断: MCTS葉をvalue化       … 50-50 = rolloutはボトルネックでない(葉評価は強さに無関係)
   B5 value天井測定(層数2/4/6)    … 全部80% = 容量でなく原理天井
      ↓ 次
   B6 CNN胴体で突破？(未着手)     … 部分的に可能性。先にフェーズ別sign_accで余地を測る
```

---

## [A] GRPO 侵食の特性化（value前）

**動機**: GRPO run が vs-teacher でピーク後に erode する。原因を切り分けたい。

| 実験 | 動機 | 結果 |
|---|---|---|
| top-K g16/g32 | 手集合を増やせばSNRで伸びるか | g16フラット/g32侵食。**group_size↑で侵食加速・KL連動** |
| beta=0.05 | アンカー緩めれば加速？ | **加速せず逆効果**(ピークから速く離れる) |
| vs-pretrained直接対決 | 本当に弱い？ | g32は pretrained に**78%(開幕多様化)で本物に強い**。但しvs-MCTSに転移せず=**非推移性** |
| TSS模倣テスト(自作) | 戦術を内在化してるか | 自然局面で両者**ほぼ0%**。GRPOで氷河的に微増 |

**結論/接続**: magnitudeレバー(group_size↑/beta↓)は **overshoot を助長するだけで向きを直さない**。
ボトルネックは **報酬/判定者**。50戦は蜃気楼(同一ckptでピークが80↔60)→**強さは200戦で測る**。
→ 「判定者を grounded で速くする」= value net へ。

## [B1] value ヘッド pretrain

**動機**: 報酬判定者を、ランダムでなく**実戦勝敗にアンカーした value net** にする。
**方法**: `data.csv.gz`(局面→手, 対局順, 1.44M局面/10万局)を対局に区切り、終局勝者→各局面に手番視点±1ラベル。
policy 胴体に value ヘッド(MLP)を足し MSE回帰。
**結果**: 1エポックで **val_sign_acc ≈ 80%** に頭打ち。胴体fine-tune+2層MLPで達成。

## [B2] v1: 固定 value 判定者（報酬を value net 化）

**動機**: GRPO の報酬(MCTSロールアウト)を value判定者に差し替え。固定=定常=非自己参照、かつ超高速。
**方法**: `agent.value_judge_rewards` = 候補手の次局面を**一括GPU value評価**(報酬=-value)+TSS上書き(即勝ち+1/相手VCF-1)。`use_value_judge` フラグ。
**結果**:
- vs-teacher 天井 **12%→18%(iter150, 2evalで一致)**。報酬が1 forward=**桁違いに速い**。
- **但し凍結ゆえ peak→erode**(18%@150→7%@200)。policyがpretrainedを追い越し**value陳腐化→漂流**。

## [B3] v1.5: value 共進化（co-train）

**動機**: 凍結の陳腐化を外す。GRPO中にvalue netも自己対戦勝敗で継続学習(AlphaZero式)。
**方法**: value専用optimizer+リプレイバッファ。full-game収集で勝者取得→全局を教師に毎iter更新。`value_cotrain` フラグ。
**結果**:
- **erode改善**: iter200=18%, 245=20% と**後半でも高値を更新、永続erodeが消えた**。value_cotrain_loss 1.0→0.36(健全)。
- **但し振動**: 隣接ckptで4.5↔20%と乱高下(共進化の不安定)。ベストは中盤ckpt(~20%)。

## [B4] 診断: MCTSの葉評価を rollout↔value に差し替え

**動機**: 「rolloutが弱い→value化で MCTS が強くなる」なら rollout がボトルネック、を測りたい。
**方法**: mcts.cpp に value コールバック(`set_value_fn_c_api`)。同一C++木で葉だけ切替え、value-MCTS vs rollout-MCTS を head-to-head。
**結果**: **TSS有52-48 / TSS無 50-50 = 引き分け**。**rolloutはボトルネックでない**。
理由: rolloutは`choose_weighted_top_move`=ヒューリスティック(純ランダムでない)で既に十分賢い。
MCTSの強さは探索/prior/TSSで決まり葉評価では律速されない。
**接続**: **value葉化MCTS(v2)を"強さ"目的で作る理由は無い**。valueの価値は**報酬側**に確定。

## [B5] value の天井は容量か原理か

**動機**: 80%が2層MLPの容量不足かもしれない。
**方法**: value ヘッド層数を 2/4/6(+hidden512)に増やして sign_acc 再測定。
**結果**: **全て 79.9-80.0% で完全一致**。**容量でなく原理天井**(先読み無し静的評価の限界。序盤局面は同一盤面が±両方のラベル=削減不能)。

## [B6] CNN で突破？（未着手・次の候補）

ヘッド容量は否定したが**特徴抽出器(胴体)は未検証**。盤は2D空間的で、1次元transformerよりCNNの方が連/形/脅威の inductive bias が合う→**数ポイント突破の可能性**。但し序盤コインフリップの**削減不能床**があるので~95%は無理。
**先にやる安いチェック**: **フェーズ(手数)別 sign_acc**。終盤が既に~95%なら80%は序盤床で占有=CNN余地小。終盤も85%止まりならCNN余地あり。

---

## 作ったツール / 実装

- `scripts/test_tss_imitation.py` — VCFソルバーをオラクルにTSS模倣を自動採点(型/自然局面)
- `scripts/eval_report.py` — vs-pretrained / vs-teacher / 劣化カーブを1ファイルに(iter範囲指定可)
- `scripts/pretrain_value.py` — value pretrain(層数可変, 各epoch保存)
- `scripts/diag_value_vs_rollout_mcts.py` — 葉だけ差し替えA/B
- `model.py` value head(任意・可変層), `agent.value_judge_rewards`, trainer value共進化, `mcts.cpp` 葉コールバック
- フラグ: `use_value_judge` / `value_cotrain`(+lr/steps/every/buffer) / `value_head_layers`/`hidden`

## 横断的な教訓

1. **強さは vs-teacher を200戦で測る**(50戦は蜃気楼)。
2. **magnitudeレバー(group_size↑/beta↓)は overshoot を助長**、向きを直さない。
3. **value net**: 報酬判定者として有効(12→18-20%)・**葉評価としては無価値**(50-50)・**80%は原理天井**・**速い**。
4. **rollout(ヒューリスティック)は MCTS のボトルネックでない**。
5. value判定者で**実験が桁違いに速くなった**(MCTSロールアウト→1 forward)。

## 開いている糸口 / 次

- **v1.5の振動抑制**(本命): 共進化ペース(`value_cotrain_every`↑/`value_cotrain_lr`↓) + ①policy一部層凍結(ドリフト抑制アンカー)。
- **CNN value**: フェーズ別sign_accで余地ありなら試す。
- 現状ベスト成果物: v1.5 の~20% ckpt(但し振動。中盤を選ぶ)。
