#!/usr/bin/env bash
# ============================================================
# 劣化カーブ計測スクリプト (並列版)
#   pretrained と各 grpo_checkpoint_N.pt を同一条件で評価し、
#   policy 勝率の推移を表 + CSV で出力する。
#
#   複数チェックポイントの評価を MAX_PAR 並列で実行する。
#   各 evaluate_versus_mcts.py は内部で 8 スレッドの MCTS プールを使うため、
#   16コア/32スレッド環境では MAX_PAR=4 (=32スレッド) が目安。
#
# 使い方:
#   bash scripts/eval_decline_curve.sh
#   CKPT_DIR=path/to/ckpts MAX_PAR=4 OPP=teacher bash scripts/eval_decline_curve.sh
# ============================================================
set -u

# ---- 設定 (環境変数で上書き可) --------------------------------
PRETRAINED="${PRETRAINED:-models/pretrained.pt}"
CKPT_DIR="${CKPT_DIR:-artifacts/exp_puct_tss_both_include_best_max80_kl_di}"
NUM_GAMES="${NUM_GAMES:-100}"
DEVICE="${DEVICE:-cpu}"
OUT_CSV="${OUT_CSV:-eval_decline_curve.csv}"
MAX_PAR="${MAX_PAR:-4}"          # 同時実行する評価プロセス数
TMP_DIR="${TMP_DIR:-.eval_curve_tmp}"

# 各プロセスの torch/OMP スレッドは絞る (MCTS の 8 スレッドプールを優先)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

# 対戦相手プリセット: OPP=uniform (弱い相手) / OPP=teacher (学習時の教師)
OPP="${OPP:-uniform}"
if [ "$OPP" = "teacher" ]; then
  REF_MODEL="$PRETRAINED"
  SIMS="${SIMS:-200}"
  USE_TSS=true
  USE_PUCT=true
  MODEL_TEMP="${MODEL_TEMP:-0}"
else
  REF_MODEL="uniform"
  SIMS="${SIMS:-50}"
  USE_TSS="${USE_TSS:-false}"
  USE_PUCT="${USE_PUCT:-false}"
  MODEL_TEMP="${MODEL_TEMP:-}"   # 空 = configデフォルト
fi

# ---- 評価対象一覧 (N<TAB>path) を数値順に作成 -----------------
# pretrained を N=0 として先頭に。grpo_checkpoint_N.pt は basename から N を抽出して数値ソート。
mkdir -p "$TMP_DIR"
LIST_FILE="$TMP_DIR/_list.tsv"
: > "$LIST_FILE"
printf "0\t%s\n" "$PRETRAINED" >> "$LIST_FILE"

if [ -d "$CKPT_DIR" ]; then
  while IFS= read -r f; do
    n=$(basename "$f" | sed -E 's/.*_([0-9]+)\.pt/\1/')
    printf "%s\t%s\n" "$n" "$f" >> "$LIST_FILE"
  done < <(find "$CKPT_DIR" -maxdepth 1 -name 'grpo_checkpoint_*.pt' 2>/dev/null)
else
  echo "WARN: CKPT_DIR not found: $CKPT_DIR (pretrained のみ評価)" >&2
fi
sort -n -k1,1 "$LIST_FILE" -o "$LIST_FILE"

# ---- 1チェックポイントを評価する関数 -------------------------
run_one() {
  local n="$1" ckpt="$2"
  local temp_arg=""
  [ -n "$MODEL_TEMP" ] && temp_arg="eval_mcts.model_temperature=${MODEL_TEMP}"

  local out
  out=$(uv run scripts/evaluate_versus_mcts.py \
        eval_mcts.model_path="$ckpt" \
        eval_mcts.ref_model_path="$REF_MODEL" \
        eval_mcts.num_games="$NUM_GAMES" \
        eval_mcts.mcts_simulations="$SIMS" \
        eval_mcts.use_tss="$USE_TSS" \
        eval_mcts.use_puct="$USE_PUCT" \
        eval_mcts.device="$DEVICE" \
        $temp_arg 2>/dev/null)

  local policy_pct mcts_pct plies win_plies loss_plies
  policy_pct=$(echo "$out" | grep "Total Wins:" | head -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
  mcts_pct=$(echo "$out"   | grep "Total Wins:" | tail -1 | sed -E 's/.*\(([0-9.]+)%\).*/\1/')
  plies=$(echo "$out" | grep "Average Game Length:" | sed -E 's/.*: +([0-9.]+).*/\1/')
  win_plies=$(echo "$out"  | grep "Model WINS"  | sed -E 's/.*: +([0-9.]+) plies.*/\1/')
  loss_plies=$(echo "$out" | grep "Model LOSES" | sed -E 's/.*: +([0-9.]+) plies.*/\1/')
  # 結果を per-iter ファイルに書く (ゼロ埋めで後段のソートを安定化)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$n" "$ckpt" "${policy_pct:-ERR}" "${mcts_pct:-ERR}" \
    "${plies:-ERR}" "${win_plies:-ERR}" "${loss_plies:-ERR}" \
    > "$TMP_DIR/res_$(printf '%06d' "$n").tsv"
  echo "  done iter=$n  policy=${policy_pct:-ERR}%"
}
export -f run_one
export PRETRAINED REF_MODEL NUM_GAMES SIMS USE_TSS USE_PUCT DEVICE MODEL_TEMP TMP_DIR

# ---- 並列ディスパッチ (MAX_PAR でスライディングウィンドウ) ----
echo "Running evals: opponent=$OPP sims=$SIMS games=$NUM_GAMES parallel=$MAX_PAR"
rm -f "$TMP_DIR"/res_*.tsv
running=0
while IFS=$'\t' read -r n ckpt; do
  run_one "$n" "$ckpt" &
  running=$((running + 1))
  if [ "$running" -ge "$MAX_PAR" ]; then
    wait -n 2>/dev/null || wait   # bash<4.3 なら全待ち
    running=$((running - 1))
  fi
done < "$LIST_FILE"
wait

# ---- 集計・出力 ----------------------------------------------
echo "iter,model_path,policy_win_pct,mcts_win_pct,avg_plies,win_plies,loss_plies" > "$OUT_CSV"
printf "\n%-14s %-11s %-10s %-9s %-9s %-9s\n" \
  "iter" "policy_win%" "avg_plies" "win_ply" "loss_ply" "mcts_win%"
printf -- "----------------------------------------------------------------\n"
for rf in $(ls "$TMP_DIR"/res_*.tsv 2>/dev/null | sort); do
  IFS=$'\t' read -r n ckpt policy mcts plies win_plies loss_plies < "$rf"
  label="$n"; [ "$n" = "0" ] && label="0_pretrained"
  printf "%-14s %-11s %-10s %-9s %-9s %-9s\n" \
    "$label" "$policy" "$plies" "$win_plies" "$loss_plies" "$mcts"
  echo "${label},${ckpt},${policy},${mcts},${plies},${win_plies},${loss_plies}" >> "$OUT_CSV"
done

echo
echo "Done. CSV -> $OUT_CSV   (opponent=$OPP, sims=$SIMS, games=$NUM_GAMES, parallel=$MAX_PAR)"