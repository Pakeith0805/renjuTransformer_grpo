#!/usr/bin/env python
"""
学習の良さレポート一括生成
==========================
対象モデル(モデルA)について、複数の強さ指標をまとめて実行し、1つの Markdown に書き出す。

  ① play_versus.py        : モデルA vs pretrained  (temp 0 で2戦 / 0.1 で100戦 / 1 で100戦)
  ② evaluate_versus_mcts  : チェックポイントを ~20 イテごとに MCTS(teacher) 評価
  ③ eval_decline_curve.csv: 既存の劣化カーブ CSV を要約

使い方:
  uv run python scripts/eval_report.py \
      --model-a artifacts/exp_minimax_topk_g32/grpo_checkpoint_115.pt \
      --ckpt-dir artifacts/exp_minimax_topk_g32 \
      --decline-csv eval_decline_curve.csv \
      --out eval_report.md

  ②を回さない: --ckpt-dir を省略 (モデルA単体を1回だけ MCTS 評価)
  ③を省略    : --decline-csv を存在しないパスにするか省略
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT_ROOT / "scripts"

WIN_RE = re.compile(r"Total Wins:\s*\d+\s*/\s*\d+\s*\(([\d.]+)%\)")
PLIES_RE = re.compile(r"Average Game Length:\s*([\d.]+)")
WINPLY_RE = re.compile(r"Model WINS\):\s*([\d.]+)")
LOSSPLY_RE = re.compile(r"Model LOSES\):\s*([\d.]+)")
UNIQUE_RE = re.compile(r"Unique Games:\s*(\d+)\s*/\s*(\d+)")


def run(cmd: list[str]) -> tuple[str, str]:
    """スクリプトを実行し (stdout, stderr) を返す。失敗しても落とさず空を返す。"""
    print("  $ " + " ".join(cmd), file=sys.stderr)
    try:
        p = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=36000)
        return p.stdout, p.stderr
    except Exception as e:  # noqa: BLE001
        return "", f"[run error] {e}"


def first(pat, text, default=None, group=1):
    m = pat.search(text)
    return m.group(group) if m else default


# --------------------------------------------------------------------------- #
# ① play_versus: モデルA vs pretrained
# --------------------------------------------------------------------------- #
def section_versus(model_a, pretrained, device, raw):
    rows = []
    for temp, n in [(0.0, 2), (0.1, 100), (1.0, 100)]:
        cmd = [
            sys.executable, str(SCRIPTS / "play_versus.py"),
            f"versus.model_a_path={pretrained}",      # A=pretrained
            f"versus.model_b_path={model_a}",         # B=対象(モデルA)
            f"versus.num_games={n}", f"versus.temperature={temp}",
            f"versus.device={device}",
        ]
        out, err = run(cmd)
        raw.append((f"play_versus temp={temp} n={n}", out + ("\n" + err if err.strip() else "")))
        wins = WIN_RE.findall(out)               # [A(pretrained)%, B(モデルA)%]
        a_pct = wins[0] if len(wins) > 0 else "?"
        b_pct = wins[1] if len(wins) > 1 else "?"
        uniq = UNIQUE_RE.search(out)
        uniq_s = f"{uniq.group(1)}/{uniq.group(2)}" if uniq else "-"
        plies = first(PLIES_RE, out, "-")
        rows.append((temp, n, b_pct, a_pct, uniq_s, plies))

    md = ["## ① play_versus: モデルA vs pretrained\n",
          "| temp | 対局数 | モデルA勝率 | pretrained勝率 | ユニーク棋譜 | 平均手数 |",
          "|---|---|---|---|---|---|"]
    for temp, n, b, a, u, p in rows:
        md.append(f"| {temp} | {n} | {b}% | {a}% | {u} | {p} |")
    md.append("\n*temp1で50%付近なら互角、低温で勝率が跳ねるのは決定論exploit/暗記の兆候"
              "(ユニーク棋譜が少ないか確認)。*\n")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# ② evaluate_versus_mcts: チェックポイントを ~20イテごとに MCTS(teacher) 評価
# --------------------------------------------------------------------------- #
def list_checkpoints(ckpt_dir, step):
    ckpts = []
    for f in Path(ckpt_dir).glob("grpo_checkpoint_*.pt"):
        m = re.search(r"_(\d+)\.pt$", f.name)
        if m:
            ckpts.append((int(m.group(1)), f))
    ckpts.sort()
    picked = [(n, f) for (n, f) in ckpts if n % step == 0]
    if ckpts and ckpts[-1] not in picked:   # 最終チェックポイントは必ず含める
        picked.append(ckpts[-1])
    return picked


def eval_one_vs_mcts(model_path, pretrained, device, sims, games, raw, label):
    cmd = [
        sys.executable, str(SCRIPTS / "evaluate_versus_mcts.py"),
        f"eval_mcts.model_path={model_path}",
        f"eval_mcts.ref_model_path={pretrained}",   # teacher: pretrained-prior MCTS
        f"eval_mcts.num_games={games}", f"eval_mcts.mcts_simulations={sims}",
        "eval_mcts.use_tss=true", "eval_mcts.use_puct=true",
        "eval_mcts.model_temperature=0", f"eval_mcts.device={device}",
    ]
    out, err = run(cmd)
    raw.append((f"evaluate_versus_mcts {label}", out + ("\n" + err if err.strip() else "")))
    wins = WIN_RE.findall(out)                  # [Target%, MCTS%]
    return dict(
        model=wins[0] if wins else "?",
        mcts=wins[1] if len(wins) > 1 else "?",
        plies=first(PLIES_RE, out, "-"),
        win_ply=first(WINPLY_RE, out, "-"),
        loss_ply=first(LOSSPLY_RE, out, "-"),
    )


def section_mcts(model_a, pretrained, ckpt_dir, step, device, sims, games, raw):
    md = [f"## ② evaluate_versus_mcts: vs teacher-MCTS "
          f"(sims={sims}, TSS/PUCT=on, temp0, {games}戦)\n",
          "| iter | モデル勝率 | MCTS勝率 | 平均手数 | 勝ち手数 | 負け手数 |",
          "|---|---|---|---|---|---|"]
    targets = []
    # iter0 として pretrained を基準に入れる
    targets.append(("0 (pretrained)", pretrained))
    if ckpt_dir:
        for n, f in list_checkpoints(ckpt_dir, step):
            targets.append((str(n), str(f)))
    else:
        targets.append(("modelA", model_a))
    for label, path in targets:
        r = eval_one_vs_mcts(path, pretrained, device, sims, games, raw, label)
        md.append(f"| {label} | {r['model']}% | {r['mcts']}% | {r['plies']} | "
                  f"{r['win_ply']} | {r['loss_ply']} |")
    md.append("\n*teacher-MCTSは本命の強さ指標。勝率が上昇すれば本物の改善、低下は侵食。*\n")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
# ③ eval_decline_curve.csv の要約
# --------------------------------------------------------------------------- #
def section_decline(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return f"## ③ eval_decline_curve.csv\n\n(ファイルが見つかりません: {csv_path})\n"
    rows = list(csv.DictReader(p.open(encoding="utf-8")))
    if not rows:
        return f"## ③ eval_decline_curve.csv\n\n(空ファイル: {csv_path})\n"

    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    wins = [fnum(r.get("policy_win_pct")) for r in rows]
    wins = [w for w in wins if w is not None]
    md = [f"## ③ 劣化カーブ要約 ({csv_path}, {len(rows)} 点)\n"]
    if wins:
        k = max(1, len(wins) // 3)
        avg = lambda xs: sum(xs) / len(xs)  # noqa: E731
        md += [
            "| 区間 | 平均policy勝率 |",
            "|---|---|",
            f"| 前期 (最初の{k}点) | {avg(wins[:k]):.1f}% |",
            f"| 中期 | {avg(wins[k:-k]) if len(wins) > 2 * k else avg(wins):.1f}% |",
            f"| 後期 (最後の{k}点) | {avg(wins[-k:]):.1f}% |",
            f"| 全体 (min/max) | {min(wins):.1f}% / {max(wins):.1f}% |",
            "\n*前期>後期なら侵食、上昇なら改善。生CSVは添付の通り。*\n",
        ]
    # 生テーブル(先頭と末尾を抜粋)
    cols = list(rows[0].keys())
    md.append("<details><summary>CSV (全行)</summary>\n")
    md.append("| " + " | ".join(cols) + " |")
    md.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        md.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md.append("\n</details>\n")
    return "\n".join(md)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-a", required=True, help="対象モデル(モデルA)のチェックポイント")
    ap.add_argument("--pretrained", default="artifacts/checkpoints/pretrained.pt")
    ap.add_argument("--ckpt-dir", default=None, help="②用: grpo_checkpoint_N.pt が入ったディレクトリ")
    ap.add_argument("--iter-step", type=int, default=20)
    ap.add_argument("--decline-csv", default="eval_decline_curve.csv")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mcts-sims", type=int, default=200)
    ap.add_argument("--mcts-games", type=int, default=50)
    ap.add_argument("--out", default="eval_report.md")
    ap.add_argument("--skip", default="", help="スキップする節をカンマ区切りで: versus,mcts,decline")
    args = ap.parse_args()

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    raw: list[tuple[str, str]] = []
    parts = [
        f"# 学習評価レポート\n",
        f"- 対象モデル(モデルA): `{args.model_a}`",
        f"- pretrained: `{args.pretrained}`",
        f"- 生成: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}  / device={args.device}\n",
    ]

    if "versus" not in skip:
        print("[1/3] play_versus ...", file=sys.stderr)
        parts.append(section_versus(args.model_a, args.pretrained, args.device, raw))
    if "mcts" not in skip:
        print("[2/3] evaluate_versus_mcts ...", file=sys.stderr)
        parts.append(section_mcts(args.model_a, args.pretrained, args.ckpt_dir,
                                  args.iter_step, args.device, args.mcts_sims, args.mcts_games, raw))
    if "decline" not in skip:
        print("[3/3] decline curve ...", file=sys.stderr)
        parts.append(section_decline(args.decline_csv))

    # 生ログを末尾に折りたたみで添付
    parts.append("\n---\n## 生ログ\n")
    for title, text in raw:
        parts.append(f"<details><summary>{title}</summary>\n\n```\n{text.strip()}\n```\n</details>\n")

    Path(args.out).write_text("\n".join(parts), encoding="utf-8")
    print(f"\nレポートを書き出しました -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
