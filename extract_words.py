#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
words.txt から先頭 n 単語を抜き出して tmp.txt に保存するスクリプト。

使い方:
    python extract_words.py N [INPUT] [OUTPUT]

引数:
    N        : 抜き出す先頭からの単語数（必須・正の整数）
    INPUT    : 入力ファイル（省略時はスクリプトと同じフォルダの words.txt）
    OUTPUT   : 出力ファイル（省略時は tmp.txt）

例:
    python extract_words.py 100            # 先頭100語を tmp.txt へ
    python extract_words.py 500            # 先頭500語を tmp.txt へ
    python extract_words.py 100 words.txt my_words.txt
"""

import os
import sys

# 標準出力・エラー出力を UTF-8 に固定（Windows コンソールの文字化け防止）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass


def main() -> int:
    # --- 引数の検証 ---------------------------------------------------------
    if len(sys.argv) < 2:
        sys.stderr.write(
            "エラー: 抜き出す単語数 N を指定してください。\n"
            "使い方: python extract_words.py N [INPUT] [OUTPUT]\n"
        )
        return 1

    n_str = sys.argv[1]
    if not n_str.lstrip("+").isdigit():
        sys.stderr.write(f"エラー: N は正の整数で指定してください（入力値: {n_str!r}）。\n")
        return 1

    n = int(n_str)
    if n <= 0:
        sys.stderr.write(f"エラー: N は 1 以上の整数で指定してください（入力値: {n}）。\n")
        return 1

    # デフォルトの入出力ファイルは「スクリプトと同じフォルダ」に解決
    here = os.path.dirname(os.path.abspath(__file__))
    input_path = sys.argv[2] if len(sys.argv) >= 3 else os.path.join(here, "words.txt")
    output_path = sys.argv[3] if len(sys.argv) >= 4 else os.path.join(here, "tmp.txt")

    # --- 入力ファイルの読み込み --------------------------------------------
    if not os.path.isfile(input_path):
        sys.stderr.write(f"エラー: 入力ファイルが見つかりません ({input_path})\n")
        return 1

    # 空白・空行を除外して「1行1単語」として扱う
    with open(input_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    if not words:
        sys.stderr.write(f"エラー: 入力ファイルに単語がありません ({input_path})\n")
        return 1

    # --- 先頭 n 個を取得（超過時は全件＋警告） ------------------------------
    if n > len(words):
        sys.stderr.write(
            f"注意: 指定された N={n} は収録語数({len(words)})を超えています。"
            f"全 {len(words)} 語を出力します。\n"
        )
    selected = words[:n]

    # --- tmp.txt へ出力 ----------------------------------------------------
    # newline="\n" で LF 改行を強制（words.txt と同じ LF に揃える）
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(selected))
        f.write("\n")  # 末尾に改行

    print(
        f"完了: {input_path} の先頭 {len(selected)} 語を {output_path} に保存しました。"
        f"（全収録語数: {len(words)}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
