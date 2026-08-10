# -*- coding: utf-8 -*-
"""セッション／テスト結果のログ出力（対話モードと自動テストで共通利用）。

出力先:  {output_dir}/{モデル名}-{タイムスタンプ}.txt
形式  :  ヘッダ数行のあと、1行1エントリ = "入力<TAB>サジェスト"
         サジェストが無い場合は "(なし)" とする。
"""
from __future__ import annotations

import os
import re
from datetime import datetime

_INVALID_CHARS = re.compile(r'[\\/:*?"<>|\r\n\t]')


def sanitize_model_name(model: str) -> str:
    """ファイル名に使えない文字を置換（例: qwen/qwen3.6-27b -> qwen-qwen3.6-27b）。"""
    name = _INVALID_CHARS.sub("-", (model or "").strip())
    name = re.sub(r"\s+", "_", name).strip("-")
    return name or "model"


def timestamp() -> str:
    """ファイル名用タイムスタンプ (YYYYMMDD-HHMMSS)。"""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def make_output_path(model: str, output_dir: str = "output", ts: str | None = None) -> str:
    """出力ディレクトリを生成し、ファイルパスを返す。"""
    os.makedirs(output_dir, exist_ok=True)
    ts = ts or timestamp()
    return os.path.join(output_dir, f"{sanitize_model_name(model)}-{ts}.txt")


def write_header(path: str, model: str, mode_label: str) -> None:
    """ファイル先頭にメタ情報（モデル/モード/時刻/列説明）を書き込む。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"# model: {model}\n")
        f.write(f"# mode: {mode_label}\n")
        f.write(f"# timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# columns: input<TAB>suggestion  (サジェスト無しは (なし))\n\n")


def format_entry(text: str, suggestion: str | None) -> str:
    sug = suggestion if suggestion else "(なし)"
    return f"{text}\t{sug}\n"


def append_entry(path: str, text: str, suggestion: str | None) -> None:
    """1エントリ（入力とサジェスト）を追記し、即時フラッシュする。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write(format_entry(text, suggestion))
        f.flush()
