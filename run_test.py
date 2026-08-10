# -*- coding: utf-8 -*-
"""
自動テスト: test-settings.yml を読み込み、入力ファイルの各行を1件ずつ推論し、
入力と推論結果（サジェスト）をファイルに出力する。

出力形式は対話モードのセッションログと同一:
    {output_dir}/{モデル名}-{タイムスタンプ}.txt

使い方:
    python run_test.py                     # 既定の test-settings.yml を使用
    python run_test.py --config other.yml  # 設定ファイルを指定

設定ファイル (test-settings.yml) の必須項目:
    model       : テストに使用するモデル名
    input_file  : 1行1単語の入力ファイルパス
任意項目（省略時は既定値）:
    words_file, base_url, output_dir
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# 標準出力・エラー出力を UTF-8 に固定（Windows コンソールの文字化け防止）
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

import httpx
import yaml
from openai import AsyncOpenAI

from ai_checker import build_system_prompt, judge_typo
from session_log import make_output_path, write_header, append_entry

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(SCRIPT_DIR, "test-settings.yml")
DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
REQUIRED_FIELDS = ["model", "input_file"]


def _resolve(path: str | None, base: str) -> str | None:
    """相対パスを base 基準で絶対化（絶対パスはそのまま）。"""
    if not path:
        return path
    return path if os.path.isabs(path) else os.path.join(base, path)


def load_yaml(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    r.raise_for_status()
    return [m.get("id") for m in r.json().get("data", [])]


async def run_batch(cfg: dict, config_dir: str) -> None:
    model: str = cfg["model"]
    base_url: str = cfg.get("base_url") or DEFAULT_BASE_URL
    words_file = _resolve(cfg.get("words_file", "input/tmp.txt"), config_dir)
    input_file = _resolve(cfg["input_file"], config_dir)
    output_dir: str = cfg.get("output_dir", "output")

    # --- 語彙読込 -------------------------------------------------------
    if not os.path.isfile(words_file):
        print(f"エラー: 語彙ファイルが見つかりません: {words_file}", file=sys.stderr)
        sys.exit(1)
    with open(words_file, encoding="utf-8") as f:
        words = [w.strip() for w in f if w.strip()]
    vocab = set(words)
    system_prompt = build_system_prompt(words)
    print(f"語彙: {words_file} ({len(words)} 語)")

    # --- 入力読込 -------------------------------------------------------
    if not os.path.isfile(input_file):
        print(f"エラー: 入力ファイルが見つかりません: {input_file}", file=sys.stderr)
        sys.exit(1)
    with open(input_file, encoding="utf-8") as f:
        inputs = [w.strip() for w in f if w.strip()]
    print(f"入力: {input_file} ({len(inputs)} 行)")

    # --- モデル確認 -----------------------------------------------------
    try:
        models = list_models(base_url)
    except Exception as e:
        print(f"エラー: LMStudio({base_url}) に接続できません: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"ロード済みモデル: {', '.join(models) if models else '（なし）'}")
    if model not in models:
        print(f"⚠ 警告: モデル '{model}' はロードされていません（呼び出しごとに失敗する可能性があります）", file=sys.stderr)

    # --- 出力ファイル準備 ----------------------------------------------
    out_path = make_output_path(model, output_dir)
    write_header(out_path, model, "auto-test")
    print(f"📄 出力: {out_path}\n")

    # --- 推論ループ -----------------------------------------------------
    client = AsyncOpenAI(base_url=base_url, api_key="lm-studio", timeout=60.0)
    ok = err = 0
    total = len(inputs)
    for i, word in enumerate(inputs, 1):
        try:
            suggestion = await judge_typo(client, model, word, system_prompt, vocab, timeout=30.0)
            ok += 1
        except Exception as e:
            suggestion = None
            err += 1
            print(f"  [{i}/{total}] {word} -> ERROR: {e}", file=sys.stderr)
        append_entry(out_path, word, suggestion)
        print(f"  [{i}/{total}] {word} -> {suggestion if suggestion else '-'}")

    print(f"\n完了: 全 {total} 件 / 成功 {ok} / エラー {err}")
    print(f"結果ファイル: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="自動テスト（test-settings.yml ベースの一括推論）",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="設定ファイル (YAML) のパス")
    args = p.parse_args()

    # --- 設定ファイル読込 ----------------------------------------------
    if not os.path.isfile(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}", file=sys.stderr)
        sys.exit(1)

    try:
        cfg = load_yaml(args.config)
    except yaml.YAMLError as e:
        print(f"エラー: 設定ファイルの YAML 解析に失敗しました: {e}", file=sys.stderr)
        sys.exit(1)

    if not isinstance(cfg, dict):
        print("エラー: 設定ファイルの形式が不正です（'キー: 値' の辞書形式にしてください）。", file=sys.stderr)
        sys.exit(1)

    # --- 必須項目チェック ----------------------------------------------
    missing = [k for k in REQUIRED_FIELDS if not cfg.get(k)]
    if missing:
        print(f"エラー: 設定ファイル（{args.config}）に必須項目が不足しています:", file=sys.stderr)
        for k in missing:
            print(f"   - {k}", file=sys.stderr)
        sys.exit(1)

    config_dir = os.path.dirname(os.path.abspath(args.config))
    asyncio.run(run_batch(cfg, config_dir))


if __name__ == "__main__":
    main()
