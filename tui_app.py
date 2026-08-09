# -*- coding: utf-8 -*-
"""
タイポサジェスト TUI アプリ（LMStudio / Gemma2 2b）

使い方:
    # 1) 事前にテストサイズの語彙を用意（先頭N語を tmp.txt へ）
    python extract_words.py 200

    # 2) LMStudio の Server で Gemma2 2b をロード後、起動
    python tui_app.py --model gemma-2-2b-it

挙動:
    入力が1秒止まると AI がタイポ判定。
      - タイポと思しき場合は下部ツールバーにサジェスト表示。
      - [Tab] で確定（入力を置換）、[Enter] で送信して継続、Ctrl-C/Ctrl-D で終了。

便利オプション:
    --check        起動診断のみ行い終了（語彙数・モデル確認）
    --once INPUT   対話モードに入らず1回だけ判定して終了
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

import httpx
from openai import AsyncOpenAI
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.styles import Style

# Git Bash/MinTTY 等（TERM=xterm）から起動した際のコンソール取得失敗を捕捉するため
try:
    from prompt_toolkit.output.win32 import NoConsoleScreenBufferError as _NoConsoleError
except Exception:  # 非 Windows 環境など
    _NoConsoleError = ()

from ai_checker import build_system_prompt, judge_typo

# 標準出力・エラー出力を UTF-8 に固定（Windows コンソールの文字化け防止）
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (ValueError, OSError):
            pass

DEFAULT_BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
DEFAULT_MODEL = os.environ.get("LMSTUDIO_MODEL", "gemma-2-2b-it")
DEFAULT_WORDS = "tmp.txt"
DEFAULT_DEBOUNCE = 1.0

STYLE = Style.from_dict({
    "prompt": "fg:ansicyan bold",
    "suggest": "fg:ansiyellow bold",
    "status": "fg:ansibrightblack",
    "ok": "fg:ansigreen",
    "warn": "fg:ansired",
})


def load_words(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return [w.strip() for w in f if w.strip()]


def list_models(base_url: str, timeout: float = 5.0) -> list[str]:
    r = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout)
    r.raise_for_status()
    return [m.get("id") for m in r.json().get("data", [])]


def choose_model_interactive(models: list[str], preselect: str | None = None) -> str | None:
    """LMStudio のモデル一覧から1つ選択させる（埋め込みモデルは候補から除外）。

    radiolist_dialog はリスト上で Enter を押しても確定できず（Tab→OK が必要）なため、
    Enter で即確定できる独自の全画面ピッカーを構築する。
    戻り値: 選択されたモデルID（キャンセル時は None）。
    """
    from prompt_toolkit.application import Application
    from prompt_toolkit.layout import Layout
    from prompt_toolkit.layout.containers import Window
    from prompt_toolkit.layout.controls import FormattedTextControl

    chat_models = [m for m in models if "embed" not in m.lower()]
    if not chat_models:
        return None
    start = chat_models.index(preselect) if preselect in chat_models else 0
    state = {"i": start}

    def render():
        out = []
        out.append(("class:title", "LMStudio モデル選択\n"))
        out.append(("class:hint", "  ↑/↓: 移動    Enter: 決定    Esc: キャンセル\n\n"))
        for i, m in enumerate(chat_models):
            if i == state["i"]:
                out.append(("class:selected", f"  ❯ {m}\n"))
            else:
                out.append(("", f"    {m}\n"))
        return out

    kb = KeyBindings()

    @kb.add("up")
    def _up(event):
        state["i"] = (state["i"] - 1) % len(chat_models)
        event.app.invalidate()

    @kb.add("down")
    def _down(event):
        state["i"] = (state["i"] + 1) % len(chat_models)
        event.app.invalidate()

    @kb.add("enter")
    def _accept(event):
        event.app.exit(result=chat_models[state["i"]])

    @kb.add("escape", eager=True)
    def _esc(event):
        event.app.exit(result=None)

    @kb.add("c-c")
    def _cancel(event):
        event.app.exit(result=None)

    control = FormattedTextControl(render, focusable=True, show_cursor=False)
    layout = Layout(Window(content=control, wrap_lines=False))
    picker_style = Style.from_dict({
        "title": "bold fg:ansicyan",
        "hint": "fg:ansibrightblack",
        "selected": "bold fg:ansiyellow",
    })
    app = Application(layout=layout, key_bindings=kb, style=picker_style, full_screen=True)
    return app.run()


class TypoSuggest:
    """デバウンス付きで AI 判定を行い、下部ツールバーにサジェストを表示する。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.words = load_words(args.words)
        self.vocab = set(self.words)
        self.system_prompt = build_system_prompt(self.words)
        self.client = AsyncOpenAI(base_url=args.base_url, api_key="lm-studio", timeout=30.0)
        self.model = args.model
        self.debounce = args.debounce

        self.current_suggestion: str | None = None
        self.status = "入力してください（1秒で判定します）"
        self._task = None  # デバウンス/判定タスク

        self.session = PromptSession()
        # バッファのテキスト変更を購読（プロンプト実行中に発火）
        self.session.default_buffer.on_text_changed += self._on_text_changed
        self.kb = self._build_keybindings()

    # ------------------------------------------------------------------
    # UI 構築
    # ------------------------------------------------------------------
    def _build_keybindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add(Keys.Tab)
        def _accept(event):
            # サジェストがあれば入力を置換して確定
            if self.current_suggestion:
                b = event.app.current_buffer
                b.text = self.current_suggestion
                b.cursor_position = len(b.text)
                self.current_suggestion = None
                self.status = "✔ 確定しました"
                self._invalidate()

        return kb

    def _invalidate(self) -> None:
        try:
            self.session.app.invalidate()
        except Exception:
            pass

    def _toolbar(self):
        if self.current_suggestion:
            return FormattedText([
                ("class:suggest", f"💡 もしかして: {self.current_suggestion}   [Tab=確定]")
            ])
        return FormattedText([("class:status", self.status)])

    # ------------------------------------------------------------------
    # デバウンス & 判定
    # ------------------------------------------------------------------
    def _on_text_changed(self, *_args) -> None:
        # 前のタスクが残っていればキャンセル（新しい入力でリセット）
        if self._task is not None and not self._task.done():
            self._task.cancel()
        text = self.session.default_buffer.text
        self.current_suggestion = None
        self.status = "入力中…" if text else "入力してください（1秒で判定します）"
        self._invalidate()
        try:
            app = self.session.app
        except Exception:
            return
        # プロンプト実行中のみスケジュール
        self._task = app.create_background_task(self._debounce_and_check(text))

    async def _debounce_and_check(self, captured: str) -> None:
        try:
            await asyncio.sleep(self.debounce)
        except asyncio.CancelledError:
            return

        # このタスクが最新の入力に対応しているか確認
        try:
            current = self.session.default_buffer.text
        except Exception:
            return
        if current != captured:
            return

        text = captured.strip()
        if not text:
            self.current_suggestion = None
            self.status = "入力してください（1秒で判定します）"
            self._invalidate()
            return

        # 高速パス: 完全一致なら正解
        if text in self.vocab:
            self.current_suggestion = None
            self.status = "✔ 正解（語彙に一致）"
            self._invalidate()
            return

        # AI 判定
        self.status = f"AI 判定中… ({self.model})"
        self.current_suggestion = None
        self._invalidate()
        try:
            suggestion = await judge_typo(
                self.client, self.model, text,
                self.system_prompt, self.vocab, timeout=15.0,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.status = f"AIエラー: {self._short(e)}"
            self.current_suggestion = None
        else:
            if suggestion:
                self.current_suggestion = suggestion
                self.status = "ミスの疑い → サジェスト表示中"
            else:
                self.current_suggestion = None
                self.status = "タイポの可能性は低そうです（OK）"
        self._invalidate()

    @staticmethod
    def _short(e: Exception) -> str:
        msg = str(e).replace("\n", " ")
        return (msg[:80] + "…") if len(msg) > 80 else msg

    # ------------------------------------------------------------------
    # メインループ
    # ------------------------------------------------------------------
    async def run(self) -> None:
        while True:
            try:
                result = await self.session.prompt_async(
                    FormattedText([("class:prompt", "🔍 入力> ")]),
                    bottom_toolbar=self._toolbar,
                    key_bindings=self.kb,
                    style=STYLE,
                )
            except (EOFError, KeyboardInterrupt):
                print()
                break

            suggestion = self.current_suggestion
            print_formatted_text(FormattedText([
                ("class:prompt", "  送信: "),
                ("", result),
                ("class:suggest", f"   （サジェスト: {suggestion}）" if suggestion else ""),
            ]))
            # 次の入力に向けてリセット
            self.current_suggestion = None
            self.status = "入力してください（1秒で判定します）"

        print("終了します。")


# ----------------------------------------------------------------------
# エントリポイント
# ----------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description="タイポサジェスト TUI（LMStudio/Gemma2 2b）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--words", default=DEFAULT_WORDS, help="読み込む語彙ファイル")
    p.add_argument("--model", default=None, help="LMStudio のモデルID（省略時は起動時に一覧から選択）")
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LMStudio の API URL")
    p.add_argument("--debounce", type=float, default=DEFAULT_DEBOUNCE, help="無入力判定までの秒数")
    p.add_argument("--check", action="store_true", help="起動診断のみ行い終了")
    p.add_argument("--once", metavar="INPUT", help="1回だけ判定して終了（非対話）")
    args = p.parse_args()

    # --- 語彙読込 -------------------------------------------------------
    try:
        words = load_words(args.words)
    except FileNotFoundError:
        print(f"エラー: 語彙ファイルが見つかりません: {args.words}", file=sys.stderr)
        print("先に `python extract_words.py N` を実行して tmp.txt を作成してください。", file=sys.stderr)
        sys.exit(1)
    char_count = sum(len(w) for w in words)
    print(f"語彙ファイル: {args.words}（{len(words)} 語 / {char_count} 文字を読み込み）")

    # --- モデル一覧取得 -------------------------------------------------
    try:
        models = list_models(args.base_url)
        conn_error = None
    except Exception as e:
        models = None
        conn_error = e

    if models is None:
        print(f"⚠ 警告: LMStudio({args.base_url}) に接続できません: {conn_error}", file=sys.stderr)
    else:
        print(f"LMStudio ロード済みモデル: {', '.join(models) if models else '（なし）'}")

    fallback = os.environ.get("LMSTUDIO_MODEL", DEFAULT_MODEL)
    effective_model = args.model  # --model 未指定なら None

    # --- 診断モード -----------------------------------------------------
    if args.check:
        effective_model = effective_model or fallback
        if models is not None and effective_model not in models:
            print(f"⚠ 警告: 使用予定モデル '{effective_model}' はロードされていません。", file=sys.stderr)
        print(f"使用予定モデル: {effective_model}")
        print(f"システムプロンプト概算長: {len(build_system_prompt(words))} 文字")
        print("目安: Gemma2 2b（コンテキスト約8k）では 数百語程度までが無難です。")
        return

    # --- 1回だけ判定モード（非対話: 選択UIは出さない）------------------
    if args.once:
        effective_model = effective_model or fallback
        if models is not None and effective_model not in models:
            print(f"⚠ 警告: 使用モデル '{effective_model}' はロードされていません。", file=sys.stderr)
        system_prompt = build_system_prompt(words)
        client = AsyncOpenAI(base_url=args.base_url, api_key="lm-studio", timeout=60.0)
        try:
            suggestion = asyncio.run(judge_typo(
                client, effective_model, args.once, system_prompt, set(words), timeout=30.0,
            ))
        except Exception as e:
            print(f"AI呼び出しでエラー: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"入力: {args.once}")
        print(f"サジェスト: {suggestion if suggestion else '（なし／正解または該当なし）'}")
        return

    # --- 対話モード -----------------------------------------------------
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print(
            "このアプリは対話型ターミナル（TTY）での実行が必要です。"
            " 通常のターミナル／コマンドプロンプトで起動してください。",
            file=sys.stderr,
        )
        print("非対話で1回だけ判定する場合は `--once <入力>` が使えます。", file=sys.stderr)
        sys.exit(1)

    if not models:
        print(
            "LMStudio からモデル一覧を取得できませんでした（サーバー未起動／モデル未ロード）。"
            " LMStudio でモデルをロードしてから再実行してください。",
            file=sys.stderr,
        )
        sys.exit(1)

    # 起動時にモデルを選択（--model 指定時はそれを優先して使用）
    try:
        if effective_model is None:
            effective_model = choose_model_interactive(models, preselect=fallback)
            if effective_model is None:
                print("モデル選択がキャンセルされました。終了します。")
                return
            print(f"選択されたモデル: {effective_model}")
        elif effective_model not in models:
            print(f"⚠ 警告: 指定モデル '{effective_model}' はロードされていませんが続行します。", file=sys.stderr)

        args.model = effective_model
        app = TypoSuggest(args)
        asyncio.run(app.run())
    except _NoConsoleError:
        print(
            "プロンプトの表示に必要な Windows コンソールを取得できませんでした。\n"
            "  - cmd.exe / PowerShell / Windows Terminal で実行してください。\n"
            "  - Git Bash / MinTTY の場合は `winpty python tui_app.py` で起動してください。",
            file=sys.stderr,
        )
        sys.exit(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
