# -*- coding: utf-8 -*-
"""
LMStudio (OpenAI 互換) 上の Gemma2 2b を使ったタイポ判定ロジック。

- build_system_prompt(words): 語彙全件を埋め込んだシステムプロンプトを生成。
- judge_typo(...): 入力文字列がタイポかを AI に判定させ、意図された語彙を返す（なければ None）。

出力形式（小モデルでも堅牢になるよう最小化）:
  タイポなら   ->  SUGGEST: <単語>
  そうでないなら ->  OK
"""
from __future__ import annotations

import difflib
import re
from typing import Iterable, Optional, Sequence

# AI への推論パラメータ
TEMPERATURE = 0.1
MAX_TOKENS = 32
STOP = ["\n"]


def build_system_prompt(words: Sequence[str]) -> str:
    """語彙リストを埋め込んだシステムプロンプトを構築する。

    同一文字列を毎回送ることで、LMStudio のプレフィックスキャッシュが効き、
    繰り返し呼び出しでも高速に推論できる。
    """
    vocab_block = ", ".join(words)
    return (
        "あなたは日本語のタイポ（打ち間違い）検出アシスタントです。\n"
        "以下は【正しい単語の語彙リスト】です。\n"
        f"語彙: {vocab_block}\n\n"
        "ユーザーが入力した文字列を確認してください。\n"
        "それが上記語彙のいずれかの打ち間違い（タイポ）であると思われる場合は、"
        "最も意図されたであろう語彙中の単語を1つだけ選び、次の形式で**1行だけ**出力してください。\n"
        "  SUGGEST: <単語>\n"
        "入力がすでに正しい単語である場合、または語彙に近い単語が見当たらない場合は、次のように出力してください。\n"
        "  OK\n"
        "例:\n"
        "- 入力「まっしゅ」はマッシュの打ち間違いなので → SUGGEST: マッシュ\n"
        "- 入力「カット」は語彙の正しい単語なので → OK\n"
        "ルール:\n"
        "- 出力は必ず上記いずれか1行のみ。\n"
        "- 説明・理由・挨拶・JSON・複数行の出力は一切しない。\n"
        "- <単語> は必ず上記語彙の中から選ぶ。"
    )


_SUGGEST_RE = re.compile(r"SUGGEST\s*[:：]\s*(.+)")
# 末尾につきそうな記号を除去
_STRIP_CHARS = "「」『』\"'()（）[]【】<＜>＞、。.,;；:：!！?？\n\r\t "

# ハルシネーション除外用: 提案語と入力の最低類似度（ひらがな正規化後）
SIMILARITY_THRESHOLD = 0.5


def _to_hiragana(s: str) -> str:
    """カタカナをひらがなに正規化（へあさろん ↔ ヘアサロン の比較用）。"""
    out = []
    for ch in s:
        code = ord(ch)
        if 0x30A1 <= code <= 0x30F6:  # ァ〜ン
            out.append(chr(code - 0x60))
        else:
            out.append(ch)
    return "".join(out)


def _similar_enough(a: str, b: str, threshold: float = SIMILARITY_THRESHOLD) -> bool:
    """2つの文字列がタイポとして許容できる程度に似ているか。"""
    na, nb = _to_hiragana(a), _to_hiragana(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:  # 片方が他方の部分文字列（短い入力の補完等）
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= threshold


def parse_suggestion(raw: str, vocab: Iterable[str], user_input: str = "") -> Optional[str]:
    """AI の生テキスト出力を解析し、語彙に存在する提案語を返す（なければ None）。

    ハルシネーション防止のため、最終的に「入力と似ている」提案語のみ返す。
    """
    raw = (raw or "").strip()
    if not raw:
        return None

    vocab_set = set(vocab)
    ui = (user_input or "").strip()

    candidate = None
    m = _SUGGEST_RE.search(raw)
    if m:
        word = m.group(1).strip().strip(_STRIP_CHARS)
        # 空白以降（説明等）を切り捨て
        word = word.split()[0] if word.split() else word
        word = word.strip(_STRIP_CHARS)
        if not word:
            return None
        if word in vocab_set:
            candidate = word
        else:
            # モデルが語彙外の語を返した場合、最も近い語彙に寄せる
            close = difflib.get_close_matches(word, list(vocab_set), n=1, cutoff=0.7)
            candidate = close[0] if close else None
    else:
        # フォールバック: プレフィックス無しでも「語彙の正しい単語そのもの」が
        # 返ってきた場合は、それを意図語とみなす（入力と異なる場合のみ）。
        word = raw.strip(_STRIP_CHARS)
        word = word.split()[0] if word.split() else word
        word = word.strip(_STRIP_CHARS)
        if word and word in vocab_set and word != ui:
            candidate = word

    if candidate is None:
        return None
    # ハルシネーション除外: 入力と似ていない提案（例: 無意味入力へのデタラメ）は採用しない
    if ui and not _similar_enough(ui, candidate):
        return None
    return candidate


async def judge_typo(
    client,
    model: str,
    user_input: str,
    system_prompt: str,
    vocab: Iterable[str],
    *,
    timeout: float = 15.0,
) -> Optional[str]:
    """入力がタイポなら意図された語彙を、そうでなければ None を返す。

    例外やタイムアウト時は None を返さず伝播させる（呼び出し元でハンドリング）。
    """
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        stop=STOP,
        timeout=timeout,
    )
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return None
    content = choices[0].message.content or ""
    return parse_suggestion(content, vocab, user_input)
