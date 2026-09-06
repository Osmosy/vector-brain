"""Чанкировка markdown по словам с перекрытием.

Стратегия: сначала пробуем резать по заголовкам (## / ###),
слишком большие секции докрашиваем словными чанками.
"""
import re

from config import CHUNK_OVERLAP, CHUNK_WORDS

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


def split_sections(text: str) -> list[tuple[str, str]]:
    """Разбивает md на (заголовок-контекст, текст секции)."""
    sections: list[tuple[str, str]] = []
    current_head, buf = "", []
    for line in text.splitlines():
        m = HEADING_RE.match(line)
        if m and len(m.group(1)) <= 3:  # # ## ### — границы секций
            if buf:
                sections.append((current_head, "\n".join(buf)))
            current_head, buf = m.group(2).strip(), [line]
        else:
            buf.append(line)
    if buf:
        sections.append((current_head, "\n".join(buf)))
    return sections if sections else [("", text)]


def _word_chunks(words: list[str]) -> list[str]:
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    out = []
    for i in range(0, len(words), step):
        chunk = words[i : i + CHUNK_WORDS]
        if len(chunk) < CHUNK_OVERLAP // 2 and out:
            break
        out.append(" ".join(chunk))
    return out


def chunk_markdown(text: str) -> list[dict]:
    """Возвращает [{'text': ..., 'section': ...}, ...]"""
    result = []
    for head, body in split_sections(text):
        words = body.split()
        if not words:
            continue
        if len(words) <= CHUNK_WORDS * 1.3:
            result.append({"text": body.strip(), "section": head})
        else:
            for i, c in enumerate(_word_chunks(words)):
                result.append({"text": c, "section": f"{head} [part {i + 1}]" if head else f"part {i + 1}"})
    return result