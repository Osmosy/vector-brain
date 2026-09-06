"""Инкрементальный индексатор long_memory -> zvec knowledge.

Хранит state.json (path -> hash). Изменённые файлы: delete_by_filter по
path+source, потом перечанковка и insert. Один писатель — по расписанию
или вручную. После bulk-удалений — optimize (компакция tombstones).
"""
import glob
import hashlib
import json
import os
import sys
import time

import zvec

import chunker
from config import EMBED_DIM, KNOWLEDGE_PATH, SOURCES
from embedder import OllamaEmbedder

# Глобальная конфигурация: ограничить потоки запросов и логи
zvec.init(
    log_type=zvec.LogType.CONSOLE,
    log_level=zvec.LogLevel.WARN,
    query_threads=2,
)

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")

SCHEMA = zvec.CollectionSchema(
    name="knowledge",
    fields=[
        zvec.FieldSchema("path",   zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("title",  zvec.DataType.STRING),
        zvec.FieldSchema("source", zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("date",   zvec.DataType.STRING),
        zvec.FieldSchema("section", zvec.DataType.STRING),
        zvec.FieldSchema("body",   zvec.DataType.STRING, index_param=zvec.FtsIndexParam()),
    ],
    vectors=zvec.VectorSchema(
        "emb", zvec.DataType.VECTOR_FP32, EMBED_DIM,
        index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE),
    ),
)


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


def file_hash(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def collect_files() -> list[tuple[str, str, str]]:
    """[(source, abs_path, rel_path)] по всем источникам."""
    out = []
    for name, root, pattern in SOURCES:
        root = os.path.expanduser(root)
        for p in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
            if os.path.isfile(p):
                out.append((name, p, os.path.relpath(p, root)))
    return out


def make_doc_id(source: str, rel_path: str, n: int) -> str:
    # ВСЕГДА хэш-формат: стабилен между прогонами, не зависит от длины пути.
    h = hashlib.sha1(f"{source}:{rel_path}".encode()).hexdigest()[:12]
    base = f"{source}-{h}-{n}"
    return base[:64] if len(base) <= 64 else f"{source}-{h}-{n}"[:64]


def escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def index_collection(create: bool = False) -> None:
    emb = OllamaEmbedder()
    state = load_state()
    knowledge_path = os.path.expanduser(KNOWLEDGE_PATH)
    col = (
        zvec.create_and_open(path=knowledge_path, schema=SCHEMA)
        if create or not os.path.exists(knowledge_path)
        else zvec.open(knowledge_path)
    )

    files = collect_files()
    new_state = {f"{source}:{rel}": file_hash(p) for source, p, rel in files}
    n_reindexed, n_new_chunks, n_deleted = 0, 0, 0

    for source, path, rel in files:
        key = f"{source}:{rel}"
        if state.get(key) == new_state[key]:
            continue  # не изменился

        # 1. Удалить старые чанки (в zvec-фильтрах равенство — одиночный '=')
        col.delete_by_filter(f"path = '{escape(path)}' and source = '{source}'")
        n_deleted += 1  # счёт файлов, не чанков

        # 2. Прочитать и чанковать
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        title = next(
            (l.lstrip("# ").strip() for l in text.splitlines() if l.strip().startswith("#")),
            rel,
        ) or rel
        mdate = time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(path)))
        chunks = chunker.chunk_markdown(text)
        if not chunks:
            chunks = [{"text": f"{title} (пустой файл)", "section": ""}]

        # 3. Эмбеддить и вставлять
        texts = [
            f"{title}\n[{c['section']}]\n{c['text']}" if c["section"] else f"{title}\n{c['text']}"
            for c in chunks
        ]
        vectors = emb.embed(texts)
        docs = [
            zvec.Doc(
                id=make_doc_id(source, rel, i),
                fields={
                    "path": path, "title": title, "source": source,
                    "date": mdate, "section": c["section"], "body": c["text"],
                },
                vectors={"emb": v},
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        col.insert(docs)
        n_new_chunks += len(docs)
        n_reindexed += 1

    # 4. Файлы, исчезнувшие из источников — tombstones; подчистит optimize
    gone = set(state) - set(new_state)

    save_state(new_state)

    # 5. Optimize: достроить HNSW для новых векторов (flat-буфер -> индекс,
    #    фон, не блокирует чтение/запись). После КАЖДОГО прогона с изменениями.
    if n_reindexed or n_deleted:
        t0 = time.time()
        col.optimize()
        st = col.stats
        print(f"optimize: {time.time() - t0:.1f}s | stats: {st}")

    print(f"файлов: {len(files)}, переиндексировано: {n_reindexed}, "
          f"новых чанков: {n_new_chunks}, изменённых файлов: {n_deleted}, "
          f"исчезло: {len(gone)}")


if __name__ == "__main__":
    index_collection(create="--create" in sys.argv)