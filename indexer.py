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
    """[(source, abs_path, rel_path)] по всем источникам.

    Симлинки: один и тот же файл может быть достижим по двум путям (каталог-симлинк
    верхнего уровня + реальный каталог в наборе, напр. `writing-skills/` ->
    `superpowers/skills/writing-skills/`). Без дедупликации такой файл индексируется
    дважды под разными id. Оставляем один путь — самый короткий (канонический).

    Из-за дедупликации файл может быть доступен только по каноническому пути: путь-
    симлинк в индексе не появляется, и его чанки не создаются. Проверку «файл без
    чанков» это не путает — realpath совпадает с каноническим (см. collection_paths).
    """
    seen: dict[str, tuple[int, str, str]] = {}   # realpath -> (глубина, source, abs_path)
    order: list[str] = []
    for name, root, pattern in SOURCES:
        root = os.path.expanduser(root)
        for p in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
            if not os.path.isfile(p):
                continue
            real = os.path.realpath(p)
            depth = os.path.relpath(p, root).count(os.sep)
            if real in seen:
                if depth < seen[real][0]:
                    seen[real] = (depth, name, p)
                continue
            seen[real] = (depth, name, p)
            order.append(real)

    out = []
    for real in order:
        _, name, p = seen[real]
        root = os.path.expanduser(dict((n, r) for n, r, _ in SOURCES)[name])
        out.append((name, p, os.path.relpath(p, root)))
    return out


def make_doc_id(source: str, rel_path: str, n: int) -> str:
    # ВСЕГДА хэш-формат: стабилен между прогонами, не зависит от длины пути.
    h = hashlib.sha1(f"{source}:{rel_path}".encode()).hexdigest()[:12]
    base = f"{source}-{h}-{n}"
    return base[:64] if len(base) <= 64 else f"{source}-{h}-{n}"[:64]


def escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def open_collection_with_retry(knowledge_path: str, create: bool = False,
                               attempts: int = 6, delay: float = 10.0):
    """Открыть коллекцию на запись, переждав чужой LOCK.

    Коллекция — однописательская: если её держит другой процесс (zvec-studio
    держит write-lock, пока к нему обращаются из UI), zvec.open падает с
    "Can't lock read-write collection". Раньше indexer.py в этом случае просто
    выходил с кодом 1 — то есть прогон по расписанию молча терялся.
    """
    last: Exception | None = None
    for i in range(attempts):
        try:
            return (
                zvec.create_and_open(path=knowledge_path, schema=SCHEMA)
                if create or not os.path.exists(knowledge_path)
                else zvec.open(knowledge_path)
            )
        except RuntimeError as e:
            last = e
            if "lock" not in str(e).lower():
                raise
            if i == attempts - 1:
                break
            print(f"коллекция занята другим процессом, повтор через {delay:.0f}с "
                  f"({i + 1}/{attempts - 1})", flush=True)
            time.sleep(delay)
    raise last


def collection_paths(col) -> set[str]:
    """Пути, у которых в коллекции есть хотя бы один чанк."""
    return {d.fields.get("path") for d in col.iter_docs()}


def doc_prefix(source: str, rel_path: str) -> str:
    """Префикс id всех чанков файла: `source-<sha1(source:rel)[:12]>-`.

    Ровно его строит make_doc_id, поэтому «свежесть» файла проверяется по id, а не
    по полю path: чанки, записанные старым (до-хэшевым) форматом id, лежат по тому же
    пути, но содержат устаревший текст — по path файл выглядел бы проиндексированным.
    """
    return make_doc_id(source, rel_path, 0).rsplit("-", 1)[0] + "-"


def collection_prefixes(col) -> set[str]:
    """id-префиксы чанков, реально доступных в коллекции (iter_docs их видит)."""
    return {d.id.rsplit("-", 1)[0] + "-" for d in col.iter_docs()}


def missing_files(col, files: list[tuple[str, str, str]]) -> list[str]:
    """Файлы источников, у которых в коллекции нет ни одного актуального чанка.

    Ловит три расхождения сразу: чанки не создавались, чанки записаны старым
    форматом id, и «скрытые» чанки (doc_id существует, но iter_docs его не выдаёт —
    так выглядят записи, удалённые во время реорганизации симлинков; лечится upsert).
    """
    have = collection_prefixes(col)
    return [rel for source, _, rel in files if doc_prefix(source, rel) not in have]


def cleanup_orphans(col, files: list[tuple[str, str, str]]) -> int:
    """Удалить из коллекции пути, которых больше нет среди источников.

    Осиротевшие чанки берутся не из state.json (в нём их может уже не быть), а из
    самой коллекции: старый id-формат, чужой писатель, потерянный префикс состояния
    — всё это оставляет чанки по несуществующим путям, которые иначе лежат вечно и
    попадают в выдачу. Файлы, проигравшие симлинк-дедуп, не осиротевшие: их realpath
    совпадает с каноническим путём в наборе.

    Отдельный режим, не часть прогона: при частично недоступном источнике
    (не смонтирован, нет прав) автоматическая чистка снесла бы живой индекс.
    """
    alive = {p for _, p, _ in files} | {os.path.realpath(p) for _, p, _ in files}
    orphans = sorted(p for p in collection_paths(col) if p and p not in alive)
    for path in orphans:
        col.delete_by_filter(f"path = '{escape(path)}'")
    if orphans:
        col.optimize()
    print(f"осиротевших путей: {len(orphans)}, удалено чанков — см. stats: {col.stats}")
    for path in orphans[:10]:
        print(f"   {path}")
    return len(orphans)


def index_collection(create: bool = False) -> None:
    emb = OllamaEmbedder()
    state = load_state()
    knowledge_path = os.path.expanduser(KNOWLEDGE_PATH)
    col = open_collection_with_retry(knowledge_path, create)

    files = collect_files()
    new_state = {f"{source}:{rel}": file_hash(p) for source, p, rel in files}
    n_reindexed, n_new_chunks, n_deleted = 0, 0, 0

    # Сверка состояния с коллекцией: файл, числящийся неизменным, может не иметь
    # актуальных чанков (не создавались; записаны старым форматом id; «скрыты» после
    # реорганизации симлинков). Без сверки он не индексируется никогда: хэш совпадает,
    # ветка «не изменился» его пропускает, а state уже отмечен как проиндексированный.
    # Один прогон — один обход коллекции.
    present = collection_paths(col)
    prefixes = collection_prefixes(col)
    if not create:
        n_stale = sum(1 for source, _, rel in files
                      if doc_prefix(source, rel) not in prefixes)
        if n_stale:
            print(f"сверка: файлов без актуальных чанков — {n_stale}, "
                  f"переиндексирую", flush=True)

    for source, path, rel in files:
        key = f"{source}:{rel}"
        if state.get(key) == new_state[key]:
            if doc_prefix(source, rel) in prefixes:
                continue  # не изменился и чанки актуальны

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
        # upsert, не insert: у документа может остаться скрытая tombstone-запись
        # (doc_id уже существует, но ни delete_by_filter, ни iter_docs его не видят —
        # так выглядят чанки, удалённые до/во время реорганизации симлинков). insert
        # на такой id возвращает code=2, файл остаётся без чанков навсегда, а state
        # уже записал хэш — то есть «индексирован, но недоступен». upsert перекрывает.
        results = col.upsert(docs)
        failed = [r for r in (results or []) if isinstance(r, dict) and r.get("code") not in (None, 0)]
        n_new_chunks += len(docs) - len(failed)
        n_reindexed += 1
        if failed:
            print(f"предупреждение: {len(failed)} из {len(docs)} чанков не записались "
                  f"({rel}): {failed[0].get('message', '')[:120]}", flush=True)

    # 4. Файлы, исчезнувшие из источников — tombstones; подчистит optimize.
    # ВАЖНО: удалённый файл недостаточно убрать из state — его чанки остаются в
    # коллекции навсегда, если не сделать delete_by_filter по сохранённому пути.
    gone = set(state) - set(new_state)
    for key in gone:
        source, rel = key.split(":", 1)
        root = os.path.expanduser(dict((n, r) for n, r, _ in SOURCES).get(source, ""))
        if not root:
            continue
        path = os.path.join(root, rel)
        col.delete_by_filter(f"path = '{escape(path)}' and source = '{source}'")
    if gone:
        n_deleted += len(gone)

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

    # Самопроверка по актуальным id-префиксам: у каждого файла источника должны быть
    # свежие чанки. «Симлинк-проигравшие» сюда не попадают — у них свой префикс, и
    # чанки есть, просто путь в поле path канонический.
    without = missing_files(col, files)
    if without:
        print(f"ВНИМАНИЕ: без актуальных чанков осталось {len(without)}: "
              f"{', '.join(without[:5])}")
    else:
        print(f"проверка: все {len(files)} файлов источников имеют актуальные чанки")


if __name__ == "__main__":
    if "--cleanup-orphans" in sys.argv:
        # Отдельный режим: удалить чанки по путям, которых нет в источниках.
        _col = open_collection_with_retry(os.path.expanduser(KNOWLEDGE_PATH))
        cleanup_orphans(_col, collect_files())
    else:
        index_collection(create="--create" in sys.argv)
