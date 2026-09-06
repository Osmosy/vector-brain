<div align="center">

<img src="assets/vector-brain-logo.png" alt="Vector Brain" width="180"/>

# Vector Brain

[![Architecture: live](https://img.shields.io/badge/Architecture-live_diagram-4f8ff7.svg)](https://osmosy.github.io/vector-brain/docs/vector-brain.dataflow.html)

**Локальная база знаний на zvec — «SQLite среди векторных БД». Гибридный поиск (семантика + BM25) над markdown-знаниями: память агента, Obsidian, навыки**

[![Hermes Agent](https://img.shields.io/badge/Hermes-Agent-blue.svg)](https://github.com/NousResearch/hermes-agent)
[![Ecosystem: Vector](https://img.shields.io/badge/Ecosystem-Vector-blue.svg)](https://osmosy.github.io/)
[![DB: zvec](https://img.shields.io/badge/DB-zvec-orange.svg)](https://github.com/alibaba/zvec)
[![Embeddings: Ollama](https://img.shields.io/badge/Embeddings-Ollama-green.svg)](https://ollama.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

Инкрементальный индексатор markdown-знаний в [zvec](https://github.com/alibaba/zvec)
(встроенная векторная БД Alibaba Tongyi Lab, Apache 2.0) + CLI гибридного поиска.
Коллекция — просто папка на диске; WAL даёт crash-safety; никакого сервера.

### Журнал версий

| Дата | Что обновлено |
|------|---------------|
| 2026-09-06 | init: 1018 файлов (long_memory + Obsidian vault + 917 SKILL.md) → 22.3k чанков; WeightedReRanker 0.5/0.5; optimize() в цикле записи; Archify-диаграмма |

## Архитектура

```
markdown-источники (long_memory, vault, skills)
        │  SHA1-хэши, чанки 500 слов
        ▼
   Indexer ──► Ollama (nomic-embed 768d)
        │
        ▼
   zvec: WAL (fsync) → сегменты → HNSW + BM25 (RocksDB)
        │
        ▼
   zq CLI ──► WeightedReRanker 0.5/0.5 ──► ответы
```

Полная живая диаграмма: [docs/vector-brain.dataflow.html](docs/vector-brain.dataflow.html)
(5 стадий, guided views, dark/light, экспорт PNG/SVG).

## Компоненты

| Файл | Назначение |
|------|-----------|
| `config.py` | Пути, источники, модель эмбеддингов, параметры чанкировки |
| `embedder.py` | Эмбеддер через Ollama (`/api/embed`, пакетный режим) |
| `chunker.py` | Чанкировка markdown: по заголовкам, затем словные чанки с перекрытием |
| `indexer.py` | Инкрементальный индексатор (SHA1-хэши файлов), `optimize()` после каждого прогона |
| `zq.py` | CLI гибридного поиска: Weighted-фьюжн вектор + BM25, фильтры по source/date |
| `lp_fetch.py` | Читалка страниц через Lightpanda CDP (служебная) |

## Схема коллекции

- Скалярные: `path` (inverted), `title`, `source` (inverted), `date`, `section`, `body` (FTS)
- Вектор: `emb` FP32 768d, HNSW COSINE (m=50, ef_construction=500)

## Быстрый старт

```bash
git clone https://github.com/Osmosy/vector-brain.git
cd vector-brain

# Зависимости: локальный Ollama + модель nomic-embed-text (или своя в config.py)
uv venv .venv && uv pip install --python .venv/bin/python zvec requests

# Пропиши свои источники в config.py (SOURCES), затем:
.venv/bin/python indexer.py          # первый прогон создаст коллекцию
.venv/bin/python indexer.py          # повторный — ничего не изменит (инкрементальность)

# Поиск
.venv/bin/python zq.py "запрос" -k 5
.venv/bin/python zq.py "запрос" --source memory --date ">= 2026-09-01"
```

## Ключевые решения и грабли

- **WeightedReRanker 0.5/0.5 вместо RRF** — RRF вытягивает наверх шаблонные
  чанки-заглушки («Ключевые решения», «Ошибки и выводы»); weighted-фьюжн
  уравновешивает векторную и BM25-ветки и даёт чистый топ.
- **Сырой score векторного запроса — косинусное РАССТОЯНИЕ**: меньше = ближе.
  Пороги релевантности отсчитывать снизу.
- **Фильтры zvec**: равенство — одиночный `=` (не `==`).
- **id документа**: regex `^[a-zA-Z0-9_!@#$%+=.-]{1,64}$` — двоеточия и
  длинные пути запрещены; используем стабильный хэш-формат `{source}-{sha1[:12]}-{n}`.
- **Имя коллекции**: `^[a-zA-Z0-9_-]{3,64}$`.
- **`optimize()` обязателен**: новые векторы копятся в flat-буфер, HNSW
  достраивается только optimize() (фон, не блокирует чтение/запись).
  Прогресс: `collection.stats` → `index_completeness`.
- **Один писатель**: индексация одним процессом; многопроцессное чтение — нативно.
- **Модель эмбеддингов фиксируется до старта**: смена = полная переиндексация.

## Надёжность

```
insert → WAL (fsync) → flat-буфер пишущего сегмента (64 МБ)
        → seal сегмента → Arrow IPC + индексы
optimize() → flat-буфер → HNSW (фоновая достройка, не блокирует чтение/запись)
crash → WAL-реплей при следующем open()
```

WAL гарантирует: kill -9 процесса и перезагрузка ноутбука не теряют данные.
Tombstones удалений выпиливаются optimize-компакцией. Читателей много,
писатель один; коллекция = папка на диске — бэкап это `cp -r`.

## Экосистема Vector

| Проект | Что это |
|--------|---------|
| [Vector Work](https://github.com/Osmosy/vector-work) | Хаб экосистемы: 17 доменов Cowork, 212 навыков |
| [Vector Wallet](https://github.com/Osmosy/vector-wallet) | Форк Keepass2Android с крупной клавиатурой |
| [Vector Prediction](https://github.com/Osmosy/vector-prediction) | Гибридный прогнозный пайплайн |

## Лицензия

MIT. zvec — Apache 2.0 (Alibaba).