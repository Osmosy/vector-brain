# Zvec как локальная база знаний для всей работы

Дата: 2026-09-06. Статус: план, не внедрено.

## Идея

Сейчас знания разбросаны по разным хранилищам с разным поиском:
- long_memory (identity, knowledge-base, 32 daily-архива) — markdown, ищется глазами/grep
- Obsidian vault (~/.hermes/vault, 17 заметок) — autograph-схема
- skills (916 SKILL.md) — выбор по имени/описанию, семантики нет
- сессии Hermes — FTS5-полнотекст, векторов нет
- знания в голове агента (MEMORY) — 5000 символов, горит лимит

Zvec закрывает это одной in-process БД: гибридный поиск (вектор + BM25 + фильтры)
над всеми источниками, без серверов. Перезагрузка ноутбука ничего не ломает —
это папки на диске, WAL-гарантия.

## Архитектура

```
данные-источники          индексация            коллекции zvec
─────────────────         ─────────────         ──────────────
long_memory/**/*.md  ─┐
vault/**/*.md        ─┼─►  инкрементальный  ─►  ~/zvec/knowledge   (гибрид)
skills/*/SKILL.md    ─┘    индексатор
сессии (state.db)    ────  (по mtime/hash)  ─►  ~/zvec/sessions
```

Одна директория `~/zvec/`, 2–3 коллекции. Копия = бэкап всей базы знаний.

## Коллекция knowledge (главная)

Схема:

```python
schema = zvec.CollectionSchema(
    name="knowledge",
    fields=[
        zvec.FieldSchema("path",    zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("title",   zvec.DataType.STRING),
        zvec.FieldSchema("source",  zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
        # source: memory | vault | skill | session | journal
        zvec.FieldSchema("date",    zvec.DataType.STRING),
        zvec.FieldSchema("tags",    zvec.DataType.STRING, index_param=zvec.InvertIndexParam()),
        zvec.FieldSchema("body",    zvec.DataType.STRING, index_param=zvec.FtsIndexParam()),
    ],
    vectors=zvec.VectorSchema("emb", zvec.DataType.VECTOR_FP32, 768,
        index_param=zvec.HnswIndexParam(metric_type=zvec.MetricType.COSINE)),
)
```

Чанкировка: заголовок+путь+дата+теги — в скалярные поля, тело режем на
~500-словные чанки с перекрытием, каждый чанк — Doc с id `{source}:{path}:{n}`.

Эмбеддинги: Ollama `nomic-embed-text` (768d, 274 МБ, уже стоит). Альтернатива
для скорости — `all-minilm` (384d), тогда размерность в схеме 384. Модель
фиксируется на этапе создания коллекции; смена модели = переиндексация.

Запросы — гибрид из коробки:

```python
col.query(queries=[
    zvec.Query(field_name="emb",  vector=embed(q)),
    zvec.Query(field_name="body", fts=zvec.Fts(match_string=q)),
], topk=10, reranker=zvec.RrfReRanker(), filter="source == 'memory'")
```

## Что это даёт на каждом сценарии

1. **Память агентов.** Вместо «лимит 5000 символов + prune» — «найди релевантное»:
   каждый факт попадает в knowledge с source='memory', агент перед задачей
   делает top-10 гибридный запрос по теме. Гибрид важен: точные термины
   (порты, имена навыков) ловит BM25, перефразировки — вектор.
2. **Выбор навыков.** 916 SKILL.md → чанк = заголовок+описание. Запрос «как
   диагностировать зависание gfx1150» → skill linux-kernel-crash-diagnosis
   семантически, а не по совпадению слов в имени.
3. **Vault-поиск.** Заметки с тегами как скалярным полем: фильтр по тегу +
   вектор по смыслу. Wiki-links сохраняем в tags.
4. **Журнал/архив.** daily-архивы с датой в скалярном поле: «что мы решали в
   марте про 1С» = filter `date >= '2026-03-01' and date < '2026-04-01'` +
   векторный смысл. Это то, что session_search делает текстовым FTS, но со
   смысловой близостью.
5. **Session-поиск** — отдельная коллекция sessions, если зайдёт.

## Инкрементальный индексатор

Скрипт `~/projects/zvec-db/indexer.py`:
- обходит источники, считает (path, mtime, size);
- в коллекции хранить хэш в скалярном поле и делать upsert по id `{source}:{path}:{n}`;
- новый/изменённый файл → delete_by_filter по path + заново чанки;
- удалённый файл → delete_by_filter;
- после bulk-загрузки — `col.optimize()` (компакция выпиливает tombstones).

Правило единственного писателя: индексация — один процесс по расписанию
(cronjob), агенты — только читатели. Многопроцессное чтение zvec
поддерживает нативно.

## Обвязка

- CLI-хелпер `zq "запрос" [--source memory] [-k 10]`: печатает path/score/сниппет.
  Использовать и человеку, и агентам из терминала.
- Для агентов: скилл `zvec-search` в ~/.hermes/skills — как пользоваться
  `zq` и когда (перед незнакомой темой, при поиске прошлых решений).
- Cronjob: индексация каждые N часов, тихая (deliver='local').

## Ресурсы

- Диск: 916 навыков + 49 md-файлов → порядка 50–150 тыс. чанков worst case;
  768d FP32 = 3 КБ/чанк → ~0.5 ГБ. С INT8-квантованием втрое меньше.
- RAM: mmap, греется по мере запросов; HNSW-граф в памяти — единицы сотен МБ.
  На 32 ГБ незаметно.
- Ollama: ~10 мс/чанк на CPU для nomic; полная переиндексация — минуты.

## Ограничения, о которых помним

- COSINE возвращает косинусное РАССТОЯНИЕ (меньше = ближе). Пороги — снизу.
- Запись — один процесс. Все записи через индексатор, не из агентов напрямую.
- Имя коллекции: regex `^[a-zA-Z0-9_-]{3,64}$`.
- Смена embedding-модели = полная переиндексация. Выбор модели — до старта.
- update в zvec = delete+insert; для knowledge это норм (файл изменился —
  перечанковали), для транзакционных данных — не то.

## Порядок внедрения (пилот)

1. venv `~/.venvs/zvec`, скелет `~/projects/zvec-db/` (indexer.py, zq.py, config).
2. Пилот только на long_memory (49 файлов — малый контур): проверить качество
   гибридного поиска руками на 5–6 реальных запросах из практики
   («зависание gfx1150», «git-credentials истекает», «1С расширение CFE»).
3. Если поиск попадает — подключить vault и skills (916 — уже объём, тут
   эффект заметнее всего).
4. cronjob индексации + скилл для агентов.
5. Решение о sessions-коллекции — по результатам пилота.