# Конфигурация zvec-базы знаний.
# Модель эмбеддингов фиксируется ДО создания коллекции;
# смена модели = полная переиндексация (см. PLAN.md).

# Ollama
OLLAMA_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

# Пути — настрой под свою машину
DB_ROOT = "~/zvec"                             # коллекции zvec
KNOWLEDGE_PATH = DB_ROOT + "/knowledge"

# Источники для индексации (name, корневая директория, glob).
# Это рабочие пути Hermes по умолчанию: долговременная память, vault и навыки.
# Для другого набора замени на свои (например: ~/notes, ~/Obsidian).
SOURCES = [
    ("memory", "~/.hermes/long_memory", "**/*.md"),
    ("vault", "~/.hermes/vault", "**/*.md"),
    ("skill", "~/.hermes/skills", "**/SKILL.md"),
]

# Чанкировка
CHUNK_WORDS = 500          # слов на чанк
CHUNK_OVERLAP = 60         # слов перекрытия

# Поиск
TOPK = 10
RRF_K = 60