"""zq — CLI гибридного поиска по zvec knowledge.

Usage:
  zq "запрос" [-k N] [--source memory] [--date 2026-03] [--fts-only] [--vec-only]
"""
import argparse
import sys
import os

import zvec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
zvec.init(log_type=zvec.LogType.CONSOLE, log_level=zvec.LogLevel.WARN, query_threads=2)
from config import KNOWLEDGE_PATH, TOPK  # noqa: E402
from embedder import OllamaEmbedder  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Гибридный поиск по базе знаний")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=TOPK)
    ap.add_argument("--source", default=None, help="фильтр по source")
    ap.add_argument("--date", default=None, help='фильтр по date, напр. "2026-03" или ">= 2026-03-01"')
    ap.add_argument("--fts-only", action="store_true")
    ap.add_argument("--vec-only", action="store_true")
    ap.add_argument("--fusion", default="weighted", choices=["weighted", "rrf"],
                    help="weighted (по умолч., баланс vec/fts) или rrf")
    ap.add_argument("--full", action="store_true", help="печатать весь body")
    args = ap.parse_args()

    col = zvec.open(os.path.expanduser(KNOWLEDGE_PATH))

    queries = []
    if not args.fts_only:
        v = OllamaEmbedder().embed_one(args.query)
        queries.append(zvec.Query(field_name="emb", vector=v))
    if not args.vec_only:
        queries.append(zvec.Query(field_name="body", fts=zvec.Fts(match_string=args.query)))
    if not queries:
        queries = [zvec.Query(field_name="body", fts=zvec.Fts(match_string=args.query))]

    filters = []
    if args.source:
        filters.append(f"source == '{args.source}'")
    if args.date:
        op = ">=" if args.date.startswith(">") else "=="
        val = args.date.lstrip("><= ")
        filters.append(f"date {op} '{val}'")
    filt = " and ".join(filters) or None

    reranker = (
        zvec.RrfReRanker() if args.fusion == "rrf"
        else zvec.WeightedReRanker(weights=[0.5, 0.5])
    )
    results = col.query(queries=queries, topk=args.k, filter=filt, reranker=reranker)
    if not results:
        print("(нет результатов)")
        return

    for i, d in enumerate(results, 1):
        f = d.fields if hasattr(d, "fields") else d.get("fields", {})
        body = f.get("body", "") if isinstance(f, dict) else f.body
        snippet = body if args.full else body[:220].replace("\n", " ") + "…"
        get = (lambda k: f.get(k, "")) if isinstance(f, dict) else (lambda k: getattr(f, k))
        print(f"{i}. [{d.score:.3f}] {get('source')}:{os.path.basename(get('path'))}  «{get('section')}»")
        print(f"   {snippet}\n")


if __name__ == "__main__":
    main()