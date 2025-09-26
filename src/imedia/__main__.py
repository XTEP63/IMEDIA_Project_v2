from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
from .raw_extractor import RawExtractor
from .bronze_transformer import BronzeTransformer
from .silver_normalizer import SilverNormalizer

def main():
    ap = argparse.ArgumentParser()
    # modos exclusivos
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--subreddit", help="Subreddit específico (ej. python)")
    group.add_argument("--discover-hot", type=int, metavar="N", help="Descubre N subreddits calientes")

    ap.add_argument("--hot-strategy", default="popular", choices=["popular", "all_hot", "all_top_day"],
                    help="Estrategia de descubrimiento (solo con --discover-hot)")
    ap.add_argument("--include-nsfw", action="store_true", help="Incluir NSFW en descubrimiento")
    ap.add_argument("--limit", type=int, default=100, help="posts por subreddit")
    ap.add_argument("--time-filter", default="day", choices=["hour","day","week","month","year","all"])
    ap.add_argument("--fetch-comments", action="store_true", help="Descargar comments del primer post de cada sub")
    args = ap.parse_args()

    raw = RawExtractor()
    bronze = BronzeTransformer()

    raw_post_paths: list[Path] = []
    raw_sub_info_paths: list[Path] = []
    raw_comments_paths: list[Path] = []
    hotlist_raw_path: Path | None = None

    # ---- MODO 1: subreddit único ----
    if args.subreddit:
        # info + posts
        raw_sub_info_paths.append(raw.subreddit_info(args.subreddit))
        raw_post_paths.append(raw.top_posts(args.subreddit, limit=args.limit, time_filter=args.time_filter))

        # comments opcionales (primer post del lote)
        if args.fetch_comments:
            dfp = pl.read_ndjson(raw_post_paths[-1])
            if dfp.height:
                first_post = dfp["id"][0]
                raw_comments_paths.append(raw.comments_of_post(first_post))

    # ---- MODO 2: descubrir N subreddits ----
    else:
        subs, hotlist_raw_path = raw.discover_subreddits(
            n=args.discover_hot,
            strategy=args.hot_strategy,
            posts_limit=max(200, args.limit * 5),   # suficiente muestra para rankear
            include_nsfw=args.include_nsfw,
        )
        if hotlist_raw_path:
            bronze.hotlist(hotlist_raw_path)

        # info de cada sub + posts por sub
        for s in subs:
            raw_sub_info_paths.append(raw.subreddit_info(s))
        raw_post_paths.extend(
            raw.top_hot_batch(subs, limit_per_sub=args.limit, time_filter=args.time_filter)
        )

        # comments opcionales (primer post de CADA sub)
        if args.fetch_comments:
            for p in raw_post_paths:
                dfp = pl.read_ndjson(p)
                if dfp.height:
                    raw_comments_paths.append(raw.comments_of_post(dfp["id"][0]))

    # ---- BRONZE ----
    for p in raw_sub_info_paths:
        bronze.subreddits(Path(p))
    for p in raw_post_paths:
        bronze.posts(Path(p))
    for p in raw_comments_paths:
        bronze.comments(Path(p))

    # ---- SILVER (+ SQLite) ----
    silver = SilverNormalizer()
    out = silver.build()
    print("OK → SILVER:", out)

if __name__ == "__main__":
    main()
