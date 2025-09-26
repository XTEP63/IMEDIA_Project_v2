from __future__ import annotations
from pathlib import Path
import polars as pl
from .config import settings
from .repo_sqlite import init_db, upsert_subreddits, upsert_authors, upsert_posts, upsert_comments

class SilverNormalizer:
    def _read_glob(self, rel: str) -> pl.DataFrame:
        paths = list((settings.bronze_dir / rel).glob("**/*.parquet"))
        return pl.concat([pl.read_parquet(p) for p in paths], how="vertical_relaxed") if paths else pl.DataFrame()

    def build(self) -> dict[str, Path]:
        posts_bz = self._read_glob("posts")
        comments_bz = self._read_glob("comments")
        subreddits_bz = self._read_glob("subreddits")
        authors_bz = self._read_glob("authors")  # opcional si existiera

        # ---------- Normalización ----------
        posts = (
            posts_bz
            .with_columns([
                pl.col("post_id").cast(pl.Utf8),
                pl.col("subreddit").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col("author").cast(pl.Utf8),
                pl.col("created_utc").cast(pl.Float64, strict=False),
                pl.col("score").cast(pl.Int64, strict=False),
                pl.col("num_comments").cast(pl.Int64, strict=False),
                pl.col("over_18").cast(pl.Boolean, strict=False),
            ])
            .unique(subset=["post_id"])
        )

        comments = (
            comments_bz
            .with_columns([
                pl.col("comment_id").cast(pl.Utf8),
                pl.col("post_id").cast(pl.Utf8),
                pl.col("author").cast(pl.Utf8),
                pl.col("created_utc").cast(pl.Float64, strict=False),
                pl.col("score").cast(pl.Int64, strict=False),
                pl.col("is_submitter").cast(pl.Boolean, strict=False),
            ])
            .unique(subset=["comment_id"])
        )

        dim_subreddit = (
            subreddits_bz
            .with_columns([
                pl.col("subreddit").cast(pl.Utf8).str.strip_chars().str.to_uppercase(),
                pl.col("subscribers").cast(pl.Int64, strict=False).fill_null(0),
                pl.col("created_utc").cast(pl.Float64, strict=False),
                pl.col("over18").cast(pl.Boolean, strict=False),
            ])
            .unique(subset=["subreddit"])
        )

        # Autores desde posts y comments (si no hay authors_bz)
        dim_author = (
            pl.concat(
                [
                    posts.select(pl.col("author").alias("author_name")),
                    comments.select(pl.col("author").alias("author_name")),
                ],
                how="vertical_relaxed",
            )
            .drop_nulls()
            .unique()
        )

        # Facts (FK por nombre ya normalizado)
        fact_posts = posts.join(dim_subreddit.select("subreddit"), on="subreddit", how="left")
        fact_comments = comments.join(fact_posts.select("post_id"), on="post_id", how="left")

        # ---------- Persistir a SILVER/parquet ----------
        out = {}
        def _w(df: pl.DataFrame, name: str) -> Path:
            path = settings.silver_dir / f"{name}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(path); out[name] = path; return path

        _w(dim_subreddit, "dim_subreddit")
        _w(dim_author, "dim_author")
        _w(fact_posts, "fact_posts")
        _w(fact_comments, "fact_comments")

        # ---------- Subir a SQLite ----------
        init_db()
        upsert_subreddits(dim_subreddit)
        upsert_authors(dim_author)
        upsert_posts(fact_posts)
        upsert_comments(fact_comments)

        return out
