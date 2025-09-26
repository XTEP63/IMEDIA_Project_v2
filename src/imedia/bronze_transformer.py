# src/imedia/bronze_transformer.py
from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import polars as pl
from .config import settings
from .utils import slugify, to_bool_expr, to_int_expr, to_float_expr

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

class BronzeTransformer:
    """Lee RAW NDJSON, hace flatten/tipado mínimo y escribe BRONZE (Parquet)."""

    def _write_part(self, df: pl.DataFrame, entity: str, created_col: str | None = None, file_suffix: str | None = None) -> Path:
        if df.is_empty():
            out = settings.bronze_dir / entity / f"empty-{settings.batch_ts}-{uuid4().hex[:8]}.parquet"
            _ensure_dir(out); df.write_parquet(out); return out

        if created_col and created_col in df.columns:
            # asegurar numérico antes de particionar
            df = df.with_columns(to_float_expr(created_col).alias(created_col))
            df2 = (
                df.with_columns(pl.from_epoch(pl.col(created_col), time_unit="s").alias("_dt"))
                  .with_columns([
                      pl.col("_dt").dt.year().alias("YYYY"),
                      pl.col("_dt").dt.month().alias("MM"),
                      pl.col("_dt").dt.day().alias("DD"),
                  ])
            )
            last = None
            for (y,m,d), part in df2.partition_by(["YYYY","MM","DD"], as_dict=True).items():
                suffix = file_suffix or uuid4().hex[:8]
                fname = f"{entity}__{settings.batch_ts}__{slugify(str(suffix))}.parquet"
                out = settings.bronze_dir / entity / f"YYYY={y}/MM={m}/DD={d}/{fname}"
                _ensure_dir(out)
                part.drop(["YYYY","MM","DD","_dt"]).write_parquet(out)
                last = out
            return last

        suffix = file_suffix or uuid4().hex[:8]
        out = settings.bronze_dir / entity / f"{entity}-{settings.batch_ts}-{slugify(str(suffix))}.parquet"
        _ensure_dir(out); df.write_parquet(out); return out

    # ------- entidades ----------
    def posts(self, raw_ndjson_path: Path) -> Path:
        df = pl.read_ndjson(raw_ndjson_path)
        want = {
            "id": "post_id", "subreddit": "subreddit", "author": "author",
            "title": "title", "selftext": "selftext", "url": "url", "permalink": "permalink",
            "score": "score", "num_comments": "num_comments", "over_18": "over_18",
            "created_utc": "created_utc", "link_flair_text": "link_flair_text",
            "is_self": "is_self", "spoiler": "spoiler", "locked": "locked", "thumbnail": "thumbnail",
        }
        cols = [c for c in want.keys() if c in df.columns]
        df2 = df.select([pl.col(c).alias(want[c]) for c in cols]).with_columns([
            to_int_expr("score").alias("score") if "score" in df.columns else pl.lit(None).alias("score"),
            to_int_expr("num_comments").alias("num_comments") if "num_comments" in df.columns else pl.lit(None).alias("num_comments"),
            to_bool_expr("over_18").alias("over_18") if "over_18" in df.columns else pl.lit(None).alias("over_18"),
            to_float_expr("created_utc").alias("created_utc") if "created_utc" in df.columns else pl.lit(None).alias("created_utc"),
            to_bool_expr("is_self").alias("is_self") if "is_self" in df.columns else pl.lit(None).alias("is_self"),
            to_bool_expr("spoiler").alias("spoiler") if "spoiler" in df.columns else pl.lit(None).alias("spoiler"),
            to_bool_expr("locked").alias("locked") if "locked" in df.columns else pl.lit(None).alias("locked"),
        ])

        file_suffix = None
        if df2.height and "subreddit" in df2.columns:
            file_suffix = df2.select(pl.col("subreddit").first()).item()

        return self._write_part(df2, "posts", created_col="created_utc", file_suffix=file_suffix)

    def comments(self, raw_ndjson_path: Path) -> Path:
        df = pl.read_ndjson(raw_ndjson_path)
        want = {
            "comment_id": "comment_id", "post_id": "post_id", "author": "author",
            "body": "body", "created_utc": "created_utc", "parent_id": "parent_id",
            "link_id": "link_id", "score": "score", "is_submitter": "is_submitter",
        }
        cols = [c for c in want.keys() if c in df.columns]
        df2 = df.select([pl.col(c).alias(want[c]) for c in cols]).with_columns([
            to_int_expr("score").alias("score") if "score" in df.columns else pl.lit(None).alias("score"),
            to_bool_expr("is_submitter").alias("is_submitter") if "is_submitter" in df.columns else pl.lit(None).alias("is_submitter"),
            to_float_expr("created_utc").alias("created_utc") if "created_utc" in df.columns else pl.lit(None).alias("created_utc"),
        ])

        file_suffix = None
        if df2.height and "post_id" in df2.columns:
            file_suffix = df2.select(pl.col("post_id").first()).item()

        return self._write_part(df2, "comments", created_col="created_utc", file_suffix=file_suffix)

    def subreddits(self, raw_ndjson_path: Path) -> Path:
        df = pl.read_ndjson(raw_ndjson_path)
        want = {
            "subreddit": "subreddit", "subscribers": "subscribers",
            "public_description": "description", "created_utc": "created_utc", "over18": "over18",
        }
        cols = [c for c in want.keys() if c in df.columns]
        df2 = df.select([pl.col(c).alias(want[c]) for c in cols]).with_columns([
            to_int_expr("subscribers").alias("subscribers") if "subscribers" in df.columns else pl.lit(None).alias("subscribers"),
            to_bool_expr("over18").alias("over18") if "over18" in df.columns else pl.lit(None).alias("over18"),
            to_float_expr("created_utc").alias("created_utc") if "created_utc" in df.columns else pl.lit(None).alias("created_utc"),
        ])

        file_suffix = None
        if "subreddit" in df2.columns and df2.height >= 1:
            file_suffix = df2["subreddit"][0]
        fname = f"subreddits-{settings.batch_ts}-{slugify(str(file_suffix) if file_suffix else uuid4().hex[:8])}.parquet"
        out = settings.bronze_dir / "subreddits" / fname
        _ensure_dir(out); df2.write_parquet(out)
        return out

    def usersubs(self, raw_ndjson_path: Path) -> Path:
        df = pl.read_ndjson(raw_ndjson_path)
        want = {"username":"username","subreddit":"subreddit","last_post_id":"last_post_id","last_created_utc":"last_created_utc"}
        cols = [c for c in want.keys() if c in df.columns]
        df2 = df.select([pl.col(c).alias(want[c]) for c in cols]).with_columns([
            to_float_expr("last_created_utc").alias("last_created_utc") if "last_created_utc" in df.columns else pl.lit(None).alias("last_created_utc"),
        ])
        out = settings.bronze_dir / "usersubs" / f"usersubs-{settings.batch_ts}-{uuid4().hex[:8]}.parquet"
        _ensure_dir(out); df2.write_parquet(out); return out

    def hotlist(self, raw_ndjson_path: Path) -> Path:
        df = pl.read_ndjson(raw_ndjson_path)
        want = {"subreddit": "subreddit", "strategy": "strategy", "posts_limit": "posts_limit", "include_nsfw":"include_nsfw"}
        cols = [c for c in want.keys() if c in df.columns]
        df2 = df.select([pl.col(c).alias(want[c]) for c in cols]).with_columns([
            to_int_expr("posts_limit").alias("posts_limit") if "posts_limit" in df.columns else pl.lit(None).alias("posts_limit"),
            to_bool_expr("include_nsfw").alias("include_nsfw") if "include_nsfw" in df.columns else pl.lit(None).alias("include_nsfw"),
        ])
        out = settings.bronze_dir / "hot_sublists" / f"hot_sublists-{settings.batch_ts}-{uuid4().hex[:8]}.parquet"
        _ensure_dir(out); df2.write_parquet(out)
        return out
