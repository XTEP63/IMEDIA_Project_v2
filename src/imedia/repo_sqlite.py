from __future__ import annotations
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import polars as pl
from .config import settings

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS subreddits (
      subreddit TEXT PRIMARY KEY,
      subscribers INTEGER,
      description TEXT,
      created_utc REAL,
      over18 INTEGER
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS authors (
      author_name TEXT PRIMARY KEY
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS posts (
      post_id TEXT PRIMARY KEY,
      title TEXT,
      selftext TEXT,
      url TEXT,
      permalink TEXT,
      score INTEGER,
      num_comments INTEGER,
      over_18 INTEGER,
      created_utc REAL,
      link_flair_text TEXT,
      is_self INTEGER,
      spoiler INTEGER,
      locked INTEGER,
      thumbnail TEXT,
      subreddit TEXT,
      author TEXT,
      FOREIGN KEY(subreddit) REFERENCES subreddits(subreddit)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS comments (
      comment_id TEXT PRIMARY KEY,
      post_id TEXT,
      author TEXT,
      body TEXT,
      created_utc REAL,
      parent_id TEXT,
      link_id TEXT,
      score INTEGER,
      is_submitter INTEGER,
      FOREIGN KEY(post_id) REFERENCES posts(post_id)
    );
    """
]

@contextmanager
def _conn():
    Path(settings.db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(settings.db_path)
    try:
        yield con
    finally:
        con.commit()
        con.close()

def init_db() -> None:
    with _conn() as con:
        cur = con.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)

# -------- UPSERTS -----------
def upsert_subreddits(df: pl.DataFrame) -> None:
    if df.is_empty(): return
    cols = ["subreddit","subscribers","description","created_utc","over18"]
    d = df.select([c for c in cols if c in df.columns])
    with _conn() as con:
        con.executemany(
            """
            INSERT INTO subreddits (subreddit, subscribers, description, created_utc, over18)
            VALUES (?,?,?,?,?)
            ON CONFLICT(subreddit) DO UPDATE SET
              subscribers=excluded.subscribers,
              description=excluded.description,
              created_utc=excluded.created_utc,
              over18=excluded.over18
            """,
            d.iter_rows()
        )

def upsert_authors(df: pl.DataFrame) -> None:
    if df.is_empty(): return
    d = df.select("author_name").unique()
    with _conn() as con:
        con.executemany("INSERT OR IGNORE INTO authors(author_name) VALUES (?)", d.iter_rows())

def upsert_posts(df: pl.DataFrame) -> None:
    if df.is_empty(): return
    cols = ["post_id","title","selftext","url","permalink","score","num_comments","over_18",
            "created_utc","link_flair_text","is_self","spoiler","locked","thumbnail","subreddit","author"]
    d = df.select([c for c in cols if c in df.columns])
    with _conn() as con:
        con.executemany(
            f"""
            INSERT INTO posts ({",".join(cols)})
            VALUES ({",".join(["?"]*len(cols))})
            ON CONFLICT(post_id) DO UPDATE SET
              title=excluded.title,
              selftext=excluded.selftext,
              url=excluded.url,
              permalink=excluded.permalink,
              score=excluded.score,
              num_comments=excluded.num_comments,
              over_18=excluded.over_18,
              created_utc=excluded.created_utc,
              link_flair_text=excluded.link_flair_text,
              is_self=excluded.is_self,
              spoiler=excluded.spoiler,
              locked=excluded.locked,
              thumbnail=excluded.thumbnail,
              subreddit=excluded.subreddit,
              author=excluded.author
            """,
            d.iter_rows()
        )

def upsert_comments(df: pl.DataFrame) -> None:
    if df.is_empty(): return
    cols = ["comment_id","post_id","author","body","created_utc","parent_id","link_id","score","is_submitter"]
    d = df.select([c for c in cols if c in df.columns])
    with _conn() as con:
        con.executemany(
            f"""
            INSERT INTO comments ({",".join(cols)})
            VALUES ({",".join(["?"]*len(cols))})
            ON CONFLICT(comment_id) DO UPDATE SET
              post_id=excluded.post_id,
              author=excluded.author,
              body=excluded.body,
              created_utc=excluded.created_utc,
              parent_id=excluded.parent_id,
              link_id=excluded.link_id,
              score=excluded.score,
              is_submitter=excluded.is_submitter
            """,
            d.iter_rows()
        )
