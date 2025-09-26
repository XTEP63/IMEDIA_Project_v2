from __future__ import annotations
from pathlib import Path
from typing import Iterable, Literal, Optional
import polars as pl
from praw.models import Submission
from .config import settings
from .reddit_client import RedditClient
import time
from .utils import slugify

TimeFilter = Literal["hour","day","week","month","year","all"]
DiscoverStrategy = Literal["popular", "all_hot", "all_top_day"]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _safe(v):
    # Conserva tipos nativos para NDJSON: int/float/bool/str/None.
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    try:
        return str(v)
    except Exception:
        return None
    

class RawExtractor:
    """Descarga de Reddit y volcado a RAW/NDJSON (sin transformar)."""
    def __init__(self, client: Optional[RedditClient] = None) -> None:
        self.reddit = (client or RedditClient()).reddit

    def _submissions_to_rows(self, subs: Iterable[Submission]) -> list[dict]:
        rows: list[dict] = []
        for s in subs:
            rows.append({
                # Identidad y vínculos básicos: (todo string para RAW)
                "id": _safe(s.id),
                "name": _safe(s.name),
                "subreddit": _safe(getattr(s.subreddit, "display_name", None)),
                "author": _safe(getattr(s.author, "name", None)),
                "title": _safe(s.title),
                "selftext": _safe(getattr(s, "selftext", None)),
                "url": _safe(s.url),
                "permalink": _safe(s.permalink),
                # Métricas y flags:
                "score": _safe(s.score),
                "num_comments": _safe(s.num_comments),
                "over_18": _safe(getattr(s, "over_18", None)),
                "stickied": _safe(getattr(s, "stickied", None)),
                # Tiempos:
                "created_utc": _safe(getattr(s, "created_utc", None)),
                "edited": _safe(getattr(s, "edited", None)),
                # Campos extra que suelen ser útiles:
                "link_flair_text": _safe(getattr(s, "link_flair_text", None)),
                "is_self": _safe(getattr(s, "is_self", None)),
                "spoiler": _safe(getattr(s, "spoiler", None)),
                "locked": _safe(getattr(s, "locked", None)),
                "thumbnail": _safe(getattr(s, "thumbnail", None)),
            })
        return rows

    def _write_raw(self, rows: list[dict], entity: str, suffix: str | None = None) -> Path:
        name = f"part-{settings.batch_ts}"
        if suffix:
            name += f"-{slugify(suffix)}"
        out = settings.raw_dir / entity / f"{name}.ndjson"
        out.parent.mkdir(parents=True, exist_ok=True)
        df = pl.DataFrame(rows) if rows else pl.DataFrame(schema={"_empty": pl.Utf8})
        df.write_ndjson(out)
        return out


    def discover_subreddits(
        self,
        n: int = 20,
        *,
        strategy: DiscoverStrategy = "popular",
        posts_limit: int = 500,
        include_nsfw: bool = False,
    ) -> tuple[list[str], Path]:
        """
        Devuelve una lista de hasta N subreddits "calientes" + escribe RAW/NDJSON con la evidencia.
        Estrategias:
          - popular: usa reddit.subreddits.popular(limit=...)
          - all_hot: toma r/all.hot(limit=posts_limit) y rankea por frecuencia del subreddit
          - all_top_day: igual pero con r/all.top(time_filter='day')
        """
        names: list[str] = []

        if strategy == "popular":
            for s in self.reddit.subreddits.popular(limit=max(n * 3, 50)):
                # Nota: .over18 puede requerir fetch; si no aparece, lo dejamos pasar
                nsfw = getattr(s, "over18", False)
                if include_nsfw or not nsfw:
                    names.append(s.display_name)
                if len(names) >= n:
                    break
        else:
            # recolecta posts de r/all y cuenta subreddits
            if strategy == "all_hot":
                stream = self.reddit.subreddit("all").hot(limit=posts_limit)
            else:  # "all_top_day"
                stream = self.reddit.subreddit("all").top(limit=posts_limit, time_filter="day")

            counts: dict[str, int] = {}
            for p in stream:
                sub = p.subreddit.display_name
                # filtra NSFW cuando toque
                if not include_nsfw and getattr(p, "over_18", False):
                    continue
                counts[sub] = counts.get(sub, 0) + 1
            # top-N por frecuencia
            names = [k for k, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:n]]

        # dump RAW para trazabilidad
        raw_rows = [{"subreddit": s, "strategy": strategy, "posts_limit": posts_limit, "include_nsfw": include_nsfw} for s in names]
        raw_path = self._write_raw_list(
            raw_rows, "hot_sublists", suffix=f"{strategy}-n{n}"
        )
        return names, raw_path

    def top_hot_batch(
        self,
        subreddits: list[str],
        *,
        limit_per_sub: int = 50,
        time_filter: Literal["hour","day","week","month","year","all"] = "day",
        sleep_ms: int = 50,
    ) -> list[Path]:
        """
        Descarga top posts de varios subreddits y devuelve las rutas RAW generadas.
        """
        paths: list[Path] = []
        for sub in subreddits:
            p = self.top_posts(sub, limit=limit_per_sub, time_filter=time_filter)
            paths.append(p)
            if sleep_ms:
                time.sleep(sleep_ms/1000.0)  # evita golpear la API
        return paths


    def _write_raw_list(self, rows: list[dict], entity: str, suffix: str | None = None) -> Path:
        name = f"part-{settings.batch_ts}"
        if suffix:
            name += f"-{slugify(suffix)}"
        out = settings.raw_dir / entity / f"{name}.ndjson"
        out.parent.mkdir(parents=True, exist_ok=True)
        (pl.DataFrame(rows) if rows else pl.DataFrame(schema={"_empty": pl.Utf8})).write_ndjson(out)
        return out

    # -------- Métodos públicos RAW ----------
    def top_posts(self, subreddit: str, limit: int = 100, time_filter: TimeFilter = "day") -> Path:
        subs = self.reddit.subreddit(subreddit).top(limit=limit, time_filter=time_filter)
        rows = self._submissions_to_rows(subs)
        return self._write_raw(rows, "posts", suffix=subreddit)

    def search(self, query: str, limit: int = 100, subreddit: str = "all") -> Path:
        subs = self.reddit.subreddit(subreddit).search(query, limit=limit)
        rows = self._submissions_to_rows(subs)
        return self._write_raw(rows, "posts", suffix=f"q-{hash(query) & 0xfffffff}")

    def subreddit_info(self, subreddit: str) -> Path:
        s = self.reddit.subreddit(subreddit)
        row = {
            "subreddit": s.display_name,
            "subscribers": getattr(s, "subscribers", None),
            "public_description": getattr(s, "public_description", None),
            "created_utc": getattr(s, "created_utc", None),
            "over18": getattr(s, "over18", None),
        }
        return self._write_raw([row], "subreddits", suffix=subreddit)


    def comments_of_post(self, post_id: str, limit_replace_more: int = 0) -> Path:
        sub = self.reddit.submission(id=post_id)
        sub.comments.replace_more(limit=limit_replace_more)
        rows = []
        for c in sub.comments.list():
            rows.append({
                "comment_id": _safe(c.id),
                "post_id": _safe(post_id),
                "author": _safe(getattr(c.author, "name", None)),
                "body": _safe(c.body),
                "created_utc": _safe(getattr(c, "created_utc", None)),
                "parent_id": _safe(getattr(c, "parent_id", None)),
                "link_id": _safe(getattr(c, "link_id", None)),
                "score": _safe(getattr(c, "score", None)),
                "is_submitter": _safe(getattr(c, "is_submitter", None)),
            })
        return self._write_raw(rows, "comments", suffix=post_id)

    def user_recent_subreddits(self, username: str, limit: int = 50) -> Path:
        user = self.reddit.redditor(username)
        rows = []
        for subm in user.submissions.new(limit=limit):
            rows.append({
                "username": _safe(username),
                "subreddit": _safe(subm.subreddit.display_name),
                "last_post_id": _safe(subm.id),
                "last_created_utc": _safe(getattr(subm, "created_utc", None)),
            })
        return self._write_raw(rows, "usersubs")
