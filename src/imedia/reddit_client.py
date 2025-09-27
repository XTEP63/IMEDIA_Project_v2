from __future__ import annotations
import praw
from prawcore.exceptions import RequestException
from .config import settings

class RedditClient:
    """Solo se encarga de autenticar y entregar un objeto praw.Reddit listo."""
    def __init__(self) -> None:
        settings.validate()
        self._reddit = praw.Reddit(
            client_id=settings.client_id,
            client_secret=settings.client_secret,
            user_agent=settings.user_agent,
            requestor_kwargs={"timeout": settings.request_timeout},
            check_for_async=False,
        )
        self._reddit.read_only = True
        # Smoke-check
        try:
            _ = self._reddit.read_only
        except RequestException as e:
            raise RuntimeError(f"Error inicializando Reddit: {e}") from e

    @property
    def reddit(self) -> praw.Reddit:
        return self._reddit
