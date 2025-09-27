from __future__ import annotations
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret: str = os.getenv("REDDIT_CLIENT_SECRET", "")
    user_agent: str = os.getenv("REDDIT_USER_AGENT", "imedia/0.1 (polars)")
    request_timeout: int = int(os.getenv("REDDIT_REQUEST_TIMEOUT", "30"))
    db_path: str = os.getenv("IMEDIA_DB_PATH", "db/imedia.sqlite")
    data_root: Path = Path(os.getenv("IMEDIA_DATA_ROOT", "data"))
    batch_ts: str = os.getenv(
        "IMEDIA_BATCH_TS", datetime.utcnow().strftime("%Y%m%d_%H%M")
    )

    @property
    def raw_dir(self) -> Path: return self.data_root / "raw" / "reddit"
    @property
    def bronze_dir(self) -> Path: return self.data_root / "bronze" / "reddit"
    @property
    def silver_dir(self) -> Path: return self.data_root / "silver" / "reddit"

    def validate(self) -> None:
        missing = []
        if not self.client_id: missing.append("REDDIT_CLIENT_ID")
        if not self.client_secret: missing.append("REDDIT_CLIENT_SECRET")
        if not self.user_agent: missing.append("REDDIT_USER_AGENT")
        if missing:
            raise ValueError(f"Faltan variables en .env: {', '.join(missing)}")

settings = Settings()
