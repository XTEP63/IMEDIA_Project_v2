from pathlib import Path
import polars as pl
from .config import settings

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def dump_raw_ndjson(df: pl.DataFrame, *, entity: str) -> Path:
    if df.is_empty():
        return settings.raw_dir / entity / f"empty-{settings.batch_ts}.ndjson"
    out = settings.raw_dir / entity / f"part-{settings.batch_ts}.ndjson"
    _ensure_dir(out)
    df.write_ndjson(out)  # Polars >=0.20
    return out

def write_bronze_parquet(df: pl.DataFrame, *, entity: str, created_utc_col: str | None = None) -> Path:
    # Particiona por fecha (UTC) si existe la columna temporal, sino usa batch_ts
    if df.is_empty():
        out = settings.bronze_dir / entity / f"empty-{settings.batch_ts}.parquet"
        _ensure_dir(out)
        df.write_parquet(out)
        return out
    if created_utc_col and created_utc_col in df.columns:
        df2 = (df
               .with_columns(pl.from_epoch(pl.col(created_utc_col), time_unit="s").alias("_dt"))
               .with_columns([
                   pl.col("_dt").dt.year().alias("YYYY"),
                   pl.col("_dt").dt.month().alias("MM"),
                   pl.col("_dt").dt.day().alias("DD"),
               ])
              )
        # Escribimos 1 archivo por partición YYYY/MM/DD
        paths = []
        for y, m, d, part in df2.partition_by(["YYYY","MM","DD"], as_dict=True).items():
            y, m, d = map(str, (y, m, d))
            out = settings.bronze_dir / entity / f"YYYY={y}/MM={m}/DD={d}/{entity}.parquet"
            _ensure_dir(out)
            part.drop(["YYYY","MM","DD","_dt"]).write_parquet(out)
            paths.append(out)
        return paths[-1]
    else:
        out = settings.bronze_dir / entity / f"{entity}-{settings.batch_ts}.parquet"
        _ensure_dir(out)
        df.write_parquet(out)
        return out
