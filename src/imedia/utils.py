# src/imedia/utils.py
from __future__ import annotations
import re
from uuid import uuid4
import polars as pl

# ---------- slugify ----------
_slug_re = re.compile(r"[^a-z0-9_-]+")

def slugify(s: str) -> str:
    s = (s or "").lower().strip().replace(" ", "_")
    s = _slug_re.sub("", s)
    return s or uuid4().hex[:8]

# ---------- casts robustos (compat con Polars viejo) ----------
TRUE_SET  = {"true","t","yes","y","1","on"}
FALSE_SET = {"false","f","no","n","0","off"}

def to_int_expr(col: str) -> pl.Expr:
    x = pl.col(col)
    # 1) si ya es int -> int; 2) utf8 -> int; 3) float -> int
    return pl.coalesce([
        x.cast(pl.Int64, strict=False),
        x.cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Int64, strict=False),
        x.cast(pl.Float64, strict=False).cast(pl.Int64, strict=False),
    ])

def to_float_expr(col: str) -> pl.Expr:
    x = pl.col(col)
    # 1) num -> float; 2) utf8 -> float
    return pl.coalesce([
        x.cast(pl.Float64, strict=False),
        x.cast(pl.Utf8, strict=False).str.strip_chars().cast(pl.Float64, strict=False),
    ])

def to_bool_expr(col: str) -> pl.Expr:
    x = pl.col(col)
    # 1) si ya es bool o num (0/1) -> bool (cast flojo); 2) strings mapeadas -> bool
    bool1 = x.cast(pl.Boolean, strict=False)
    s = x.cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()
    bool2 = pl.when(s.is_in(list(TRUE_SET))).then(pl.lit(True)) \
             .when(s.is_in(list(FALSE_SET))).then(pl.lit(False)) \
             .otherwise(pl.lit(None, dtype=pl.Boolean))
    return pl.coalesce([bool1, bool2])
