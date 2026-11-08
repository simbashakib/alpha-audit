"""Universal market data provider with Yahoo Finance dynamic fetching,
corporate-action adjustment (splits & dividends), SHA-256 provenance hashing,
and local disk caching.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # type: ignore

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "cache"
MANIFEST_PATH = DATA_DIR / "data_manifest.json"

# Natural-language name to Yahoo Finance ticker mapping
ENTITY_TO_TICKER: dict[str, str] = {
    "s&p 500": "SPY",
    "s&p500": "SPY",
    "s&p": "SPY",
    "sp500": "SPY",
    "spy": "SPY",
    "^gspc": "^GSPC",
    "nasdaq": "QQQ",
    "nasdaq 100": "QQQ",
    "qqq": "QQQ",
    "dow jones": "DIA",
    "russell 2000": "IWM",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "tesla": "TSLA",
    "netflix": "NFLX",
    "amd": "AMD",
    "intel": "INTC",
    "coca-cola": "KO",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "pepsico": "PEP",
    "exxon": "XOM",
    "exxonmobil": "XOM",
    "exxon mobil": "XOM",
    "chevron": "CVX",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "bank of america": "BAC",
    "goldman sachs": "GS",
    "morgan stanley": "MS",
    "citigroup": "C",
    "wells fargo": "WFC",
    "visa": "V",
    "mastercard": "MA",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
    "eli lilly": "LLY",
    "walmart": "WMT",
    "costco": "COST",
    "target corp": "TGT",
    "target stock": "TGT",
    "home depot": "HD",
    "lowes": "LOW",
    "lowe's": "LOW",
    "disney": "DIS",
    "mcdonald's": "MCD",
    "mcdonalds": "MCD",
    "starbucks": "SBUX",
    "nike": "NKE",
    "boeing": "BA",
    "airbus": "EADSY",
    "ford": "F",
    "general motors": "GM",
    "gm": "GM",
    "tsmc": "TSM",
    "taiwan semiconductor": "TSM",
    "asml": "ASML",
    "broadcom": "AVGO",
    "oracle": "ORCL",
    "salesforce": "CRM",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "coinbase": "COIN",
    "palantir": "PLTR",
    "gold": "GLD",
    "silver": "SLV",
    "crude oil": "CL=F",
    "treasuries": "TLT",
    "20y treasury": "TLT",
    "bonds": "TLT",
    "ftse 100": "^FTSE",
    "ftse": "^FTSE",
    "shell": "SHEL",
    "bp": "BP",
    "astrazeneca": "AZN",
    "hsbc": "HSBC",
    "unilever": "UL",
    "barclays": "BCS",
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
}

CATALOG_TICKERS: list[dict[str, str]] = [
    {"ticker": "SPY", "name": "SPDR S&P 500 ETF Trust", "category": "US Equity ETF"},
    {"ticker": "QQQ", "name": "Invesco QQQ Trust (Nasdaq-100)", "category": "US Equity ETF"},
    {"ticker": "IWM", "name": "iShares Russell 2000 ETF", "category": "US Equity ETF"},
    {"ticker": "DIA", "name": "SPDR Dow Jones Industrial Average ETF", "category": "US Equity ETF"},
    {"ticker": "AAPL", "name": "Apple Inc.", "category": "US Equities — Tech"},
    {"ticker": "MSFT", "name": "Microsoft Corporation", "category": "US Equities — Tech"},
    {"ticker": "NVDA", "name": "NVIDIA Corporation", "category": "US Equities — Semiconductors"},
    {"ticker": "GOOGL", "name": "Alphabet Inc.", "category": "US Equities — Tech"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "category": "US Equities — Consumer/Cloud"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "category": "US Equities — Tech"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "category": "US Equities — Auto/Tech"},
    {"ticker": "AMD", "name": "Advanced Micro Devices Inc.", "category": "US Equities — Semiconductors"},
    {"ticker": "KO", "name": "The Coca-Cola Company", "category": "US Equities — Consumer Staples"},
    {"ticker": "PEP", "name": "PepsiCo Inc.", "category": "US Equities — Consumer Staples"},
    {"ticker": "XOM", "name": "Exxon Mobil Corporation", "category": "US Equities — Energy"},
    {"ticker": "CVX", "name": "Chevron Corporation", "category": "US Equities — Energy"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "category": "US Equities — Financials"},
    {"ticker": "BAC", "name": "Bank of America Corporation", "category": "US Equities — Financials"},
    {"ticker": "GS", "name": "The Goldman Sachs Group Inc.", "category": "US Equities — Financials"},
    {"ticker": "MS", "name": "Morgan Stanley", "category": "US Equities — Financials"},
    {"ticker": "V", "name": "Visa Inc.", "category": "US Equities — Payments"},
    {"ticker": "MA", "name": "Mastercard Incorporated", "category": "US Equities — Payments"},
    {"ticker": "WMT", "name": "Walmart Inc.", "category": "US Equities — Retail"},
    {"ticker": "COST", "name": "Costco Wholesale Corporation", "category": "US Equities — Retail"},
    {"ticker": "HD", "name": "The Home Depot Inc.", "category": "US Equities — Retail"},
    {"ticker": "LOW", "name": "Lowe's Companies Inc.", "category": "US Equities — Retail"},
    {"ticker": "GLD", "name": "SPDR Gold Shares", "category": "Commodities ETF"},
    {"ticker": "TLT", "name": "iShares 20+ Year Treasury Bond ETF", "category": "Fixed Income ETF"},
    {"ticker": "BTC-USD", "name": "Bitcoin USD", "category": "Digital Assets"},
    {"ticker": "ETH-USD", "name": "Ethereum USD", "category": "Digital Assets"},
]


def safe_filename(ticker: str) -> str:
    """Convert ticker symbol into a filesystem-safe identifier."""
    cleaned = re.sub(r"[^A-Za-z0-9._^-]", "_", ticker.strip().upper())
    return cleaned.replace("^", "IDX_")


def resolve_ticker(raw: str) -> str:
    """Resolve a company name, common alias, or raw Yahoo Finance symbol."""
    key = raw.strip().lower()
    if key in ENTITY_TO_TICKER:
        return ENTITY_TO_TICKER[key]
    return raw.strip().upper()


def _normalize_yfinance_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Yahoo Finance DataFrame and apply corporate-action adjustment."""
    if df is None or df.empty:
        raise ValueError("Empty DataFrame returned from data provider.")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [col[0] for col in df.columns]

    df = df.copy().reset_index()
    date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
    df["date"] = pd.to_datetime(df[date_col]).dt.tz_localize(None).dt.strftime("%Y-%m-%d")

    col_map = {c.lower(): c for c in df.columns}
    close_col = col_map.get("close")
    open_col = col_map.get("open", close_col)
    high_col = col_map.get("high", close_col)
    low_col = col_map.get("low", close_col)
    adj_col = col_map.get("adj close", close_col)
    vol_col = col_map.get("volume")

    if not close_col:
        raise ValueError("Missing Close column in market data.")

    raw_close = pd.to_numeric(df[close_col], errors="coerce")
    raw_open = pd.to_numeric(df[open_col], errors="coerce").fillna(raw_close)
    raw_high = pd.to_numeric(df[high_col], errors="coerce").fillna(raw_close)
    raw_low = pd.to_numeric(df[low_col], errors="coerce").fillna(raw_close)
    adj_close = pd.to_numeric(df[adj_col], errors="coerce").fillna(raw_close)
    volume = pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0) if vol_col else pd.Series(0.0, index=df.index)

    # Corporate-action ratio adjustment (splits + dividends) applied consistently to OHLC
    adj_ratio = np.where(raw_close > 0, adj_close / raw_close, 1.0)
    out = pd.DataFrame(
        {
            "date": df["date"],
            "open": np.round(raw_open * adj_ratio, 6),
            "high": np.round(raw_high * adj_ratio, 6),
            "low": np.round(raw_low * adj_ratio, 6),
            "close": np.round(adj_close, 6),
            "raw_close": np.round(raw_close, 6),
            "volume": volume.astype(np.int64),
        }
    )
    out = out.dropna(subset=["date", "open", "close"])
    out = out[(out["open"] > 0) & (out["close"] > 0)]
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date").reset_index(drop=True)
    if len(out) < 40:
        raise ValueError(f"Insufficient valid trading days ({len(out)} < 40).")
    return out


def fetch_and_cache_ticker(ticker: str, start_date: str = "2015-01-01", force_refresh: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load daily corporate-action-adjusted OHLCV for any Yahoo Finance ticker.
    Checks local disk cache first; fetches live from Yahoo Finance if not cached.
    """
    symbol = resolve_ticker(ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CACHE_DIR / f"{safe_filename(symbol)}.csv"

    source_note = "Yahoo Finance (Corporate-Action Adjusted Daily OHLCV — Cached)"
    if csv_path.exists() and not force_refresh:
        df = pd.read_csv(csv_path)
        raw_bytes = csv_path.read_bytes()
    else:
        if yf is None:
            raise ValueError(f"Ticker {symbol} is not in local cache and yfinance is not installed.")
        try:
            yt = yf.Ticker(symbol)
            hist = yt.history(start=start_date, auto_adjust=False, actions=False)
            df = _normalize_yfinance_df(hist)
            df.to_csv(csv_path, index=False)
            raw_bytes = csv_path.read_bytes()
            source_note = "Yahoo Finance Live Fetch (Corporate-Action Adjusted Daily OHLCV)"
        except Exception as exc:
            raise ValueError(
                f"Could not fetch historical data for ticker '{symbol}' from Yahoo Finance ({exc}). "
                "Verify that the ticker symbol is valid on Yahoo Finance (e.g., AAPL, SPY, KO, BRK-B, BTC-USD, ^FTSE)."
            ) from exc

    sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
    meta = {
        "ticker": symbol,
        "source": source_note,
        "adjustment": "CRSP/Yahoo Total-Return & Split Ratio Applied to Open/High/Low/Close",
        "first_date": str(df["date"].iloc[0]),
        "last_date": str(df["date"].iloc[-1]),
        "total_rows": int(len(df)),
        "sha256": sha256_hash,
    }
    _update_manifest(symbol, meta)
    return df, meta


def _update_manifest(symbol: str, meta: dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"updated_at_utc": datetime.now(timezone.utc).isoformat(), "tickers": {}}
    if MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    manifest.setdefault("tickers", {})[symbol] = meta
    manifest["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_aligned_series(
    asset_a: str,
    start_date: str,
    end_date: str,
    asset_b: str | None = None,
    benchmark_ticker: str = "SPY",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and date-align corporate-action-adjusted series for asset_a, optional asset_b,
    and benchmark_ticker over [start_date, end_date].
    """
    df_a, meta_a = fetch_and_cache_ticker(asset_a)
    df_bench, meta_bench = fetch_and_cache_ticker(benchmark_ticker)

    merged = df_a[["date", "open", "high", "low", "close", "volume"]].rename(
        columns={
            "open": "open_a",
            "high": "high_a",
            "low": "low_a",
            "close": "close_a",
            "volume": "volume_a",
        }
    )

    provenance: dict[str, Any] = {
        "asset_a": meta_a,
        "benchmark": meta_bench,
    }

    if asset_b:
        df_b, meta_b = fetch_and_cache_ticker(asset_b)
        provenance["asset_b"] = meta_b
        merged = pd.merge(
            merged,
            df_b[["date", "open", "high", "low", "close", "volume"]].rename(
                columns={
                    "open": "open_b",
                    "high": "high_b",
                    "low": "low_b",
                    "close": "close_b",
                    "volume": "volume_b",
                }
            ),
            on="date",
            how="inner",
        )

    merged = pd.merge(
        merged,
        df_bench[["date", "open", "close"]].rename(
            columns={"open": "open_bench", "close": "close_bench"}
        ),
        on="date",
        how="inner",
    )

    merged = merged[(merged["date"] >= start_date) & (merged["date"] <= end_date)].sort_values("date").reset_index(drop=True)
    if len(merged) < 50:
        raise ValueError(
            f"Overlapping date range [{start_date}, {end_date}] contains only {len(merged)} aligned trading days "
            f"(minimum 50 required for valid formation + out-of-sample evaluation)."
        )

    # Compute exact SHA-256 fingerprint of the aligned slice used in the experiment
    slice_csv = merged.to_csv(index=False).encode("utf-8")
    provenance["aligned_slice_sha256"] = hashlib.sha256(slice_csv).hexdigest()
    provenance["aligned_trading_days"] = int(len(merged))
    provenance["effective_start_date"] = str(merged["date"].iloc[0])
    provenance["effective_end_date"] = str(merged["date"].iloc[-1])

    return merged, provenance


def validate_ticker_symbol(ticker: str) -> dict[str, Any]:
    """Validate and preview any Yahoo Finance ticker symbol."""
    resolved = resolve_ticker(ticker)
    try:
        df, meta = fetch_and_cache_ticker(resolved)
        latest_close = float(df["close"].iloc[-1])
        first_close = float(df["close"].iloc[0])
        return {
            "valid": True,
            "requested": ticker,
            "resolved_ticker": resolved,
            "first_date": meta["first_date"],
            "last_date": meta["last_date"],
            "total_rows": meta["total_rows"],
            "latest_close": round(latest_close, 2),
            "first_close": round(first_close, 2),
            "sha256": meta["sha256"][:16],
            "source": meta["source"],
        }
    except Exception as exc:
        return {
            "valid": False,
            "requested": ticker,
            "resolved_ticker": resolved,
            "error": str(exc),
        }
