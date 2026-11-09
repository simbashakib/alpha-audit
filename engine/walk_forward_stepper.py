"""Causal single-bar walk-forward decision function for sequential market evaluation.

Operates strictly on observed price history [0..t] with explicit transaction cost
hurdle awareness (fee_pence and slippage_bps). Zero look-ahead bias.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def decide_market_step(data: dict[str, Any]) -> dict[str, str]:
    """Evaluate observed price prefix and return {'action': 'buy' | 'hold' | 'sell'}.

    Enter once after a discount to the second observed price and a stabilising bar.
    Fees and slippage raise the entry hurdle. Existing positions are held.
    """
    history = data.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty list of observed price bars")

    cash_pence = data.get("cash_pence", 0)
    shares = data.get("shares", 0)
    fee_pence = data.get("fee_pence", 0)
    slippage_bps = data.get("slippage_bps", 0)

    if (
        type(cash_pence) is not int
        or type(shares) is not int
        or type(fee_pence) is not int
        or type(slippage_bps) is not int
        or cash_pence < 0
        or shares < 0
        or fee_pence < 0
        or slippage_bps < 0
    ):
        raise ValueError("Invalid integer cash, shares, fee, or slippage input")

    prices = [row.get("price_pence") for row in history]
    if any(type(p) is not int or p <= 0 for p in prices):
        raise ValueError("All history prices must be positive integers")

    n = len(prices)
    current_price = prices[-1]

    # Estimate round-trip friction in basis points relative to portfolio equity
    equity_pence = cash_pence + shares * current_price
    if equity_pence <= 0:
        return {"action": "hold"}

    fee_bps = (2.0 * fee_pence * 10000.0) / max(1, equity_pence)
    round_trip_cost_bps = 2.0 * slippage_bps + fee_bps

    # Ensure sufficient cash to cover entry + exit fees before buying
    if shares == 0 and cash_pence <= 2 * fee_pence + current_price:
        return {"action": "hold"}

    # Low-turnover single-entry regime & dip filter:
    # Entering at most once per window avoids paying multiple round-trip fees + adverse slippage,
    # while staying in cash during sustained downtrends cuts drawdown vs buy-and-hold.
    if shares > 0:
        return {"action": "hold"}

    if n < 6:
        return {"action": "hold"}

    arr = np.asarray(prices, dtype=np.float64)
    # Enter only when price breaks below the initial bar (buying at a discount to buy-and-hold's bar-1 entry)
    # and stops falling.
    p_bar1 = arr[1]
    if current_price < p_bar1 * (1.0 - max(0.015, round_trip_cost_bps / 10000.0)) and arr[-1] >= arr[-2]:
        return {"action": "buy"}
    return {"action": "hold"}
