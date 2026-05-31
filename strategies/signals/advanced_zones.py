"""
Advanced ICT Zone Detection
============================
Four supplementary zone types that upgrade OB confluence scoring:

  1. Volume Imbalance (VI)   — consecutive body gap = undelivered price
  2. Breaker Blocks          — failed OBs that flip polarity (supply ↔ demand)
  3. Rejection Blocks        — large-wick candles at key levels = body is entry zone
  4. Mitigation Blocks       — OBs tested once but not broken = second-chance entry

Zone Quality Scorer
-------------------
  Ranks any OB by how many confluence layers stack on it:
    • OB + FVG overlap        → institutional area  (+0.15)
    • OB + VI overlap         → strong order flow   (+0.15)
    • OB near BSL/SSL         → liquidity confluece (+0.20)  ← highest priority
    • Breaker at BSL/SSL      → stop-hunt entry     (+0.20)
    • All three overlapping   → triple confluence   (+0.10)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ── Volume Imbalance ─────────────────────────────────────────────────────────

def detect_volume_imbalance(
    df: pd.DataFrame,
    min_size_atr: float = 0.10,
) -> pd.DataFrame:
    """
    Volume Imbalance: gap between consecutive candle bodies (open vs prior close).

    Bull VI: current open > prior close  → gap up in delivery
             Zone = [prior_close, current_open]
    Bear VI: current open < prior close  → gap down in delivery
             Zone = [current_open, prior_close]

    Tracks last unfilled VI; invalidated when price closes through zone.
    """
    opens  = df["open"].values
    closes = df["close"].values
    atrs   = df["atr"].values if "atr" in df.columns else np.full(len(df), 1e-9)
    n = len(df)

    vi_bull_top = np.full(n, np.nan)
    vi_bull_bot = np.full(n, np.nan)
    vi_bear_top = np.full(n, np.nan)
    vi_bear_bot = np.full(n, np.nan)

    cur_vb_top = cur_vb_bot = cur_va_top = cur_va_bot = np.nan

    for i in range(1, n):
        atr = atrs[i] if (not np.isnan(atrs[i]) and atrs[i] > 0) else 1e-9

        gap_bull = opens[i] - closes[i - 1]
        gap_bear = closes[i - 1] - opens[i]

        if gap_bull >= min_size_atr * atr:
            cur_vb_top = opens[i]
            cur_vb_bot = closes[i - 1]

        if gap_bear >= min_size_atr * atr:
            cur_va_top = closes[i - 1]
            cur_va_bot = opens[i]

        # Invalidate when price closes through the VI zone (delivered)
        c = closes[i]
        if not np.isnan(cur_vb_bot) and c < cur_vb_bot:
            cur_vb_top = cur_vb_bot = np.nan
        if not np.isnan(cur_va_top) and c > cur_va_top:
            cur_va_top = cur_va_bot = np.nan

        vi_bull_top[i] = cur_vb_top
        vi_bull_bot[i] = cur_vb_bot
        vi_bear_top[i] = cur_va_top
        vi_bear_bot[i] = cur_va_bot

    out = df.copy()
    out["vi_bull_top"] = vi_bull_top
    out["vi_bull_bot"] = vi_bull_bot
    out["vi_bear_top"] = vi_bear_top
    out["vi_bear_bot"] = vi_bear_bot

    c = df["close"]
    out["in_vi_bull"] = (
        c.ge(out["vi_bull_bot"]) & c.le(out["vi_bull_top"]) & out["vi_bull_bot"].notna()
    )
    out["in_vi_bear"] = (
        c.ge(out["vi_bear_bot"]) & c.le(out["vi_bear_top"]) & out["vi_bear_bot"].notna()
    )
    return out


# ── Breaker Blocks ───────────────────────────────────────────────────────────

def detect_breaker_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Breaker: failed OB that flips polarity.

    Bull OB broken (price closes below wick low)
      → Former demand zone becomes supply  = bear breaker
    Bear OB broken (price closes above wick high)
      → Former supply zone becomes demand  = bull breaker

    Requires ob_bull_top/bot/wick_low and ob_bear_top/bot/wick_high in df
    (produced by _rolling_ob with wick tracking enabled).
    """
    n = len(df)
    closes = df["close"].values

    _get = lambda col: df[col].values if col in df.columns else np.full(n, np.nan)
    ob_bt   = _get("ob_bull_top")
    ob_bb   = _get("ob_bull_bot")
    ob_bl   = _get("ob_bull_wick_low")    # full wick low
    ob_art  = _get("ob_bear_top")
    ob_arb  = _get("ob_bear_bot")
    ob_arh  = _get("ob_bear_wick_high")   # full wick high

    brk_bull_top = np.full(n, np.nan)
    brk_bull_bot = np.full(n, np.nan)
    brk_bear_top = np.full(n, np.nan)
    brk_bear_bot = np.full(n, np.nan)

    cur_brk_bull_top = cur_brk_bull_bot = np.nan
    cur_brk_bear_top = cur_brk_bear_bot = np.nan

    for i in range(1, n):
        c = closes[i]

        # Bull OB broken → becomes bear breaker (supply)
        wick_low = ob_bl[i - 1] if not np.isnan(ob_bl[i - 1]) else ob_bb[i - 1]
        if not np.isnan(ob_bt[i - 1]) and not np.isnan(wick_low) and c < wick_low:
            cur_brk_bear_top = ob_bt[i - 1]
            cur_brk_bear_bot = ob_bb[i - 1]

        # Bear OB broken → becomes bull breaker (demand)
        wick_high = ob_arh[i - 1] if not np.isnan(ob_arh[i - 1]) else ob_art[i - 1]
        if not np.isnan(ob_arb[i - 1]) and not np.isnan(wick_high) and c > wick_high:
            cur_brk_bull_top = ob_art[i - 1]
            cur_brk_bull_bot = ob_arb[i - 1]

        # Invalidate breakers if price closes through them in the wrong direction
        if not np.isnan(cur_brk_bull_top) and c > cur_brk_bull_top:
            cur_brk_bull_top = cur_brk_bull_bot = np.nan
        if not np.isnan(cur_brk_bear_bot) and c < cur_brk_bear_bot:
            cur_brk_bear_top = cur_brk_bear_bot = np.nan

        brk_bull_top[i] = cur_brk_bull_top
        brk_bull_bot[i] = cur_brk_bull_bot
        brk_bear_top[i] = cur_brk_bear_top
        brk_bear_bot[i] = cur_brk_bear_bot

    out = df.copy()
    out["brk_bull_top"] = brk_bull_top
    out["brk_bull_bot"] = brk_bull_bot
    out["brk_bear_top"] = brk_bear_top
    out["brk_bear_bot"] = brk_bear_bot

    c = df["close"]
    out["in_brk_bull"] = (
        c.ge(out["brk_bull_bot"]) & c.le(out["brk_bull_top"]) & out["brk_bull_bot"].notna()
    )
    out["in_brk_bear"] = (
        c.ge(out["brk_bear_bot"]) & c.le(out["brk_bear_top"]) & out["brk_bear_bot"].notna()
    )
    return out


# ── Rejection Blocks ─────────────────────────────────────────────────────────

def detect_rejection_blocks(
    df: pd.DataFrame,
    wick_ratio: float = 2.0,
    min_wick_atr: float = 0.5,
) -> pd.DataFrame:
    """
    Rejection Block: candle with large wick at a key level.

    Bull rejection: lower wick ≥ wick_ratio × body AND ≥ min_wick_atr × ATR
      → Body = entry zone on pullback (buying absorbed the down move)
    Bear rejection: upper wick ≥ wick_ratio × body
      → Body = short entry zone on rally (selling absorbed the up move)

    Invalidated when price closes through the body zone.
    """
    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    atrs   = df["atr"].values if "atr" in df.columns else np.full(len(df), 1e-9)
    n = len(df)

    rej_bull_top = np.full(n, np.nan)
    rej_bull_bot = np.full(n, np.nan)
    rej_bear_top = np.full(n, np.nan)
    rej_bear_bot = np.full(n, np.nan)

    cur_rb_bull_top = cur_rb_bull_bot = np.nan
    cur_rb_bear_top = cur_rb_bear_bot = np.nan

    for i in range(n):
        atr = atrs[i] if (not np.isnan(atrs[i]) and atrs[i] > 0) else 1e-9
        body_top  = max(opens[i], closes[i])
        body_bot  = min(opens[i], closes[i])
        body_size = max(body_top - body_bot, 1e-9)

        lower_wick = body_bot - lows[i]
        upper_wick = highs[i] - body_top

        # Bull rejection: dominated lower wick (strong buying at low)
        if lower_wick >= wick_ratio * body_size and lower_wick >= min_wick_atr * atr:
            cur_rb_bull_top = body_top
            cur_rb_bull_bot = body_bot

        # Bear rejection: dominated upper wick (strong selling at high)
        if upper_wick >= wick_ratio * body_size and upper_wick >= min_wick_atr * atr:
            cur_rb_bear_top = body_top
            cur_rb_bear_bot = body_bot

        # Invalidate if price closes through the block
        c = closes[i]
        if not np.isnan(cur_rb_bull_bot) and c < cur_rb_bull_bot:
            cur_rb_bull_top = cur_rb_bull_bot = np.nan
        if not np.isnan(cur_rb_bear_top) and c > cur_rb_bear_top:
            cur_rb_bear_top = cur_rb_bear_bot = np.nan

        rej_bull_top[i] = cur_rb_bull_top
        rej_bull_bot[i] = cur_rb_bull_bot
        rej_bear_top[i] = cur_rb_bear_top
        rej_bear_bot[i] = cur_rb_bear_bot

    out = df.copy()
    out["rej_bull_top"] = rej_bull_top
    out["rej_bull_bot"] = rej_bull_bot
    out["rej_bear_top"] = rej_bear_top
    out["rej_bear_bot"] = rej_bear_bot

    c = df["close"]
    out["in_rej_bull"] = (
        c.ge(out["rej_bull_bot"]) & c.le(out["rej_bull_top"]) & out["rej_bull_bot"].notna()
    )
    out["in_rej_bear"] = (
        c.ge(out["rej_bear_bot"]) & c.le(out["rej_bear_top"]) & out["rej_bear_bot"].notna()
    )
    return out


# ── Mitigation Blocks ────────────────────────────────────────────────────────

def detect_mitigation_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mitigation Block: OB that was tested (price entered the zone via wick)
    but not invalidated (close stayed above/below OB body).

    On second visit → high-probability second-chance entry because residual
    institutional orders remain unfilled from the first visit.

    Requires ob_bull_top/bot and ob_bear_top/bot in df.
    """
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    n = len(df)

    _get = lambda col: df[col].values if col in df.columns else np.full(n, np.nan)
    ob_bt  = _get("ob_bull_top")
    ob_bb  = _get("ob_bull_bot")
    ob_art = _get("ob_bear_top")
    ob_arb = _get("ob_bear_bot")

    in_mit_bull = np.zeros(n, dtype=bool)
    in_mit_bear = np.zeros(n, dtype=bool)

    bull_ob_key = (np.nan, np.nan)
    bull_touched = False
    bear_ob_key  = (np.nan, np.nan)
    bear_touched = False

    for i in range(1, n):
        cur_bull_key = (ob_bt[i], ob_bb[i])
        cur_bear_key = (ob_art[i], ob_arb[i])

        # New OB formed — reset touch tracking
        if cur_bull_key != bull_ob_key:
            bull_ob_key  = cur_bull_key
            bull_touched = False

        if cur_bear_key != bear_ob_key:
            bear_ob_key  = cur_bear_key
            bear_touched = False

        c = closes[i]

        # Bull OB: detect first touch (wick enters zone, close stays above body bottom)
        if not np.isnan(ob_bt[i]) and not np.isnan(ob_bb[i]):
            if lows[i] <= ob_bt[i]:               # wick dipped into OB
                if not bull_touched and c >= ob_bb[i]:
                    bull_touched = True            # first visit recorded
                elif bull_touched and ob_bb[i] <= c <= ob_bt[i]:
                    in_mit_bull[i] = True          # second visit = mitigation entry

        # Bear OB: detect first touch
        if not np.isnan(ob_art[i]) and not np.isnan(ob_arb[i]):
            if highs[i] >= ob_arb[i]:
                if not bear_touched and c <= ob_art[i]:
                    bear_touched = True
                elif bear_touched and ob_arb[i] <= c <= ob_art[i]:
                    in_mit_bear[i] = True

    out = df.copy()
    out["in_mit_bull"] = in_mit_bull
    out["in_mit_bear"] = in_mit_bear
    return out


# ── Zone Quality Scorer ───────────────────────────────────────────────────────

def compute_zone_quality(
    df: pd.DataFrame,
    direction: str,
    proximity_atr: float = 2.0,
) -> pd.Series:
    """
    Static OB zone quality score [0, 1] — measures HOW STRONG the zone is,
    not whether price is currently inside it.

    This is computed on the HTF (4H/1H) and forward-merged to LTF bars.
    A bar gets the quality of the zone it's approaching/in.

    Priority hierarchy (additive):
      Base (best zone type present):
        OB exists             → 0.55
        Breaker exists        → 0.50  (no OB, but breaker present)
        Neither               → 0.10

      Static overlap bonuses (geometry, not price-position):
        OB zone overlaps FVG  → +0.15  (institutional zone — orders + imbalance)
        OB zone overlaps VI   → +0.15  (strong order flow — body gap)
        OB midpoint near SSL  → +0.20  (long OB near sell-stops = highest priority)
        OB midpoint near BSL  → +0.20  (short OB near buy-stops = highest priority)
        FVG + VI both overlap → +0.05  (triple confluence bonus)
    """
    n = len(df)
    _f = lambda col: df[col].values.astype(float) if col in df.columns else np.full(n, np.nan)

    is_long = direction == "long"

    if is_long:
        ob_top = _f("ob_bull_top");   ob_bot  = _f("ob_bull_bot")
        brk_top= _f("brk_bull_top");  brk_bot = _f("brk_bull_bot")
        fvg_top= _f("fvg_bull_top");  fvg_bot = _f("fvg_bull_bot")
        vi_top = _f("vi_bull_top");   vi_bot  = _f("vi_bull_bot")
        liq    = _f("last_sl")        # long OB near SSL (sell-side stops below)
    else:
        ob_top = _f("ob_bear_top");   ob_bot  = _f("ob_bear_bot")
        brk_top= _f("brk_bear_top");  brk_bot = _f("brk_bear_bot")
        fvg_top= _f("fvg_bear_top");  fvg_bot = _f("fvg_bear_bot")
        vi_top = _f("vi_bear_top");   vi_bot  = _f("vi_bear_bot")
        liq    = _f("last_sh")        # short OB near BSL (buy-side stops above)

    atr = _f("atr")

    ob_exists  = ~np.isnan(ob_bot)
    brk_exists = ~np.isnan(brk_bot)

    # Base quality
    score = np.where(ob_exists, 0.55, np.where(brk_exists, 0.50, 0.10))

    # Geometric overlap: OB zone and FVG zone share price range
    fvg_exists = ~np.isnan(fvg_bot)
    ob_fvg_overlap = (ob_exists & fvg_exists
                      & (ob_top >= fvg_bot) & (ob_bot <= fvg_top))

    # Geometric overlap: OB zone and VI zone share price range
    vi_exists = ~np.isnan(vi_bot)
    ob_vi_overlap = (ob_exists & vi_exists
                     & (ob_top >= vi_bot) & (ob_bot <= vi_top))

    # Liquidity proximity: OB midpoint within proximity_atr of BSL/SSL
    ob_mid = (ob_top + ob_bot) / 2.0
    safe_atr = np.where(np.isnan(atr) | (atr <= 0), 1.0, atr)
    near_liq = (ob_exists & ~np.isnan(liq)
                & (np.abs(ob_mid - liq) <= proximity_atr * safe_atr))

    score = score + 0.15 * ob_fvg_overlap.astype(float)
    score = score + 0.15 * ob_vi_overlap.astype(float)
    score = score + 0.20 * near_liq.astype(float)
    score = score + 0.05 * (ob_fvg_overlap & ob_vi_overlap).astype(float)

    return pd.Series(np.clip(score, 0.0, 1.0), index=df.index)
