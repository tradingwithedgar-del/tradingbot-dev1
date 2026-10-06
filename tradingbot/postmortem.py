"""Trade post-mortems: why did this trade win or lose?

Each closed trade gets tags at close time, then is "watched" for a number of bars
after exit to see what price did next (did the stop get hunted and price go on to the
target? did a win leave a lot on the table?). Tags drive the learner's tuning.
"""
from __future__ import annotations

TAG_NOTES = {
    "gave_back_profit": "Was at least +1R in profit before reversing to a loss. Protecting at break-even may help.",
    "immediately_wrong": "Never moved meaningfully in favour (<0.3R). Entry thesis or timing was wrong.",
    "against_htf_bias": "Traded against the higher-timeframe bias (EMA50/200).",
    "against_structure": "Traded against the HH/HL - LH/LL structure.",
    "high_volatility_loss": "Lost in a high-volatility regime.",
    "stop_too_tight": "Stopped out, then price reached the original target without going much further against. Stop was likely too tight.",
    "thesis_invalid": "After the stop, price kept going against the trade. The stop did its job.",
    "target_too_close": "After the target, price ran a further 2R+. Target may have been conservative.",
    "clean_win": "Reached target without drawing down more than 0.5R.",
    "closed_manually": "You closed this trade in TradeLocker. The agent keeps following the original plan virtually to see what it would have done.",
    "closed_by_agent": "The agent closed this trade early (e.g. before earnings).",
    "survived_drawdown_win": "Won after drawing down more than 0.7R first.",
}


def initial_tags(trade: dict) -> list[str]:
    f = trade.get("features") or {}
    r = trade.get("r_multiple") or 0.0
    mfe, mae = trade.get("mfe_r") or 0.0, trade.get("mae_r") or 0.0
    side = trade["side"]
    tags: list[str] = []
    if r < 0:
        if mfe >= 1.0:
            tags.append("gave_back_profit")
        elif mfe < 0.3:
            tags.append("immediately_wrong")
        if (side == "buy" and f.get("bias") == "bear") or (side == "sell" and f.get("bias") == "bull"):
            tags.append("against_htf_bias")
        st = f.get("structure", "")
        if (side == "buy" and st.startswith("down")) or (side == "sell" and st.startswith("up")):
            tags.append("against_structure")
        if f.get("volatility") == "high_vol":
            tags.append("high_volatility_loss")
    elif r > 0:
        tags.append("clean_win" if mae < 0.5 else ("survived_drawdown_win" if mae > 0.7 else "win"))
    return tags


def new_watch(trade: dict, atr: float) -> dict:
    return {"bars": 0, "atr": atr, "hit_target": False, "beyond_stop": False, "max_after_r": 0.0}


def update_watch(trade: dict, watch: dict, high: float, low: float) -> dict:
    """Advance the post-exit watcher by one bar."""
    side = trade["side"]
    entry, stop, tp = trade["entry"], trade["stop"], trade["take_profit"]
    risk = abs(entry - stop) or 1e-12
    ext = watch["atr"]
    watch["bars"] += 1
    if side == "buy":
        fav, adv = high, low
        if not watch["beyond_stop"] and not watch["hit_target"]:
            if adv < stop - ext:
                watch["beyond_stop"] = True
            elif fav >= tp:
                watch["hit_target"] = True
        watch["max_after_r"] = max(watch["max_after_r"], (fav - tp) / risk)
    else:
        fav, adv = low, high
        if not watch["beyond_stop"] and not watch["hit_target"]:
            if adv > stop + ext:
                watch["beyond_stop"] = True
            elif fav <= tp:
                watch["hit_target"] = True
        watch["max_after_r"] = max(watch["max_after_r"], (tp - fav) / risk)
    return watch


def final_tags(trade: dict, watch: dict) -> list[str]:
    tags = list((trade.get("postmortem") or {}).get("tags", []))
    r = trade.get("r_multiple") or 0.0
    if r < 0:
        if watch["hit_target"]:
            tags.append("stop_too_tight")
        elif watch["beyond_stop"]:
            tags.append("thesis_invalid")
    elif r > 0 and watch["max_after_r"] >= 2.0:
        tags.append("target_too_close")
    return tags


def explain(tags: list[str]) -> list[str]:
    return [TAG_NOTES[t] for t in tags if t in TAG_NOTES]
