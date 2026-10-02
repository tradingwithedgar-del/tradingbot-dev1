"""The trading agent: one cycle = look at every symbol's newest closed bar, manage
what is open, learn from what closed, and decide on new trades."""
from __future__ import annotations

import logging
import random
from datetime import datetime, timezone

import pandas as pd

from . import postmortem as pm
from .analysis.context import MarketContext
from .broker.base import Broker
from .compliance import Compliance
from .config import Settings
from .journal import Journal
from .learning import Learner
from .risk import RiskManager
from .strategies import Signal, Strategy, core_strategies

log = logging.getLogger(__name__)
CHART_PRE_BARS = 60
CHART_MAX_BARS = 260


def _bar_row(ts: pd.Timestamp, bar: pd.Series) -> list:
    return [ts.isoformat(), float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])]


class Agent:
    def __init__(self, settings: Settings, broker: Broker, journal: Journal, rng: random.Random | None = None) -> None:
        self.s = settings
        self.broker = broker
        self.j = journal
        self.risk = RiskManager(settings.risk)
        self.compliance = Compliance(settings.compliance)
        self.learner = Learner(journal, settings.learning, settings.risk, live=settings.is_live, rng=rng)
        self.core: list[Strategy] = core_strategies()
        self.learner.apply_params(self.core)
        self.j.clock = self.now
        self.last_bar: dict[str, str] = journal.get(f"last_bar:{journal.mode}", {}) or {}
        if settings.experiments_allowed:
            self.learner.evolve(self.now())

    # --- helpers ---------------------------------------------------------------
    def now(self) -> str:
        if hasattr(self.broker, "now"):
            return self.broker.now().isoformat()
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def strategies(self) -> list[Strategy]:
        out = list(self.core)
        if self.s.experiments_allowed:
            out += self.learner.experimental_strategies()
        return out

    def _strategy(self, name: str) -> Strategy | None:
        return next((s for s in self.core if s.name == name), None)

    @property
    def halted(self) -> str | None:
        return self.j.get(f"halted:{self.j.mode}")

    def _peak(self, equity: float) -> float:
        peak = max(float(self.j.get(f"peak:{self.j.mode}", 0.0) or 0.0), equity)
        self.j.set(f"peak:{self.j.mode}", peak)
        return peak

    def _day_start(self, equity: float) -> float:
        today = self.now()[:10]
        d = self.j.get(f"day:{self.j.mode}")
        if not d or d.get("date") != today:
            d = {"date": today, "equity": equity}
            self.j.set(f"day:{self.j.mode}", d)
        return float(d["equity"])

    # --- main cycle -------------------------------------------------------------
    def run_cycle(self) -> None:
        self._sync_closed_positions()
        equity, balance = self.broker.equity(), self.broker.balance()
        peak = self._peak(equity)
        self.j.record_equity(balance, equity, peak, ts=self.now())

        for symbol in self.s.symbols:
            try:
                df = self.broker.get_bars(symbol, self.s.timeframe, self.s.history_bars)
            except Exception as e:  # pragma: no cover - network
                log.warning("bars for %s failed: %s", symbol, e)
                continue
            if len(df) < 220:
                continue
            bar_key = df.index[-1].isoformat()
            if self.last_bar.get(symbol) == bar_key:
                continue
            self.last_bar[symbol] = bar_key
            try:
                quote = self.broker.get_quote(symbol)
                spread = quote.spread
            except Exception:  # pragma: no cover - network
                quote, spread = None, 0.0
            ctx = MarketContext(symbol, df, spread=spread)
            self._manage_open(ctx)
            self._advance_shadows(ctx)
            self._advance_watchers(ctx)
            if quote is not None:
                self._look_for_trades(ctx, quote)
        self.j.set(f"last_bar:{self.j.mode}", self.last_bar)

        # drawdown halt
        equity = self.broker.equity()
        peak = self._peak(equity)
        if not self.halted and peak > 0 and (peak - equity) / peak >= self.s.risk.max_drawdown_halt:
            msg = f"HALTED: drawdown {(peak - equity) / peak:.1%} hit the {self.s.risk.max_drawdown_halt:.0%} limit. Review, then run `python -m tradingbot resume`."
            self.j.set(f"halted:{self.j.mode}", msg)
            self.j.event("halt", msg, ts=self.now())
            log.error(msg)

    # --- open trade management -------------------------------------------------------
    def _manage_open(self, ctx: MarketContext) -> None:
        for t in self.j.open_trades(shadow=False):
            if t["symbol"] != ctx.symbol:
                continue
            dec = t.get("decision") or {}
            if not dec.get("filled"):
                px = self.broker.fill_price(t["broker_id"])
                if px is not None:
                    dec["filled"] = True
                    self.j.update_trade(t["id"], entry=px, decision=dec)
                    t["entry"] = px
                else:
                    continue
            self._update_excursions(t, ctx)
            be_at = self.learner.breakeven.get(t["strategy"])
            if be_at and not dec.get("breakeven_moved") and (t["mfe_r"] or 0) >= be_at:
                try:
                    if self.broker.modify_stop(t["broker_id"], t["entry"]):
                        dec["breakeven_moved"] = True
                        self.j.update_trade(t["id"], decision=dec)
                        self.j.event("manage", f"#{t['id']} {t['symbol']} stop moved to break-even at +{be_at}R", ts=self.now())
                except Exception:  # pragma: no cover - network
                    log.exception("break-even move failed")

    def _update_excursions(self, t: dict, ctx: MarketContext) -> None:
        bar = ctx.bar
        risk = abs(t["entry"] - t["stop"]) or 1e-12
        if t["side"] == "buy":
            fav, adv = (bar["high"] - t["entry"]) / risk, (t["entry"] - bar["low"]) / risk
        else:
            fav, adv = (t["entry"] - bar["low"]) / risk, (bar["high"] - t["entry"]) / risk
        chart = t.get("chart") or []
        chart.append(_bar_row(ctx.time, bar))
        t["mfe_r"] = max(t["mfe_r"] or 0.0, float(fav))
        t["mae_r"] = max(t["mae_r"] or 0.0, float(adv))
        self.j.update_trade(t["id"], mfe_r=t["mfe_r"], mae_r=t["mae_r"], bars_held=(t["bars_held"] or 0) + 1,
                            chart=chart[-CHART_MAX_BARS:])

    def _sync_closed_positions(self) -> None:
        open_ids = {p.id for p in self.broker.open_positions()}
        for t in self.j.open_trades(shadow=False):
            if t["broker_id"] in open_ids:
                continue
            info = self.broker.closed_info(t["broker_id"], t["symbol"], t["side"], t["stop"], t["take_profit"])
            if info is None:
                continue
            entry = info.entry_price if info.entry_price is not None else t["entry"]
            direction = 1 if t["side"] == "buy" else -1
            risk_dist = abs(t["entry"] - t["stop"]) or 1e-12
            if info.pnl is not None and t["risk_amount"]:
                r = info.pnl / t["risk_amount"]
                pnl = info.pnl
            else:
                r = (info.exit_price - entry) * direction / risk_dist
                pnl = r * (t["risk_amount"] or 0.0)
            self._close_trade(t, info.exit_price, info.exit_time.isoformat(), pnl, r, entry=entry, estimated=info.estimated)

    def _close_trade(self, t: dict, exit_price: float, exit_time: str, pnl: float, r: float, entry: float | None = None,
                     estimated: bool = False) -> None:
        t.update(exit_price=exit_price, r_multiple=r, pnl=pnl)
        if entry is not None:
            t["entry"] = entry
        outcome = "win" if r > 0.05 else ("loss" if r < -0.05 else "breakeven")
        tags = pm.initial_tags(t)
        atr = float((t.get("features") or {}).get("atr") or 0.0)
        watch = pm.new_watch(t, atr)
        self.j.update_trade(t["id"], status="closed", closed_at=exit_time, exit_price=exit_price, pnl=pnl,
                            r_multiple=r, outcome=outcome, entry=t["entry"], exit_estimated=int(estimated),
                            postmortem={"tags": tags, "notes": pm.explain(tags), "final": False}, watch=watch)
        t["outcome"] = outcome
        self.learner.record(t)
        kind = "shadow" if t.get("shadow") else "trade"
        self.j.event(kind, f"#{t['id']} {t['strategy']} {t['symbol']} {t['side']} closed {r:+.2f}R ({outcome})"
                           + (f" - {', '.join(tags)}" if tags else ""), {"trade_id": t["id"]}, ts=exit_time)
        if t.get("experimental") and self.s.experiments_allowed:
            self.learner.evolve(self.now())

    # --- shadow trades -----------------------------------------------------------------
    def _advance_shadows(self, ctx: MarketContext) -> None:
        bar = ctx.bar
        hi, lo = float(bar["high"]), float(bar["low"])
        for t in self.j.open_trades(shadow=True):
            if t["symbol"] != ctx.symbol or t["opened_at"] >= ctx.time.isoformat():
                continue
            self._update_excursions(t, ctx)
            exit_price = None
            if t["side"] == "buy":
                if lo <= t["stop"]:
                    exit_price = t["stop"]
                elif hi >= t["take_profit"]:
                    exit_price = t["take_profit"]
            else:
                if hi >= t["stop"]:
                    exit_price = t["stop"]
                elif lo <= t["take_profit"]:
                    exit_price = t["take_profit"]
            if exit_price is not None:
                d = 1 if t["side"] == "buy" else -1
                r = (exit_price - t["entry"]) * d / (abs(t["entry"] - t["stop"]) or 1e-12)
                self._close_trade(t, exit_price, ctx.time.isoformat(), 0.0, r)

    def _advance_watchers(self, ctx: MarketContext) -> None:
        bar = ctx.bar
        for t in self.j.watching_trades():
            if t["symbol"] != ctx.symbol or (t["closed_at"] or "") >= ctx.time.isoformat():
                continue
            w = pm.update_watch(t, t["watch"], float(bar["high"]), float(bar["low"]))
            chart = (t.get("chart") or []) + [_bar_row(ctx.time, bar)]
            resolved_loss = (t["r_multiple"] or 0) < 0 and (w["hit_target"] or w["beyond_stop"])
            if w["bars"] < self.s.learning.postmortem_watch_bars and not resolved_loss:
                self.j.update_trade(t["id"], watch=w, chart=chart[-CHART_MAX_BARS:])
                continue
            tags = pm.final_tags(t, w)
            self.j.update_trade(t["id"], watch=None, chart=chart[-CHART_MAX_BARS:],
                                postmortem={"tags": tags, "notes": pm.explain(tags), "final": True, "after_exit": w})
            strat = self._strategy(t["strategy"])
            if strat is not None:
                recent = self.j.trades("strategy = ? AND status = 'closed' AND watch IS NULL AND mode = ?",
                                       (t["strategy"], self.j.mode), order="id DESC", limit=self.s.learning.tune_window)
                self.learner.tune(strat, recent)

    # --- entries -------------------------------------------------------------------------
    def _look_for_trades(self, ctx: MarketContext, quote) -> None:
        signals: list[Signal] = []
        for strat in self.strategies():
            try:
                sig = strat.generate(ctx)
            except Exception:
                log.exception("strategy %s failed", strat.name)
                continue
            if sig is not None and sig.risk_distance > 0:
                sig.set_target(self.s.risk.reward_multiple)
                signals.append(sig)
        if not signals:
            return
        core_names = {s.name for s in self.core}
        for sig in signals:
            sig.features["confluence"] = sum(1 for o in signals if o.side == sig.side and o.strategy in core_names)
            sig.features["symbol"] = ctx.symbol
        # Strongest confluence first, so the best setup gets the real-money slot.
        signals.sort(key=lambda s: (-s.features["confluence"], s.experimental))

        open_real = self.j.open_trades(shadow=False)
        open_shadow = {(t["strategy"], t["symbol"]) for t in self.j.open_trades(shadow=True)}
        real_symbols = {t["symbol"] for t in open_real}

        for sig in signals:
            strat = next(s for s in self.strategies() if s.name == sig.strategy)
            decision = self.learner.decide(strat, sig.features, ctx.symbol, self.s.live_approved_strategies)
            why = list(decision.reasons)
            action = decision.action
            if action == "skip":
                continue
            if action == "trade":
                block = self._entry_blocker(sig, ctx, quote, open_real, real_symbols)
                if block:
                    action = "shadow"
                    why.append(block)
            dec = {"action": action, "p": round(decision.p, 3), "expected_r": round(decision.expected_r, 3),
                   "stat_key": decision.stat_key, "why": why}
            if action == "trade":
                if self._open_real(sig, ctx, dec, open_real):
                    real_symbols.add(sig.symbol)
                    open_real = self.j.open_trades(shadow=False)
                    continue
                dec["action"] = "shadow"
            if (sig.strategy, sig.symbol) not in open_shadow:
                self._open_shadow(sig, ctx, dec)
                open_shadow.add((sig.strategy, sig.symbol))

    def _entry_blocker(self, sig: Signal, ctx: MarketContext, quote, open_real: list[dict], real_symbols: set) -> str | None:
        if self.halted:
            return "agent halted"
        if self.s.stop_file.exists():
            return "STOP file present - no new real entries"
        if sig.symbol in real_symbols:
            return "already in a real trade on this symbol"
        equity = self.broker.equity()
        peak = self._peak(equity)
        open_risk = sum(t["risk_pct"] or 0 for t in open_real)
        limit = self.risk.check_limits(equity, peak, self._day_start(equity), open_risk, len(open_real))
        if limit:
            return limit
        now = None if hasattr(self.broker, "now") else pd.Timestamp.now(tz="UTC")
        return self.compliance.check(sig.side, sig.symbol, sig.entry, sig.stop, ctx.close, ctx.atr_now, quote,
                                     self.broker.open_positions(), now)

    def _open_real(self, sig: Signal, ctx: MarketContext, dec: dict, open_real: list[dict]) -> bool:
        equity = self.broker.equity()
        peak = self._peak(equity)
        open_risk = sum(t["risk_pct"] or 0 for t in open_real)
        try:
            spec = self.broker.spec(sig.symbol)
        except Exception as e:  # pragma: no cover - network
            dec["why"].append(f"instrument spec failed: {e}")
            return False
        size = self.risk.size(equity, peak, sig.entry, sig.stop, spec, open_risk)
        if not size.ok:
            dec["why"].append(size.reason)
            return False
        try:
            pid = self.broker.place_market(sig.symbol, sig.side, size.qty, sig.stop, sig.take_profit, sig.strategy)
        except Exception as e:
            dec["why"].append(f"order failed: {e}")
            self.j.event("error", f"order failed {sig.symbol} {sig.side}: {e}", ts=self.now())
            return False
        self.compliance.record_order()
        tid = self.j.open_trade(
            shadow=0, symbol=sig.symbol, strategy=sig.strategy, experimental=int(sig.experimental), side=sig.side,
            opened_at=ctx.time.isoformat(), entry=sig.entry, stop=sig.stop, take_profit=sig.take_profit, qty=size.qty,
            risk_amount=size.risk_amount, risk_pct=size.risk_pct, reason=sig.reason, features=sig.features,
            decision=dec, broker_id=pid, chart=self._chart(ctx),
        )
        self.j.event("trade", f"#{tid} OPEN {sig.strategy} {sig.symbol} {sig.side} risk {size.risk_pct:.1%} "
                              f"({size.risk_amount:.2f}) - {sig.reason}", {"trade_id": tid}, ts=ctx.time.isoformat())
        return True

    def _open_shadow(self, sig: Signal, ctx: MarketContext, dec: dict) -> None:
        half = ctx.spread / 2
        entry = sig.entry + half if sig.side == "buy" else sig.entry - half
        self.j.open_trade(
            shadow=1, symbol=sig.symbol, strategy=sig.strategy, experimental=int(sig.experimental), side=sig.side,
            opened_at=ctx.time.isoformat(), entry=entry, stop=sig.stop, take_profit=sig.take_profit, qty=0,
            risk_amount=0, risk_pct=0, reason=sig.reason, features=sig.features, decision=dec, chart=self._chart(ctx),
        )

    @staticmethod
    def _chart(ctx: MarketContext) -> list:
        tail = ctx.df.iloc[-CHART_PRE_BARS:]
        return [_bar_row(ts, row) for ts, row in tail.iterrows()]
