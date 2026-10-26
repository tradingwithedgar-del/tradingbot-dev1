"""The trading agent: one cycle = look at every symbol's newest closed bar, manage
what is open, learn from what closed, and decide on new trades."""
from __future__ import annotations

import logging
import random
import time
from datetime import datetime, timezone

import pandas as pd

from . import postmortem as pm
from .analysis.context import MarketContext
from . import plan as weekly
from . import protect
from .analysis import htf
from .analysis import candles as cdl
from .analysis.view import chart_view
from .broker.base import TIMEFRAME_SECONDS, Broker
from .compliance import Compliance
from .config import Settings
from .journal import Journal
from .learning import Learner
from .risk import RiskManager
from .learning import CANDIDATE
from .news.assets import asset_class
from .sessions import entry_block, flatten_reason
from .strategies import Signal, Strategy, load_library

log = logging.getLogger(__name__)
CHART_PRE_BARS = 60
CHART_MAX_BARS = 2000   # ~20 days of 15m candles per trade


def _trim_chart(rows: list) -> list:
    """Keep the candles before entry and the most recent ones; never drop the start of a trade
    unless it has run for weeks."""
    if len(rows) <= CHART_MAX_BARS:
        return rows
    return rows[:CHART_PRE_BARS] + rows[-(CHART_MAX_BARS - CHART_PRE_BARS):]


def _bar_row(ts: pd.Timestamp, bar: pd.Series) -> list:
    return [ts.isoformat(), float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])]


CYCLE_BUDGET_SECONDS = 150     # live: stop starting new markets after this long; the rest wait for the next cycle
PLAN_BATCH = 25                # markets planned per cycle at the start of a week


class Agent:
    def __init__(self, settings: Settings, broker: Broker, journal: Journal, rng: random.Random | None = None,
                 news=None, earnings=None, desk=None) -> None:
        self.s = settings
        self.broker = broker
        self.j = journal
        self.risk = RiskManager(settings.risk)
        self.compliance = Compliance(settings.compliance)
        self.learner = Learner(journal, settings.learning, settings.risk, live=settings.is_live, rng=rng)
        self.news = news                      # NewsMonitor, or None (backtests)
        self.earnings = earnings              # EarningsDesk, or None
        self.desk = desk                      # trading desk (Claude agents), or None
        self._snap: dict[str, dict] = {}      # per-market snapshot for the Research Analyst
        self._lab_seen = ""
        self._started = pd.Timestamp.now(tz="UTC")
        self._htf_cache: dict[str, tuple[str, dict]] = {}
        self._rot = 0                         # where the last cycle stopped (fair rotation over many markets)
        self._plan_retry: dict[str, pd.Timestamp] = {}   # markets that couldn't be planned yet (no data): try later
        # The strategy library: built-ins plus anything in my_strategies/ (minus disabled ones).
        self.core: list[Strategy] = load_library(settings.db_path.parent, journal=journal)
        self.news_strats: list[Strategy] = [st for st in self.core if "news" in st.needs]
        self.learner.apply_params(self.core)
        self.last_fast: dict[str, str] = {}
        self._tf = pd.Timedelta(seconds=TIMEFRAME_SECONDS[settings.timeframe])
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
        if self.s.mode == "demo":
            # each strategy is shadowed by a tweaked variant; the better version wins over time
            out += [c for c in (self.learner.challenger_for(st) for st in self.core) if c is not None]
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
    def _news_state(self, symbol: str, now: pd.Timestamp) -> dict:
        if not self.news:
            return {}
        try:
            return self.news.state(symbol, now)
        except Exception:  # pragma: no cover - defensive
            log.exception("news state failed")
            return {}

    def _earnings_state(self, symbol: str, now: pd.Timestamp) -> dict:
        if not self.earnings:
            return {}
        try:
            return self.earnings.context(symbol, now)
        except Exception:
            log.exception("earnings context failed for %s", symbol)
            return {}

    def run_cycle(self) -> None:
        now_ts = pd.Timestamp(self.now())
        if self.news:
            try:
                self.news.refresh(now_ts)
            except Exception:
                log.exception("news refresh failed")
            self._flatten_for_news(now_ts)
        self._flatten_for_session(now_ts)
        self._sync_closed_positions()
        self._merge_lab()
        self._ensure_plans(now_ts)
        if self.desk:
            self._desk_results()
        equity, balance = self.broker.equity(), self.broker.balance()
        peak = self._peak(equity)
        self.j.record_equity(balance, equity, peak, ts=self.now())

        live_clock = not hasattr(self.broker, "now")
        tf = pd.Timedelta(seconds=TIMEFRAME_SECONDS[self.s.timeframe])
        syms = list(self.s.symbols)
        if live_clock and len(syms) > 1:
            k = self._rot % len(syms)
            syms = syms[k:] + syms[:k]
        deadline = time.monotonic() + CYCLE_BUDGET_SECONDS
        for n, symbol in enumerate(syms):
            if live_clock and n and time.monotonic() > deadline:
                self._rot += n              # next cycle starts where this one stopped
                log.info("cycle time budget used: %d of %d markets this round, the rest next", n, len(syms))
                break
            if live_clock and symbol in self.last_bar:
                # The next bar closes at last bar's open + 2 timeframes; don't hammer the API before that.
                if pd.Timestamp.now(tz="UTC") < pd.Timestamp(self.last_bar[symbol]) + 2 * tf:
                    continue
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
            ctx = MarketContext(symbol, df, spread=spread, news=self._news_state(symbol, now_ts),
                                earnings=self._earnings_state(symbol, now_ts), htf=self._htf_frames(symbol, df))
            self._check_plan(symbol, ctx, now_ts)
            c = df["close"]
            self._snap[symbol] = {"price": float(c.iloc[-1]), "trend": ctx.trend,
                                  "chg_1d": float(c.iloc[-1] / c.iloc[-min(len(c), 96)] * 100 - 100),
                                  "chg_5d": float(c.iloc[-1] / c.iloc[0] * 100 - 100)}
            if live_clock:
                try:
                    self.j.set(f"view:{self.j.mode}:{symbol}", self._view_with_htf(ctx))
                except Exception:
                    log.exception("chart view failed for %s", symbol)
            self._manage_open(ctx)
            self._advance_shadows(ctx)
            self._advance_watchers(ctx)
            if quote is not None:
                self._look_for_trades(ctx, quote)
        self.j.set(f"last_bar:{self.j.mode}", self.last_bar)
        # The first brief waits until every market has been seen (or 30 min), so none is left out.
        if self.desk and (len(self._snap) >= len(self.s.symbols)
                          or pd.Timestamp.now(tz="UTC") - self._started > pd.Timedelta(minutes=30)):
            self.desk.research_tick(now_ts, self._snap)
        if self.news and live_clock:
            self._fast_news_pass(now_ts)

        # drawdown halt
        equity = self.broker.equity()
        peak = self._peak(equity)
        if not self.halted and peak > 0 and (peak - equity) / peak >= self.s.risk.max_drawdown_halt:
            msg = f"HALTED: drawdown {(peak - equity) / peak:.1%} hit the {self.s.risk.max_drawdown_halt:.0%} limit. Review, then run `python -m tiim resume`."
            self.j.set(f"halted:{self.j.mode}", msg)
            self.j.event("halt", msg, ts=self.now())
            log.error(msg)

    # --- top-down (higher timeframe) analysis ---------------------------------------------
    def _htf_frames(self, symbol: str, df: pd.DataFrame) -> dict:
        """Monthly/Weekly/Daily/4H/1H read, refreshed once an hour (cached in between)."""
        # 1H breaks matter, so refresh every hour; each market on its own minute so they don't all refresh together
        offset = pd.Timedelta(minutes=sum(map(ord, symbol)) % 60)
        key = (df.index[-1] - offset).floor("1h").isoformat()
        hit = self._htf_cache.get(symbol)
        if hit and hit[0] == key:
            return hit[1]
        frames = {}
        try:
            if getattr(self.broker, "supports_htf", False):
                daily = self.broker.get_bars(symbol, "1D", 1500, max_days=2300)
                h4 = self.broker.get_bars(symbol, "4H", 400, max_days=120)
                h1 = self.broker.get_bars(symbol, "1H", 400, max_days=40)
            else:   # backtests: build them from the lower-timeframe history
                base = self.broker.get_bars(symbol, self.s.timeframe, 3000)
                daily, h4, h1 = htf.resample(base, "1D"), htf.resample(base, "4h"), htf.resample(base, "1h")
            frames = htf.build(daily, h4, h1)
        except Exception as e:  # pragma: no cover - network
            log.warning("higher-timeframe bars for %s failed: %s", symbol, e)
        self._htf_cache[symbol] = (key, frames)
        return frames

    def _view_with_htf(self, ctx: MarketContext) -> dict:
        v = chart_view(ctx)
        v["candles"] = cdl.scan(ctx.df, ctx.atr.to_numpy(), bars=len(v["t"]))
        b, score = htf.bias(ctx.htf)
        v["htf"] = {"bias": b, "score": score, "frames": [ctx.htf[tf].to_dict() for tf in htf.FRAMES if tf in ctx.htf]}
        lo, hi = min(v["low"][-200:] or [0]), max(v["high"][-200:] or [0])
        span = (hi - lo) or 1.0
        for tf in ("1W", "1D", "4H"):
            f = ctx.htf.get(tf)
            for z in (f.zones if f else []):
                if z["top"] >= lo - span and z["bottom"] <= hi + span:   # near the visible price range
                    v["zones"].append({"kind": z["kind"], "top": z["top"], "bottom": z["bottom"], "from": v["t"][0],
                                       "fresh": z["fresh"], "touches": 0, "tf": htf.SHORT[tf]})
        return v

    # --- weekly plan ----------------------------------------------------------------------
    @property
    def _plan_key(self) -> str:
        return f"plan:{self.j.mode}"

    def _book(self, now: pd.Timestamp) -> dict:
        wk = weekly.week_key(now)
        book = self.j.get(self._plan_key) or {}
        if book.get("week") != wk:
            if book.get("week"):
                self.j.set(f"plan_prev:{self.j.mode}", book)
            book = {"week": wk, "created": now.isoformat(), "markets": {}, "briefing": "", "notes": {}}
        return book

    def plan_for(self, symbol: str) -> dict | None:
        return ((self.j.get(self._plan_key) or {}).get("markets") or {}).get(symbol)

    def _ensure_plans(self, now: pd.Timestamp) -> None:
        """Plan every market for the week (from Sunday 18:00 UTC), even ones whose market is still closed."""
        book = self._book(now)
        missing = [s for s in self.s.symbols if s not in book["markets"]
                   and self._plan_retry.get(s, now - pd.Timedelta(seconds=1)) <= now][:PLAN_BATCH]
        if not missing and book.get("week") == (self.j.get(self._plan_key) or {}).get("week"):
            self._maybe_brief(book, now)
            return
        for sym in missing:
            try:
                df = self.broker.get_bars(sym, self.s.timeframe, self.s.history_bars)
                frames = self._htf_frames(sym, df) if len(df) >= 60 else {}
                if not frames:
                    self._plan_retry[sym] = now + pd.Timedelta(hours=1)
                    continue
                book["markets"][sym] = weekly.market_plan(sym, frames, float(df["close"].iloc[-1]), now)
                self.j.event("plan", f"{book['week']} plan {sym}: {book['markets'][sym]['summary']}", ts=self.now())
            except Exception as e:  # pragma: no cover - network
                self._plan_retry[sym] = now + pd.Timedelta(hours=1)
                log.warning("weekly plan for %s failed: %s", sym, e)
        self.j.set(self._plan_key, book)
        self._maybe_brief(book, now)

    def _check_plan(self, symbol: str, ctx: MarketContext, now: pd.Timestamp) -> None:
        """A daily close beyond the plan's invalidation level means the plan was wrong: re-plan that market."""
        book = self.j.get(self._plan_key) or {}
        p = (book.get("markets") or {}).get(symbol)
        day = ctx.htf.get("1D")
        if not p or not day or not weekly.invalidated(p, day.close):
            return
        old = dict(p, status="invalidated")
        book.setdefault("history", []).append(old)
        book["markets"][symbol] = dict(weekly.market_plan(symbol, ctx.htf, ctx.close, now),
                                       replanned_from=old["summary"])
        self.j.set(self._plan_key, book)
        self.j.event("plan", f"{symbol}: daily close beyond {old['invalidation']:.5g} proved the weekly plan wrong "
                             f"- re-planned: {book['markets'][symbol]['summary']}", ts=self.now())

    def _maybe_brief(self, book: dict, now: pd.Timestamp) -> None:
        if (self.desk and not book.get("briefing") and not book.get("briefing_asked")
                and len(book["markets"]) >= 0.9 * len(self.s.symbols) and self.desk.available):
            book["briefing_asked"] = True
            self.j.set(self._plan_key, book)
            self.desk.weekly_briefing(book["week"], now)

    # --- news handling -------------------------------------------------------------------
    def _flatten_for_news(self, now: pd.Timestamp) -> None:
        """Close open stock trades shortly before earnings (gap risk is not something to gamble on)."""
        for t in self.j.open_trades(shadow=False):
            reason = self.news.must_flatten(t["symbol"], now)
            if not reason:
                continue
            dec = t.get("decision") or {}
            if dec.get("closed_by_agent"):
                continue
            try:
                self.broker.close_position(t["broker_id"])
                dec["closed_by_agent"] = reason
                self.j.update_trade(t["id"], decision=dec)
                self.j.event("manage", f"#{t['id']} {t['symbol']} {reason}", ts=self.now())
            except Exception:
                log.exception("could not close #%s before earnings", t["id"])

    def _flatten_for_session(self, now: pd.Timestamp) -> None:
        """Be flat before stock closes and weekend closes: stops don't protect across a gap."""
        for t in self.j.open_trades(shadow=False):
            reason = flatten_reason(t["symbol"], now)
            dec = t.get("decision") or {}
            if not reason or dec.get("closed_by_agent"):
                continue
            try:
                self.broker.close_position(t["broker_id"])
                dec["closed_by_agent"] = reason
                self.j.update_trade(t["id"], decision=dec)
                self.j.event("manage", f"#{t['id']} {t['symbol']} {reason}", {"trade_id": t["id"]}, ts=self.now())
            except Exception:
                log.exception("could not close #%s before the close", t["id"])

    def _fast_news_pass(self, now: pd.Timestamp) -> None:
        """While news is driving a symbol, check the news strategies on 5-minute bars between 15m closes."""
        for symbol in self.news.hot_symbols(now):
            try:
                df = self.broker.get_bars(symbol, self.s.news.fast_timeframe, 300)
            except Exception as e:  # pragma: no cover - network
                log.warning("fast bars for %s failed: %s", symbol, e)
                continue
            if len(df) < 60:
                continue
            key = df.index[-1].isoformat()
            if self.last_fast.get(symbol) == key:
                continue
            self.last_fast[symbol] = key
            try:
                quote = self.broker.get_quote(symbol)
            except Exception:  # pragma: no cover - network
                continue
            ctx = MarketContext(symbol, df, spread=quote.spread, news=self._news_state(symbol, now),
                                earnings=self._earnings_state(symbol, now))
            self._look_for_trades(ctx, quote, only=self.news_strats)

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
            if self._news_guard(t, ctx, dec):
                continue
            if self._time_up(t) and not dec.get("closed_by_agent"):
                try:
                    self.broker.close_position(t["broker_id"])
                    dec["closed_by_agent"] = f"time stop: scalp still open after {t['bars_held']} candles"
                    self.j.update_trade(t["id"], decision=dec)
                    self.j.event("manage", f"#{t['id']} {t['symbol']} {dec['closed_by_agent']} - closed", ts=self.now())
                except Exception:  # pragma: no cover - network
                    log.exception("time stop failed for #%s", t["id"])
                continue
            self._protect(t, ctx, dec, real=True)

    def _protect(self, t: dict, ctx: MarketContext, dec: dict, real: bool) -> None:
        """Move the stop toward profit as the trade works (break-even plus costs, then small locked profits,
        then behind the latest swing). Only ever tightens; the original stop stays on record for R."""
        if not self.s.profit_protection or dec.get("closed_by_agent"):
            return
        if dec.get("protect_retry_at", 0) > (t["bars_held"] or 0):
            return
        scale = self.learner.protection_scale(t["strategy"])
        idx = ctx.df.index.get_indexer([pd.Timestamp(t["opened_at"])], method="nearest")[0]
        step = protect.plan(t, dec, ctx, scale=scale, be_at=self.learner.breakeven.get(t["strategy"]), swing_after=int(idx))
        if step is None:
            return
        new, label = step
        if real:
            try:
                if not self.broker.modify_stop(t["broker_id"], new):
                    raise RuntimeError("broker refused the stop")
            except Exception as e:  # pragma: no cover - network
                dec["protect_retry_at"] = (t["bars_held"] or 0) + 3
                self.j.update_trade(t["id"], decision=dec)
                log.warning("stop move for #%s failed: %s", t["id"], e)
                return
        old = dec.get("stop_now", t["stop"])
        dec["stop_now"], dec["protected"] = float(new), label
        dec["breakeven_moved"] = True
        self.j.update_trade(t["id"], decision=dec)
        risk = abs(t["entry"] - t["stop"]) or 1e-12
        locked = (new - t["entry"]) * (1 if t["side"] == "buy" else -1) / risk
        self.j.event("manage", f"#{t['id']} {t['symbol']} {'' if real else '(virtual) '}stop {old:.5g} -> {new:.5g}: "
                               f"{label} ({locked:+.2f}R secured, trade reached +{t['mfe_r']:.1f}R)",
                     {"trade_id": t["id"]}, ts=self.now())

    def _news_guard(self, t: dict, ctx: MarketContext, dec: dict) -> bool:
        """Major news broke against an open trade: protect it. Returns True if the trade was closed."""
        n = ctx.news or {}
        if not self.news or n.get("news_state") != "breaking" or (n.get("news_impact") or 0) < self.s.news.guard_min_impact:
            return False
        against = (n.get("news_dir", 0) > 0) != (t["side"] == "buy") and n.get("news_dir", 0) != 0
        title = n.get("news_title", "")
        if not against or dec.get("news_guard") == title or (n.get("news_age_min") or 0) > 30:
            return False
        risk = abs(t["entry"] - t["stop"]) or 1e-12
        cur_r = (ctx.close - t["entry"]) * (1 if t["side"] == "buy" else -1) / risk
        dec["news_guard"] = title
        action = "noted - stop left where it is (it already caps the loss)"
        closed = False
        try:
            if cur_r >= self.s.news.guard_close_r:
                self.broker.close_position(t["broker_id"])
                dec["closed_by_agent"] = f"took profit at {cur_r:+.1f}R: major news against the trade"
                action, closed = f"took the profit at {cur_r:+.1f}R", True
            elif cur_r >= self.s.news.guard_breakeven_r and not dec.get("breakeven_moved"):
                if self.broker.modify_stop(t["broker_id"], t["entry"]):
                    dec["breakeven_moved"] = True
                    dec["stop_now"] = float(t["entry"])
                    action = f"stop moved to break-even (trade was {cur_r:+.1f}R)"
        except Exception:
            log.exception("news guard action failed for #%s", t["id"])
            action = "tried to protect the trade but the broker call failed"
        self.j.update_trade(t["id"], decision=dec)
        self.j.event("manage", f"#{t['id']} {t['symbol']} {t['side']}: major news against the trade "
                               f"('{title[:70]}') - {action}", {"trade_id": t["id"]}, ts=self.now())
        return closed

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
        t["bars_held"] = (t["bars_held"] or 0) + 1
        self.j.update_trade(t["id"], mfe_r=t["mfe_r"], mae_r=t["mae_r"], bars_held=t["bars_held"],
                            chart=_trim_chart(chart))

    @staticmethod
    def _time_up(t: dict) -> bool:
        """Scalps have a time stop: out after max_bars candles if neither stop nor target was hit."""
        n = int((t.get("features") or {}).get("max_bars") or 0)
        return n > 0 and (t["bars_held"] or 0) >= n

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
            reason = self._close_reason(t, info.exit_price, entry, info.estimated)
            self._close_trade(t, info.exit_price, info.exit_time.isoformat(), pnl, r, entry=entry, estimated=info.estimated,
                              reason=reason)

    @staticmethod
    def _close_reason(t: dict, exit_price: float, entry: float, estimated: bool) -> str:
        """target | stop | breakeven | agent | manual (you closed it in TradeLocker)."""
        dec = t.get("decision") or {}
        if dec.get("closed_by_agent"):
            return "agent"
        if estimated:
            return "stop" if abs(exit_price - t["stop"]) < abs(exit_price - t["take_profit"]) else "target"
        tol = 0.15 * (abs(entry - t["stop"]) or 1e-12)
        if abs(exit_price - t["take_profit"]) <= tol:
            return "target"
        if abs(exit_price - t["stop"]) <= tol:
            return "stop"
        if dec.get("stop_now") is not None and abs(exit_price - dec["stop_now"]) <= tol:
            return "protected"     # the moved (profit-protecting) stop did its job
        if (exit_price - t["stop"]) * (1 if t["side"] == "buy" else -1) < 0:
            return "stop"   # filled beyond the stop: the market gapped over it
        if dec.get("breakeven_moved") and abs(exit_price - entry) <= tol:
            return "protected"
        return "manual"

    def _close_trade(self, t: dict, exit_price: float, exit_time: str, pnl: float, r: float, entry: float | None = None,
                     estimated: bool = False, reason: str = "") -> None:
        t.update(exit_price=exit_price, r_multiple=r, pnl=pnl)
        if entry is not None:
            t["entry"] = entry
        outcome = "win" if r > 0.05 else ("loss" if r < -0.05 else "breakeven")
        manual = reason == "manual"
        tags = pm.initial_tags(t)
        if manual:
            tags = ["closed_manually"] + tags
        elif reason == "agent":
            tags = ["closed_by_agent"] + tags
        elif reason == "protected":
            tags = ["protected_exit"] + tags
        atr = float((t.get("features") or {}).get("atr") or 0.0)
        watch = None if manual else pm.new_watch(t, atr)
        dec = t.get("decision") or {}
        if reason:
            dec["close_reason"] = reason
        self.j.update_trade(t["id"], status="closed", closed_at=exit_time, exit_price=exit_price, pnl=pnl,
                            r_multiple=r, outcome=outcome, entry=t["entry"], exit_estimated=int(estimated),
                            postmortem={"tags": tags, "notes": pm.explain(tags), "final": manual}, watch=watch,
                            decision=dec)
        t["outcome"] = outcome
        kind = "shadow" if t.get("shadow") else "trade"
        if manual:
            # Your decision, not the strategy's: keep following the original plan virtually so the
            # strategy is judged on what its trade would have done.
            sid = self.j.open_trade(
                shadow=1, symbol=t["symbol"], strategy=t["strategy"], experimental=t.get("experimental", 0),
                side=t["side"], opened_at=t["opened_at"], entry=t["entry"], stop=t["stop"],
                take_profit=t["take_profit"], qty=0, risk_amount=0, risk_pct=0, reason=t.get("reason", ""),
                features=t.get("features") or {}, chart=t.get("chart") or [], mfe_r=t.get("mfe_r") or 0,
                mae_r=t.get("mae_r") or 0,
                decision={"action": "shadow", "why": [f"continued virtually after you closed real trade #{t['id']}"],
                          "continues": t["id"]})
            self.j.event("trade", f"#{t['id']} {t['symbol']} {t['side']} closed by you at {r:+.2f}R - "
                                  f"following the original plan virtually as #{sid} to see what it would have done",
                         {"trade_id": t["id"]}, ts=exit_time)
            return
        self.learner.record(t)
        if CANDIDATE in t["strategy"]:
            base = self._strategy(t["strategy"].split(CANDIDATE)[0])
            if base is not None:
                self.learner.judge_challenger(base)
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
            dec = t.get("decision") or {}
            stop = float(dec.get("stop_now", t["stop"]))        # the protected stop, once one is set
            if t["side"] == "buy":
                if lo <= stop:
                    exit_price = stop
                elif hi >= t["take_profit"]:
                    exit_price = t["take_profit"]
            else:
                if hi >= stop:
                    exit_price = stop
                elif lo <= t["take_profit"]:
                    exit_price = t["take_profit"]
            if exit_price is None and (self._time_up(t) or flatten_reason(t["symbol"], ctx.time + self._tf)):
                exit_price = float(bar["close"])
            if exit_price is not None:
                d = 1 if t["side"] == "buy" else -1
                r = (exit_price - t["entry"]) * d / (abs(t["entry"] - t["stop"]) or 1e-12)
                moved = "stop_now" in dec and exit_price == stop and stop != t["stop"]
                self._close_trade(t, exit_price, ctx.time.isoformat(), 0.0, r, reason="protected" if moved else "")
            else:
                self._protect(t, ctx, dec, real=False)

    def _advance_watchers(self, ctx: MarketContext) -> None:
        bar = ctx.bar
        for t in self.j.watching_trades():
            if t["symbol"] != ctx.symbol or (t["closed_at"] or "") > ctx.time.isoformat():
                continue
            if (t["closed_at"] or "") == ctx.time.isoformat():
                # the candle the trade closed in: keep it on the chart, but don't judge "after exit" with it
                rows = t.get("chart") or []
                if not rows or rows[-1][0] != ctx.time.isoformat():
                    self.j.update_trade(t["id"], chart=_trim_chart(rows + [_bar_row(ctx.time, ctx.bar)]))
                continue
            w = pm.update_watch(t, t["watch"], float(bar["high"]), float(bar["low"]))
            chart = (t.get("chart") or []) + [_bar_row(ctx.time, bar)]
            resolved_loss = (t["r_multiple"] or 0) < 0 and (w["hit_target"] or w["beyond_stop"])
            if w["bars"] < self.s.learning.postmortem_watch_bars and not resolved_loss:
                self.j.update_trade(t["id"], watch=w, chart=_trim_chart(chart))
                continue
            tags = pm.final_tags(t, w)
            self.j.update_trade(t["id"], watch=None, chart=_trim_chart(chart),
                                postmortem={"tags": tags, "notes": pm.explain(tags), "final": True, "after_exit": w})
            strat = self._strategy(t["strategy"])
            if strat is not None:
                recent = self.j.trades("strategy = ? AND status = 'closed' AND watch IS NULL AND mode = ?",
                                       (t["strategy"], self.j.mode), order="id DESC", limit=self.s.learning.tune_window)
                self.learner.tune(strat, recent)

    # --- entries -------------------------------------------------------------------------
    def _look_for_trades(self, ctx: MarketContext, quote, only: list[Strategy] | None = None) -> None:
        signals: list[Signal] = []
        pool = self.strategies()
        by_name = {st.name: st for st in pool}
        for strat in (only if only is not None else pool):
            if not strat.applies(ctx):
                continue
            try:
                sig = strat.generate(ctx)
            except Exception:
                log.exception("strategy %s failed", strat.name)
                continue
            if sig is not None and sig.risk_distance > 0:
                sig.set_target(sig.reward_multiple or self.s.risk.reward_multiple)
                if sig.max_bars:
                    sig.features["max_bars"] = sig.max_bars
                signals.append(sig)
        if not signals:
            return
        core_names = {s.name for s in self.core}
        for sig in signals:
            sig.features["confluence"] = sum(1 for o in signals if o.side == sig.side and o.strategy in core_names)
            sig.features["symbol"] = ctx.symbol
            nd = int((ctx.news or {}).get("news_dir", 0) or 0)
            sig.features["news_agree"] = "none" if not nd else ("with" if (nd > 0) == (sig.side == "buy") else "against")
            sig.features.update(htf.features(ctx.htf, sig.side, float(ctx.bar["low"]), float(ctx.bar["high"]),
                                             sig.entry, sig.stop))
            found = cdl.detect(ctx.df, ctx.atr.to_numpy(), lookback=2)
            mine, other = cdl.best(found, sig.side), cdl.best(found, "sell" if sig.side == "buy" else "buy", 2)
            sig.features["candle"] = mine["type"] if mine else "none"
            sig.features["candle_agree"] = "with" if mine else ("against" if other else "none")
            wp = self.plan_for(ctx.symbol)
            sig.features["plan_agree"] = ("none" if not wp else "both" if wp["direction"] == "both"
                                          else "with" if wp["direction"] == sig.side else "against")
            brief = self.desk.brief_for(ctx.symbol) if self.desk else None
            bias = (brief or {}).get("bias", "neutral")
            sig.features["brief_agree"] = "none" if bias == "neutral" else (
                "with" if (bias == "bull") == (sig.side == "buy") else "against")
        # Strongest confluence first, so the best setup gets the real-money slot.
        signals.sort(key=lambda s: (-s.features["confluence"], s.experimental))

        open_real = self.j.open_trades(shadow=False)
        open_shadow = {(t["strategy"], t["symbol"]) for t in self.j.open_trades(shadow=True)}
        real_symbols = {t["symbol"] for t in open_real}

        for sig in signals:
            strat = by_name.get(sig.strategy) or next(st for st in (only or []) if st.name == sig.strategy)
            decision = self.learner.decide(strat, sig.features, ctx.symbol, self.s.live_approved_strategies,
                                           rr=sig.reward_multiple)
            why = list(decision.reasons)
            action = decision.action
            if action == "skip":
                continue
            if action == "trade":
                block = self._entry_blocker(sig, ctx, quote, open_real, real_symbols)
                if not block and self.desk and self.desk.busy_with(sig.symbol):
                    block = "the desk is already debating another setup on this market"
                if block:
                    action = "shadow"
                    why.append(block)
            dec = {"action": action, "p": round(decision.p, 3), "expected_r": round(decision.expected_r, 3),
                   "stat_key": decision.stat_key, "why": why}
            if action == "trade" and self.desk:
                if self.desk.submit(sig, ctx, dec):
                    real_symbols.add(sig.symbol)     # one debate per market at a time
                    continue                          # the verdict comes back on a later cycle
                why.append("desk unavailable (no Claude or daily budget used) - TIIM decided alone")
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
        gap = entry_block(sig.symbol, pd.Timestamp(self.now()))
        if gap:
            return gap
        f = sig.features
        if self.s.plan_filter:
            why = weekly.against(self.plan_for(sig.symbol), sig.side, f)
            if why:
                return why
        if self.s.extension_filter and f.get("extended") == "exhausted":
            return (f"chasing an exhausted move ({htf.extension_text(f, sig.side)}) - a pullback against the "
                    f"trade is likely before the target; waiting for a better entry")
        if self.s.htf_filter and f.get("htf_align") == "against" and not f.get("htf_reversal"):
            frames = ", ".join(f"{k[4:]} {v}" for k, v in f.items() if k in ("htf_M", "htf_W", "htf_D", "htf_4H"))
            return (f"against the higher timeframes ({frames}; trend {f.get('htf_structure')}) - TIIM only trades with "
                    f"the market structure until a 4H/Daily break confirms a reversal")
        if self.news:
            strat = next((st for st in self.core if st.name == sig.strategy.split(CANDIDATE)[0]), None)
            reason = self.news.blackout(sig.symbol, pd.Timestamp(self.now()),
                                        earnings_strategy=bool(strat and "earnings" in strat.needs))
            if reason:
                return reason
            if (sig.features.get("news_agree") == "against"
                    and (sig.features.get("news_impact") or 0) >= self.s.news.guard_min_impact):
                return f"major news points the other way: {sig.features.get('news_title', '')[:80]}"
        equity = self.broker.equity()
        peak = self._peak(equity)
        open_risk = sum(t["risk_pct"] or 0 for t in open_real)
        limit = self.risk.check_limits(equity, peak, self._day_start(equity), open_risk, len(open_real))
        if limit:
            return limit
        group = asset_class(sig.symbol)
        same = sum(1 for t in open_real if t["side"] == sig.side and asset_class(t["symbol"]) == group)
        if same >= self.s.risk.max_same_group:
            return f"correlated: already {same} {sig.side} trades open in {group}"
        now = None if hasattr(self.broker, "now") else pd.Timestamp.now(tz="UTC")
        return self.compliance.check(sig.side, sig.symbol, sig.entry, sig.stop, ctx.close, ctx.atr_now, quote,
                                     self.broker.open_positions(), now)

    def _desk_results(self) -> None:
        """Act on the desk's verdicts. The judge can take, halve or skip; the Risk Officer checks run again
        because time passed while the desk debated."""
        for p, v in self.desk.poll():
            sig, ctx, dec = p.sig, p.ctx, p.dec
            dec["desk"] = v.to_dict()
            why = dec["why"]
            action, scale = "trade", 1.0
            if v.action == "skip":
                if self.desk.veto_allowed():
                    action = "shadow"
                    why.append(f"desk veto: {v.reason}")
                else:
                    why.append(f"desk wanted to skip ({v.reason}) but its vetoes have been costing money - "
                               "veto power suspended, TIIM took it")
            elif v.action == "half":
                scale = 0.5
                why.append(f"desk: half size - {v.reason}")
            elif v.action == "take":
                why.append(f"desk: take - {v.reason}")
            else:
                why.append(v.reason)
            if action == "trade":
                open_real = self.j.open_trades(shadow=False)
                try:
                    quote = self.broker.get_quote(sig.symbol)
                except Exception:  # pragma: no cover - network
                    quote = None
                mid = (quote.bid + quote.ask) / 2 if quote is not None else sig.entry
                drift = abs(mid - sig.entry) / (ctx.atr_now or 1e-12)
                block = None
                if quote is None:
                    block = "no price quote after the debate"
                elif drift > self.s.desk.max_price_drift_atr:
                    block = f"price moved {drift:.2f} ATR while the desk debated - setup no longer valid"
                else:
                    block = self._entry_blocker(sig, ctx, quote, open_real, {t["symbol"] for t in open_real})
                if block:
                    action = "shadow"
                    why.append(block)
                elif self._open_real(sig, ctx, dec, open_real, scale=scale):
                    continue
                else:
                    action = "shadow"
            dec["action"] = action
            open_shadow = {(t["strategy"], t["symbol"]) for t in self.j.open_trades(shadow=True)}
            if (sig.strategy, sig.symbol) not in open_shadow:
                self._open_shadow(sig, ctx, dec)
            self.j.event("desk", f"{sig.symbol} {sig.side} {sig.strategy}: desk said {v.action}"
                                 + (f" - {v.reason}" if v.reason else "") + f" -> {'virtual' if action == 'shadow' else 'real'}",
                         ts=self.now())

    def _merge_lab(self) -> None:
        if not self.s.experiments_allowed:
            return
        rep = self.j.get("lab:report")
        if rep and rep.get("ran_at") != self._lab_seen:
            self._lab_seen = rep.get("ran_at")
            self.learner.merge_lab(rep, self.now())

    def _open_real(self, sig: Signal, ctx: MarketContext, dec: dict, open_real: list[dict], scale: float = 1.0) -> bool:
        equity = self.broker.equity()
        peak = self._peak(equity)
        open_risk = sum(t["risk_pct"] or 0 for t in open_real)
        try:
            spec = self.broker.spec(sig.symbol)
        except Exception as e:  # pragma: no cover - network
            dec["why"].append(f"instrument spec failed: {e}")
            return False
        size = self.risk.size(equity, peak, sig.entry, sig.stop, spec, open_risk, scale=scale)
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
            decision=dec, broker_id=pid, chart=self._chart(ctx), entry_view=self._entry_view(ctx),
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
    def _entry_view(ctx: MarketContext) -> dict | None:
        """Exactly what TIIM saw when it took the trade (zones, swings, indicators), for the dashboard."""
        try:
            return chart_view(ctx, bars=120)
        except Exception:
            log.exception("entry snapshot failed")
            return None

    @staticmethod
    def _chart(ctx: MarketContext) -> list:
        tail = ctx.df.iloc[-CHART_PRE_BARS:]
        return [_bar_row(ts, row) for ts, row in tail.iterrows()]
