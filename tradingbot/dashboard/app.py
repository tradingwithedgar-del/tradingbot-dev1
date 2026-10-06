"""Read-only dashboard API + single-page UI. It only reads the journal; it can't place trades."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import base64
import hmac
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response

from ..config import Settings
from ..journal import Journal
from ..learning import CANDIDATE, EdgeStat
from ..strategies.registry import disabled, discover

STATIC = Path(__file__).parent / "static"
DIMENSIONS = ["regime", "session", "structure", "bias", "volatility", "symbol", "rsi_zone", "confluence",
              "news_state", "news_kind", "news_agree"]


def stats_for(trades: list[dict]) -> dict:
    rs = [t["r_multiple"] or 0.0 for t in trades]
    wins = [r for r in rs if r > 0]
    losses = [r for r in rs if r <= 0]
    gross_loss = -sum(losses)
    return {
        "trades": len(rs),
        "wins": len(wins),
        "win_rate": len(wins) / len(rs) if rs else None,
        "expectancy_r": sum(rs) / len(rs) if rs else None,
        "total_r": sum(rs),
        "avg_win_r": sum(wins) / len(wins) if wins else None,
        "avg_loss_r": sum(losses) / len(losses) if losses else None,
        "profit_factor": (sum(wins) / gross_loss) if gross_loss > 0 else None,
        "pnl": sum(t["pnl"] or 0.0 for t in trades),
    }


def survival(st: dict, dd: float, halted: bool) -> dict:
    exp = st["expectancy_r"]
    n = st["trades"]
    if halted or dd >= 0.25:
        return {"level": "critical", "label": "Termination zone", "detail": "Drawdown limit hit or about to be hit"}
    if n < 20:
        return {"level": "warning", "label": "Proving itself", "detail": f"{n}/20 real trades before a verdict"}
    if exp is not None and exp > 0.3 and dd < 0.10:
        return {"level": "good", "label": "Thriving", "detail": f"{exp:+.2f}R per trade"}
    if exp is not None and exp > 0:
        return {"level": "good", "label": "Surviving", "detail": f"{exp:+.2f}R per trade"}
    return {"level": "serious", "label": "At risk", "detail": "Negative expectancy over recent trades"}


def create_app(db_path: str | Path | None = None) -> FastAPI:
    settings = Settings()
    path = Path(db_path) if db_path else settings.db_path
    if not path.exists():
        raise SystemExit(f"No journal at {path}. Run the agent or a backtest first.")
    j = Journal(path)
    app = FastAPI(title="TIIM dashboard")

    password = os.getenv("DASHBOARD_PASSWORD", "")
    if password:
        # Simple login prompt in the browser (user name can be anything) - set DASHBOARD_PASSWORD in .env
        @app.middleware("http")
        async def require_password(request: Request, call_next):
            header = request.headers.get("authorization", "")
            ok = False
            if header.startswith("Basic "):
                try:
                    _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                    ok = hmac.compare_digest(given, password)
                except Exception:
                    ok = False
            if not ok:
                return Response("Login required", status_code=401, headers={"WWW-Authenticate": 'Basic realm="TIIM"'})
            return await call_next(request)

    def closed(mode: str, shadow: int | None = 0) -> list[dict]:
        if shadow is None:
            return j.trades("status='closed' AND mode=?", (mode,), order="closed_at")
        return j.trades("status='closed' AND mode=? AND shadow=?", (mode, shadow), order="closed_at")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/plotly.min.js")
    def plotly_js():
        # Served from the installed `plotly` Python package so the dashboard works offline.
        import plotly

        return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", media_type="text/javascript")

    @app.get("/api/modes")
    def modes():
        rows = j.conn.execute("SELECT DISTINCT mode FROM trades UNION SELECT DISTINCT mode FROM equity").fetchall()
        found = [r[0] for r in rows if r[0]]
        order = ["live", "demo", "backtest"]
        return sorted(found, key=lambda m: order.index(m) if m in order else 9)

    @app.get("/api/summary")
    def summary(mode: str):
        curve = j.equity_curve(mode)
        last = curve[-1] if curve else {"equity": 0, "balance": 0, "peak": 0, "drawdown": 0}
        first = curve[0]["equity"] if curve else 0
        max_dd = max((c["drawdown"] for c in curve), default=0.0)
        real = closed(mode, 0)
        st = stats_for(real)
        recent = stats_for(real[-30:])
        halted = j.get(f"halted:{mode}")
        open_real = j.trades("status='open' AND mode=? AND shadow=0", (mode,))
        return {
            "mode": mode, "equity": last["equity"], "balance": last["balance"], "peak": last["peak"],
            "drawdown": last["drawdown"], "max_drawdown": max_dd,
            "return_pct": (last["equity"] - first) / first if first else 0.0,
            "all": st, "recent": recent, "shadow": stats_for(closed(mode, 1)),
            "open_trades": len(open_real), "open_risk_pct": sum(t["risk_pct"] or 0 for t in open_real),
            "halted": halted, "survival": survival(recent, last["drawdown"], bool(halted)),
            "risk_per_trade": settings.risk.risk_per_trade, "reward_multiple": settings.risk.reward_multiple,
        }

    @app.get("/api/equity")
    def equity(mode: str):
        curve = j.equity_curve(mode)
        # thin very long curves to keep the page light
        step = max(1, len(curve) // 2000)
        return [{"ts": c["ts"], "equity": c["equity"], "balance": c["balance"], "drawdown": c["drawdown"]} for c in curve[::step]]

    @app.get("/api/trades")
    def trades(mode: str, kind: str = "real", limit: int = 300):
        where = "mode=?"
        if kind == "real":
            where += " AND shadow=0"
        elif kind == "shadow":
            where += " AND shadow=1"
        rows = j.trades(where, (mode,), order="id DESC", limit=limit)
        keep = ["id", "shadow", "symbol", "strategy", "experimental", "side", "status", "opened_at", "closed_at", "entry",
                "stop", "take_profit", "exit_price", "pnl", "r_multiple", "outcome", "mfe_r", "mae_r", "risk_pct", "reason"]
        out = []
        for r in rows:
            d = {k: r.get(k) for k in keep}
            d["regime"] = (r.get("features") or {}).get("regime")
            d["tags"] = (r.get("postmortem") or {}).get("tags", [])
            out.append(d)
        return out

    @app.get("/api/trade/{trade_id}")
    def trade(trade_id: int):
        t = j.trade(trade_id)
        if not t:
            raise HTTPException(404)
        return t

    @app.get("/api/strategies")
    def strategies(mode: str):
        state = j.get(f"learner:{mode}") or {}
        pop = state.get("population", {})
        filters = state.get("filters", {})
        params = state.get("params", {})
        be = state.get("breakeven", {})
        stats = {k: EdgeStat(**v) for k, v in state.get("stats", {}).items()}
        all_closed = closed(mode, None)
        by: dict[str, dict[str, list]] = defaultdict(lambda: {"real": [], "shadow": []})
        for t in all_closed:
            by[t["strategy"]]["shadow" if t["shadow"] else "real"].append(t)
        classes, _ = discover()
        lib = {c.name: c() for c in classes}
        off = disabled(settings.db_path.parent)
        challengers = state.get("challengers", {})
        names = sorted(set(by) | set(pop) | set(lib), key=lambda n: (n not in lib, n.split(CANDIDATE)[0], n))
        out = []
        for n in names:
            info = pop.get(n)
            base = n.split(CANDIDATE)[0]
            meta = lib.get(base)
            cum, total = [], 0.0
            for t in by[n]["real"]:
                total += t["r_multiple"] or 0
                cum.append({"ts": t["closed_at"], "r": round(total, 3)})
            st = stats.get(n)
            if CANDIDATE in n:
                kind, stage = "variant", "testing a tweak"
                desc = "Tweaked copy of " + base + ": " + ", ".join(
                    f"{k}={v:.2f}" for k, v in (challengers.get(base, {}).get("changed") or {}).items())
            elif info or n.startswith("exp_"):
                kind, stage, desc = "experimental", (info or {}).get("status", "retired"), (info or {}).get("description")
            else:
                kind = "library"
                stage = ("disabled" if n in off else "approved for live" if n in settings.live_approved_strategies
                         else "demo only")
                desc = meta.description if meta else None
            out.append({
                "name": n,
                "title": (meta.title if meta and CANDIDATE not in n else None),
                "author": meta.author if meta else "TIIM",
                "markets": list(meta.markets) if meta else [],
                "kind": kind,
                "stage": stage,
                "description": desc,
                "real": stats_for(by[n]["real"]), "shadow": stats_for(by[n]["shadow"]),
                "learned_win_rate": st.p_mean(settings.learning) if st else None,
                "filters": filters.get(n, []), "params": params.get(n), "breakeven_at_r": be.get(n),
                "cum_r": cum,
            })
        return out

    @app.get("/api/breakdown")
    def breakdown(mode: str, by: str = "regime", kind: str = "all"):
        if by not in DIMENSIONS:
            raise HTTPException(400, f"by must be one of {DIMENSIONS}")
        rows = closed(mode, None if kind == "all" else (1 if kind == "shadow" else 0))
        cells: dict[tuple, list] = defaultdict(list)
        for t in rows:
            v = t["symbol"] if by == "symbol" else (t.get("features") or {}).get(by)
            cells[(t["strategy"], str(v))].append(t)
        return [{"strategy": s, "value": v, **stats_for(ts)} for (s, v), ts in sorted(cells.items())]

    @app.get("/api/postmortems")
    def postmortems(mode: str, kind: str = "all"):
        rows = closed(mode, None if kind == "all" else (1 if kind == "shadow" else 0))
        loss_tags, win_tags = Counter(), Counter()
        for t in rows:
            tags = (t.get("postmortem") or {}).get("tags", [])
            (loss_tags if (t["r_multiple"] or 0) < 0 else win_tags).update(tags)
        return {"losses": loss_tags.most_common(), "wins": win_tags.most_common()}

    @app.get("/api/symbols")
    def symbols(mode: str):
        rows = j.conn.execute("SELECT key FROM kv WHERE key LIKE ?", (f"view:{mode}:%",)).fetchall()
        found = [r[0].split(":", 2)[2] for r in rows]
        order = {sym: i for i, sym in enumerate(settings.symbols)}
        return sorted(found, key=lambda x: order.get(x, 999))

    @app.get("/api/view")
    def view(mode: str, symbol: str):
        v = j.get(f"view:{mode}:{symbol}")
        if not v:
            raise HTTPException(404, "no chart yet for this symbol - it appears after the next bar closes")
        start = v["t"][0] if v["t"] else ""
        keep = ["id", "shadow", "side", "status", "strategy", "opened_at", "closed_at", "entry", "stop", "take_profit",
                "exit_price", "r_multiple", "reason"]
        trades = j.trades("mode=? AND symbol=? AND (status='open' OR closed_at >= ?)", (mode, symbol, start),
                          order="id DESC", limit=40)
        v["trades"] = [{k: t.get(k) for k in keep} | {"stop_now": (t.get("decision") or {}).get("breakeven_moved")}
                       for t in trades if not t["shadow"] or t["status"] == "open"]
        return v

    @app.get("/api/news")
    def news(limit: int = 60):
        now = pd.Timestamp.now(tz="UTC")
        upcoming = [n for n in j.news(300, kind="event", since=(now - pd.Timedelta(hours=2)).isoformat())]
        upcoming.sort(key=lambda n: n["ts"])
        heads = [n for n in j.news(limit * 3, kind="headline") if (n["impact"] or 0) >= 2][:limit]
        return {"upcoming": upcoming[:40], "headlines": heads}

    @app.get("/api/events")
    def events(mode: str, limit: int = 200):
        return j.events(limit, mode)

    return app
