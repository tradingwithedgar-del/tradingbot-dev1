# TIIM: self-learning TradeLocker trading agent (PlexyTrade)

TIIM is an autonomous trading agent for TradeLocker, built to run on a **PlexyTrade demo account first**
and only move to the live account after you approve it.

Its rule for every trade: **"How much am I willing to lose to be wrong?"** It picks the stop first
(where the idea is proven wrong), then sizes the position so that being wrong costs exactly the
allowed risk. The target is 3× that risk: **5% risk for a 15% gain**.

**New here? Follow [SETUP.md](SETUP.md) for step-by-step install and run instructions.**

Markets traded by default: **US30, US500, USTECH (US indices), XAUUSD (gold), NVDA, AAPL, TSLA, XTIUSD (WTI oil), BTCUSD, ETHUSD, SOLUSD**.
Exact names differ per broker; `python -m tiim symbols` shows what your account calls them.

```
python -m tiim symbols                  # log in and check symbol names
python -m tiim backtest --synthetic     # smoke test with fake data, no account needed
python -m tiim dashboard --db data/backtest.db
python -m tiim run                      # trade the account in .env (demo by default)
```

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in your TradeLocker demo login
python -m pytest            # 20 tests, about 20s
```

`.env` settings:

| Variable | Meaning |
|---|---|
| `TL_ENVIRONMENT` | `https://demo.tradelocker.com` (demo) or `https://live.tradelocker.com` |
| `TL_EMAIL`, `TL_PASSWORD`, `TL_SERVER` | Your TradeLocker login and the server name shown on the TradeLocker login screen |
| `TL_ACC_NUM` | Which account to use if you have several (optional) |
| `BOT_MODE` | `demo` (default) or `live` |
| `ALLOW_LIVE_TRADING` | Must be `YES` before `BOT_MODE=live` will start |
| `BOT_SYMBOLS` | default `US30,US500,USTECH,XAUUSD,NVDA,AAPL,TSLA,XTIUSD,BTCUSD,ETHUSD,SOLUSD` (must match TradeLocker names exactly) |
| `BOT_TIMEFRAME` | `5m`, `15m`, `30m`, `1H`, `4H` |
| `LIVE_APPROVED_STRATEGIES` | Strategies allowed to trade real money on live |

Safety interlocks: demo mode refuses to start against the live server, live mode refuses to start against
the demo server, and live mode needs `ALLOW_LIVE_TRADING=YES`.

### Verify position sizing on the first demo trades
Position size depends on each instrument's contract size and currency conversion, which the agent reads
from TradeLocker. **Check the first few demo trades**: the loss at the stop shown in TradeLocker should match
the `risk` in the dashboard (about 5% of equity). If a symbol is off, set it in `data/instruments.json`:
```json
{"XAUUSD": {"value_per_point": 100, "qty_step": 0.01, "min_qty": 0.01, "max_qty": 50}}
```
`value_per_point` = account-currency profit for 1.0 lot when price moves by 1.0.
Non-FX symbols (indices, stocks, oil) are assumed to be priced in USD. If one isn't, add
`"quote_currency": "EUR"` (for example) to its entry.

---

## 2. How the agent thinks

Every time a bar closes, for each symbol the agent:

1. **Reads the market**: HH/HL/LH/LL swing structure and breaks of structure, fresh supply/demand zones,
   RSI/MACD divergences (regular and hidden), higher-timeframe bias, regime (trending/ranging ×
   volatility) and session.
2. **Generates signals** from its strategies:
   * `sd_reversal`: retest of a fresh demand/supply zone with a rejection candle; stop beyond the zone.
   * `divergence`: regular/hidden RSI or MACD divergence confirmed by a candle; stop beyond the swing.
   * `structure_trend`: pullback to the last HL in an uptrend or LH in a downtrend; stop beyond that swing.
   * `exp_xxxxxxxx`: **experimental strategies it builds itself** from indicators (demo only, see §4).
3. **Asks its memory** whether this kind of setup has been making money (see §3), and in which conditions.
4. **Checks the rules**: risk limits, then the PlexyTrade compliance guard (§5).
5. **Trades it for real**, or **shadow-trades** it. A shadow trade is virtual: it's tracked to the stop or
   target but risks no money. The agent keeps learning from setups it chose not to risk money on.
6. **Writes a post-mortem** for every closed trade, then keeps watching price for 40 bars after exit to
   see what happened next.

### Risk rules (`tradingbot/config.py → RiskConfig`)
| Rule | Default |
|---|---|
| Risk per trade | 5% of equity |
| Target | 3R (15%) |
| Max open trades / max total open risk | 2 / 10% |
| Daily loss limit | −10%: no new trades for the rest of the day |
| Drawdown scaling | risk shrinks linearly from 5% → 1% as drawdown approaches the halt level |
| Hard halt | −30% from peak: the agent stops and waits for you (`python -m tiim resume`) |
| Never | martingale, grid, averaging down, or increasing size after a loss |

---

## 3. How it learns and adapts

* **Edge tracking with memory decay.** For each strategy, and each strategy × condition (regime,
  session, structure, HTF bias, RSI zone, symbol, confluence…), it keeps a Bayesian estimate of win
  rate and average win/loss in R. Older trades fade (decay 0.97 per trade), so it follows the market
  as it changes.
* **Decisions.** It only risks money when expected R is positive. On demo it *explores*
  (Thompson sampling: it sometimes tries setups it is unsure about). On live it is *conservative*:
  it uses a pessimistic estimate of the win rate.
* **Learned filters ("lessons").** When a strategy has clearly lost money in a specific condition (e.g.
  `divergence` in `ranging_high_vol`, at least 12 trades), that condition is blocked for real money.
  Blocked setups keep running as shadow trades, so if the condition starts working again the filter is
  lifted automatically.
* **Self-tuning from post-mortems.** Post-mortem tags drive parameter changes, within fixed bounds:
  * `stop_too_tight` (stopped, then price hit the original target) → wider stop buffer
  * `immediately_wrong` (never got +0.3R) → stricter entry rules
  * `gave_back_profit` (was +1R, ended −1R) → move stop to break-even at +1.5R
* **Evolution (demo only).** It keeps a population of 8 experimental indicator strategies (RSI, EMA
  stacks, MACD turns, Bollinger touches, stochastic crosses, ADX, zones, structure, momentum candles).
  New ones start as shadow trades. Those reaching +0.3R/trade over 20 trades are promoted to real demo
  trades; losers are retired and replaced by mutations of the best or new random ideas.
  After 30 profitable real demo trades, a strategy becomes a **live candidate** and the journal asks for
  your approval. **No experiment ever reaches the live account on its own.** You add it to
  `LIVE_APPROVED_STRATEGIES`.
* When you switch to live, the agent starts with everything it learned on demo (experiments excluded).

---

## 3a. Strategy library

TIIM pulls its strategies from a library. Each strategy describes what it does, which markets it fits
(indices, stocks, gold, oil, crypto, FX) and what data it needs (news, earnings). TIIM only runs the ones
that fit the situation, and its learned filters switch strategies off in conditions where they lose.

| Strategy | What it does |
|---|---|
| `sd_reversal` | Retest of a fresh supply/demand zone with a rejection candle |
| `divergence` | RSI/MACD divergence at a swing, confirmed by a candle |
| `structure_trend` | Pullbacks in HH/HL uptrends and LH/LL downtrends |
| `news_momentum` / `news_fade` | Price-confirmed reaction to high-impact news, or its over-extension |
| `earnings_runup` | Trades a stock's usual drift into earnings (from its own history) only when the chart agrees; closed before the release |
| `earnings_drift` | After a release, joins the reaction once price breaks the first hour's range, if the stock's earnings moves usually continue |

* **Your strategies:** add a file to `my_strategies/` (see `my_strategies/README.md` and `TEMPLATE.py`).
  TIIM loads it at start-up. `python -m tiim strategies` lists the whole library with results.
* **Switch one off:** `data/strategies.json` → `{"disabled": ["news_fade"]}`.
* **TIIM improves each strategy (demo):** a tweaked copy of every strategy (1–2 settings nudged) runs as
  virtual trades. After 25+ trades, if the tweak earns at least 0.15R per trade more, TIIM adopts it
  ("TIIM improved ..." in the journal). Otherwise the tweak is dropped and a new one is tried.
* **Earnings, price action first:** for each stock TIIM measures past releases from the daily chart:
  the run-up in the 5 sessions before, the reaction gap, and whether the move kept going over the
  next 5 sessions. Dates come from Finnhub or `data/earnings.json` when available, otherwise from the
  chart's largest quarterly gaps. History only sets the bias; the current chart must agree before it trades.

## 3b. News awareness

* **Scheduled events:** the weekly economic calendar (CPI, Fed/FOMC, jobs, GDP, PCE, ISM, oil inventories) and
  earnings dates for your stocks (`FINNHUB_API_KEY`, or `data/earnings.json`). No new trades from 15 min before
  to 5 min after a high-impact release, no new stock trades in the 20 h before earnings, and open stock trades are
  closed 45 min before earnings.
* **Breaking headlines:** public news feeds checked every 90 s. With `ANTHROPIC_API_KEY` set, Claude rates
  each headline (impact 0–3) and says which of your symbols it should push up or down. Without a key a
  keyword reader is used.
* **Trading the news (demo only until you approve them):**
  * `news_momentum`: after high-impact news, rides a strong impulse candle if it agrees with the headline.
  * `news_fade`: trades the snap-back when a news spike over-extends and gets rejected.
  While news is hot, these also check 5-minute bars, so the agent reacts within minutes.
* **News protects, price action decides.** When a major headline (impact 3) breaks against an open trade,
  TIIM takes the profit if the trade is up 1.5R or more, or moves the stop to break-even if it's up 0.3R or more.
  If the trade is at a loss, the stop stays where it is. TIIM also won't open a new trade straight into
  fresh major news pointing the other way.
* **Learning:** every trade records the news situation (`news_state`, `news_kind`, `news_agree`), so the
  agent learns which kinds of news are worth trading on which symbols, and blocks the ones that lose.
* Headlines reach public feeds minutes after the fact. The edge is in reading the reaction and its
  follow-through, not in being first. `python -m tiim news` shows what the agent currently sees.

## 4. Dashboard

```bash
python -m tiim dashboard                 # demo/live journal, http://127.0.0.1:8000
python -m tiim dashboard --db data/backtest.db
```
* **What the agent sees:** per-symbol chart with its supply/demand zones, HH/HL/LH/LL labels, EMA 50/200,
  RSI, MACD, divergences, open trades (entry/stop/target), and its read of trend, bias, regime and news
* Upcoming high-impact events and market-moving headlines
* **You can close any trade yourself in TradeLocker.** The agent records it as "closed by you", doesn't blame
  the strategy for your decision, and keeps following the original plan virtually to show what would have happened
* Survival status (Proving itself / Surviving / Thriving / At risk / Termination zone)
* Equity, drawdown, return, win rate, expectancy (R), profit factor, open risk
* R per trade and cumulative R per strategy
* Strategy table: stage, real vs shadow results, learned win rate, blocked conditions, tuned parameters
* **Where it works and where it doesn't**: heat map of average R by strategy × regime, session,
  structure, bias, volatility, symbol, RSI zone or confluence
* **Why trades lost**: post-mortem tag counts
* Agent journal: lessons, tuning, evolution, approvals needed, halts
* Click any trade to replay it: candles with entry/stop/target and exit, the market snapshot at entry,
  why the agent took it, and its post-mortem

The dashboard is read-only and binds to `127.0.0.1` by default.

---

## 5. PlexyTrade terms of use: what is and isn't allowed

I couldn't open plexytrade.com from the build environment, so this is based on published excerpts of the
PlexyTrade **Account Opening Agreement** (the "Market Abuse and Manipulation" section and related
clauses). **Please read the current agreement yourself before going live.**

* **Allowed:** EAs/automated trading, scalping, hedging, news trading.
* **Not allowed:** arbitrage of any kind; manipulating prices, execution or the platform; trading on
  errors, omissions or misquotes; Negative Balance Protection abuse; "any other strategy deemed abusive",
  meaning patterns designed to extract artificial gain at the broker's expense rather than normal market
  participation. The broker may cancel trades or profits from these.

So "exploiting vulnerabilities" here means **market** inefficiencies the agent finds in price itself
(structure, zones, divergences, session and volatility behaviour). It never means broker or platform
weaknesses. `tradingbot/compliance.py` enforces this. The agent:
* uses only the broker's own price feed, with no external feed comparison (no latency arbitrage)
* refuses to trade abnormal spreads, stale quotes, or quotes far from the last close (misquotes)
* refuses stops under 0.5 ATR (latency-scalping pattern)
* never opens opposite positions on the same symbol, and runs on one account
* limits itself to 4 orders per minute
* never uses martingale or grid sizing, and never increases size after losses (protects against NBP-abuse claims)

---

## 6. Commands

| Command | What it does |
|---|---|
| `python -m tiim strategies` | List the strategy library, who wrote each one, and how each is doing |
| `python -m tiim news` | Check the news feeds, upcoming events and how headlines are read |
| `python -m tiim symbols [--search NAS]` | Log in, list instrument names, check `BOT_SYMBOLS` |
| `python -m tiim run` | Run the agent (polls every 20s, acts on each new closed bar) |
| `python -m tiim status` | Mode, halt state, open trades, latest journal entries |
| `python -m tiim stop` | Creates `data/STOP`: no new real trades (open trades keep their SL/TP) |
| `python -m tiim resume` | Clears STOP and a drawdown halt; drawdown is measured from now |
| `python -m tiim backtest --csv EURUSD=eurusd_15m.csv` | Backtest on your own data (`time,open,high,low,close`) |
| `python -m tiim backtest --tradelocker --days 60` | Backtest on history downloaded from TradeLocker |
| `python -m tiim backtest --synthetic` | Plumbing test on random data (it has no edge to find) |

Backtests fill at the next bar's open, include spread, and assume the stop was hit first whenever a bar
touches both stop and target, so results lean pessimistic.

---

## 7. Honest expectations

* **5% per trade is very aggressive.** Five losses in a row (normal, even for a good 3R strategy) costs
  roughly 20% of the account, even with drawdown scaling. The scaling and the −30% halt are there to keep the account alive. Consider
  starting live at 1–2% and raising it once the demo record justifies it (`RiskConfig.risk_per_trade`).
* **Stocks and indices gap.** NVDA, AAPL and TSLA only trade during US market hours, and indices and oil
  close daily and at weekends. A price that opens beyond the stop (overnight news, earnings) fills at the
  open, so a single loss can be bigger than 5%. The backtester models this. An
  earnings-day filter or closing stock positions before the bell are good next additions.
* **At 3R, win rates of 30–45% are good.** Breakeven is 25%. A system targeting 3R will not show the
  70%+ win rates of systems that take small profits. Judge it by **expectancy (R per trade)** and
  **profit factor**, not win rate alone.
* **Learning needs data.** Filters need at least 12 trades in a condition and tuning needs 20 per strategy.
  Expect the first weeks on demo to be the agent finding its feet. Shadow trades speed this up.
* Run it on demo until it has **100+ real demo trades** with positive expectancy before switching
  to live.

## Layout
```
tradingbot/
  analysis/      indicators, structure (HH/HL/LH/LL), zones, divergence, regime, context
  strategies/    core strategies + experimental genome strategies
  broker/        TradeLocker connector, simulated broker
  agent.py       the decision loop
  learning.py    edge tracking, filters, tuning, evolution
  postmortem.py  trade post-mortems
  risk.py        sizing and limits
  compliance.py  PlexyTrade guardrails
  journal.py     SQLite journal
  dashboard/     FastAPI + Plotly dashboard
tests/
```
