# TIIM's trading desk

```
market data ──► Research Analyst (Claude, every 2 h + major news) ──► market brief ─┐
            ──► Technical Analyst (code, every 15m candle) ──► setup ───────────────┤
            ──► Strategy Lab (code + Claude ideas, nightly) ──► new strategies       │
                                                                                     ▼
                                    setup dossier ─► Bull advocate ◄─1 rebuttal─► Bear advocate
                                                                │                       │
                                                                └──► TIIM, the judge ◄──┘
                                                                     take / half / skip
                                                                          │
                                         Risk Officer (code, last word: size, limits, gaps, broker rules)
                                                                          │
                                                                Executor ─► TradeLocker
Scorekeeper (code): grades every agent on real results, including the setups the desk skipped.
```

## Who does what

| Agent | Runs on | When | Can do |
|---|---|---|---|
| Research Analyst | Claude (`DESK_MODEL_RESEARCH`, default sonnet) | every `DESK_RESEARCH_MINUTES` (120), sooner after major news | writes a bull/bear/neutral bias per market with drivers and risks |
| Technical Analyst | code | every 15m candle | finds setups with the strategy library (zones, structure, divergence, earnings, news, experiments) |
| Bull / Bear advocates | Claude (`DESK_MODEL_ADVOCATE`, sonnet) | only when a setup would become a real trade | argue each side, then one rebuttal each (`DESK_REBUTTAL=1`) |
| TIIM, the judge | Claude (`DESK_MODEL_JUDGE`, opus) | after the debate | **take**, **half** size or **skip**, with a reason. Veto power from day one |
| Risk Officer | code | after the judge | sizes the trade or blocks it. The judge can't raise risk, move stops or bypass limits |
| Scorekeeper | code | after every trade | report cards. If vetoed setups average better than +0.15R over 20+ vetoes, the judge loses veto power until that recovers |
| Strategy Lab | code + Claude ideas | nightly 04:20 UTC (`tiim lab` runs it now) | backtests ~150 ideas on ~6 months of real 15m history; adds the ones that pass the gate (demo only) |

## The Strategy Lab gate

A strategy is added only if **all** hold:

* win rate **≥ 70%** (`LAB_MIN_WIN_RATE`) on the history used to find it, and again on the most recent 30% it never saw
* at least **100 trades** (`LAB_MIN_TRADES`)
* it doesn't lose money overall after trading costs (`LAB_REQUIRE_PROFIT=1`). A 70% win rate with tiny wins and big losses
  still loses; set `LAB_REQUIRE_PROFIT=0` to judge on win rate alone.

Strategies the Lab added are re-tested every night and withdrawn if they stop passing. They trade on demo only;
live still needs your approval in `LIVE_APPROVED_STRATEGIES`.

## Claude usage (Pro plan)

The desk never uses more than `DESK_DAILY_CALLS` (60) Claude calls a day: a research brief is 1 call, a debated setup is
5 (3 when the budget runs low and the rebuttal is skipped). When the allowance is used up, or Claude is unreachable,
TIIM decides alone exactly as before. The headline reader is separate (`NEWS_CLI_INTERVAL_SECONDS`).
Only one `claude` process runs at a time on the server.

`tiim desk` shows the latest brief, recent debates, report cards and the Lab's last run. The dashboard has the same
in its "Trading desk" card, and each trade's "Why TIIM took…" panel shows its debate.
