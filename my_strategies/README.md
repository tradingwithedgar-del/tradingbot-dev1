# Your strategies

Put your own strategies here. Each `.py` file in this folder is loaded into TIIM's strategy
library automatically the next time TIIM starts. Files starting with `_` or `TEMPLATE` are ignored.

The easiest way to add one: **describe your strategy to Claude in plain words** (what you look for
on the chart, where you'd enter, where your stop goes, which markets) and ask for it as a TIIM
strategy file. `TEMPLATE.py` shows the shape.

What happens after you add one:
1. On demo, TIIM trades it with the same 5% risk / 3R rules and the same safety checks as every
   other strategy.
2. TIIM keeps score of it per market condition (trend/range, volatility, session, news, symbol) and
   stops using it where it loses.
3. TIIM tests tweaked versions of its settings as virtual trades and adopts a tweak when it clearly
   earns more (you'll see "TIIM improved ..." in the dashboard journal).
4. It only trades the live account once you add its `name` to `LIVE_APPROVED_STRATEGIES` in `.env`.

Check that TIIM found it: `python -m tiim strategies`. A file with a mistake is reported there and
skipped. It never stops TIIM.
