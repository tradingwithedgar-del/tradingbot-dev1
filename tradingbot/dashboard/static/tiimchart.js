/* TIIM chart: draws what TIIM sees with TradingView Lightweight Charts.
 * Candles + EMA 50/200, supply/demand zones (shaded boxes), HH/HL/LH/LL swing labels,
 * divergences, trades (entry arrow, entry/stop/target, exit), RSI and MACD panes.
 *
 *   const c = TiimChart.render(element, view, {trades, colors, onTradeClick});
 *   c.destroy();
 */
(function () {
  const LW = window.LightweightCharts;
  const ts = (iso) => Math.floor(Date.parse(iso) / 1000);

  // Shaded rectangles (zones) drawn behind the candles.
  class Boxes {
    constructor(boxes) { this.boxes = boxes; this._chart = null; this._series = null; }
    attached({ chart, series }) { this._chart = chart; this._series = series; }
    detached() { this._chart = null; this._series = null; }
    updateAllViews() {}
    paneViews() {
      const self = this;
      return [{
        zOrder: () => 'bottom',
        renderer: () => ({
          draw: () => {},
          drawBackground(target) {
            if (!self._chart || !self._series) return;
            const tsApi = self._chart.timeScale();
            target.useMediaCoordinateSpace(({ context: ctx, mediaSize }) => {
              for (const b of self.boxes) {
                let x0 = tsApi.timeToCoordinate(b.t0);
                let x1 = b.t1 == null ? mediaSize.width : tsApi.timeToCoordinate(b.t1);
                if (x0 == null) x0 = 0;
                if (x1 == null) x1 = mediaSize.width;
                const y0 = self._series.priceToCoordinate(b.top);
                const y1 = self._series.priceToCoordinate(b.bottom);
                if (y0 == null || y1 == null) continue;
                const top = Math.min(y0, y1), h = Math.max(2, Math.abs(y1 - y0));
                ctx.fillStyle = b.fill;
                ctx.fillRect(x0, top, Math.max(1, x1 - x0), h);
                ctx.strokeStyle = b.stroke;
                ctx.lineWidth = 1;
                ctx.strokeRect(x0 + 0.5, top + 0.5, Math.max(1, x1 - x0) - 1, h - 1);
                if (b.label) {
                  ctx.font = '11px system-ui, -apple-system, Segoe UI, sans-serif';
                  ctx.fillStyle = b.text;
                  ctx.textAlign = 'right';
                  ctx.fillText(b.label, Math.min(x1, mediaSize.width) - 6, top + 13);
                }
              }
            });
          },
        }),
      }];
    }
  }

  function alpha(hex, a) {
    const h = hex.replace('#', '');
    const n = parseInt(h.length === 3 ? h.split('').map(c => c + c).join('') : h, 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // Snap an arbitrary time to the candle that contains it (trades can come from 5m bars).
  function snapper(times) {
    return (t) => {
      let lo = 0, hi = times.length - 1, ans = times[0];
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (times[mid] <= t) { ans = times[mid]; lo = mid + 1; } else hi = mid - 1;
      }
      return ans;
    };
  }

  function render(el, view, opts) {
    opts = opts || {};
    const C = opts.colors;
    el.innerHTML = '';
    const chart = LW.createChart(el, {
      autoSize: true,
      layout: { background: { type: 'solid', color: C.surface }, textColor: C.ink2, fontSize: 11,
                fontFamily: 'system-ui, -apple-system, Segoe UI, sans-serif', panes: { separatorColor: C.grid } },
      grid: { vertLines: { color: C.grid }, horzLines: { color: C.grid } },
      rightPriceScale: { borderColor: C.axis },
      timeScale: { borderColor: C.axis, timeVisible: true, secondsVisible: false, rightOffset: 6 },
      crosshair: { mode: 0 },
      localization: { locale: 'en-US' },
    });

    const times = view.t.map(ts);
    const snap = snapper(times);
    const tMin = times[0], tMax = times[times.length - 1];
    const candles = chart.addSeries(LW.CandlestickSeries, {
      upColor: C.up, downColor: C.down, wickUpColor: C.up, wickDownColor: C.down, borderVisible: false,
    });
    candles.setData(times.map((t, i) => ({ time: t, open: view.open[i], high: view.high[i], low: view.low[i], close: view.close[i] }))
      .filter(b => b.open != null));

    const line = (pane, values, color, width, title, style) => {
      const s = chart.addSeries(LW.LineSeries, { color, lineWidth: width || 1, title: title || '', lineStyle: style || 0,
        priceLineVisible: false, lastValueVisible: !!title, crosshairMarkerVisible: false }, pane);
      s.setData(times.map((t, i) => (values[i] == null ? { time: t } : { time: t, value: values[i] })));
      return s;
    };
    if (view.ema50) line(0, view.ema50, C.ema50, 2, 'EMA 50');
    if (view.ema200) line(0, view.ema200, C.ema200, 2, 'EMA 200');

    // zones
    const boxes = [];
    if (opts.zones !== false) {
      for (const z of view.zones || []) {
        const c = z.kind === 'demand' ? C.demand : C.supply;
        boxes.push({ t0: snap(ts(z.from)), t1: null, top: z.top, bottom: z.bottom,
          fill: alpha(c, z.fresh ? 0.22 : 0.10), stroke: alpha(c, 0.6), text: C.ink2,
          label: `${z.kind}${z.fresh ? ' (fresh)' : ` (${z.touches}x tested)`}` });
      }
    }
    if (opts.highlightZone) boxes.push(opts.highlightZone);
    candles.attachPrimitive(new Boxes(boxes));

    // markers: swings + trades
    const markers = [];
    if (opts.swings !== false) {
      for (const s of view.swings || []) {
        if (!s.label) continue;
        const t = ts(s.t);
        if (t < tMin || t > tMax) continue;
        markers.push({ time: snap(t), position: s.kind === 'H' ? 'aboveBar' : 'belowBar', shape: 'circle', size: 0.1,
          color: s.label === 'HH' || s.label === 'HL' ? C.up : C.down, text: s.label });
      }
    }
    const tradeAt = {};
    for (const tr of opts.trades || []) {
      const t0 = ts(tr.opened_at);
      if (t0 >= tMin && t0 <= tMax + 3600) {
        const buy = tr.side === 'buy';
        const at = snap(t0);
        tradeAt[at] = tr.id;
        markers.push({ time: at, position: buy ? 'belowBar' : 'aboveBar', shape: buy ? 'arrowUp' : 'arrowDown',
          color: buy ? C.up : C.down, size: 1.4, text: `#${tr.id} ${buy ? 'BUY' : 'SELL'}${tr.shadow ? ' (virtual)' : ''}` });
      }
      if (tr.closed_at && tr.exit_price != null) {
        const t1 = ts(tr.closed_at);
        if (t1 >= tMin && t1 <= tMax + 3600) {
          const win = (tr.r_multiple || 0) > 0;
          markers.push({ time: snap(t1), position: 'inBar', shape: 'square', size: 0.8, color: win ? C.good : C.bad,
            text: `exit ${tr.r_multiple >= 0 ? '+' : ''}${(tr.r_multiple || 0).toFixed(2)}R` });
        }
      }
      // entry / stop / target from entry to exit (or to the right edge while open)
      const from = snap(Math.max(t0, tMin));
      const to = tr.closed_at ? snap(Math.min(ts(tr.closed_at), tMax)) : tMax;
      if (from <= tMax) {
        for (const [price, color, style, title] of [[tr.entry, C.entry, 0, 'entry'], [tr.stop, C.bad, 2, 'stop'], [tr.take_profit, C.good, 2, 'target']]) {
          if (price == null) continue;
          const seg = chart.addSeries(LW.LineSeries, { color, lineWidth: tr.status === 'open' ? 2 : 1, lineStyle: style,
            priceLineVisible: false, lastValueVisible: tr.status === 'open', title: tr.status === 'open' ? `${title} #${tr.id}` : '',
            crosshairMarkerVisible: false });
          seg.setData(from === to ? [{ time: from, value: price }] : [{ time: from, value: price }, { time: to, value: price }]);
        }
      }
    }
    markers.sort((a, b) => a.time - b.time);
    LW.createSeriesMarkers(candles, markers);

    // divergences
    for (const d of view.divergences || []) {
      const a = ts(d.t1), b = ts(d.t2);
      if (a < tMin || b > tMax || a === b) continue;
      const s = chart.addSeries(LW.LineSeries, { color: d.kind.endsWith('bull') ? C.demand : C.supply, lineWidth: 2, lineStyle: 1,
        priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
      s.setData([{ time: snap(a), value: d.p1 }, { time: snap(b), value: d.p2 }]);
    }

    // RSI pane
    if (view.rsi) {
      const r = line(1, view.rsi, C.rsi, 2, 'RSI 14');
      r.createPriceLine({ price: 70, color: C.axis, lineStyle: 2, lineWidth: 1, axisLabelVisible: false });
      r.createPriceLine({ price: 30, color: C.axis, lineStyle: 2, lineWidth: 1, axisLabelVisible: false });
    }
    // MACD pane
    if (view.macd_hist) {
      const h = chart.addSeries(LW.HistogramSeries, { priceLineVisible: false, lastValueVisible: false, title: 'MACD hist' }, 2);
      h.setData(times.map((t, i) => (view.macd_hist[i] == null ? { time: t }
        : { time: t, value: view.macd_hist[i], color: view.macd_hist[i] >= 0 ? alpha(C.up, 0.8) : alpha(C.down, 0.8) })));
    }
    const panes = chart.panes();
    if (panes[0]) panes[0].setStretchFactor(3.2);
    if (panes[1]) panes[1].setStretchFactor(0.9);
    if (panes[2]) panes[2].setStretchFactor(0.9);

    if (opts.focus) {
      const i = times.indexOf(snap(ts(opts.focus)));
      const j = opts.focusEnd ? times.indexOf(snap(ts(opts.focusEnd))) : -1;
      const end = j > i ? j + 12 : i + 50;
      if (i >= 0) chart.timeScale().setVisibleLogicalRange({ from: Math.max(0, i - 60), to: Math.min(times.length + 8, end) });
    } else {
      chart.timeScale().fitContent();
    }
    if (opts.onTradeClick) {
      chart.subscribeClick((p) => {
        if (p && p.time != null && tradeAt[p.time] != null) opts.onTradeClick(tradeAt[p.time]);
      });
    }
    return { chart, destroy: () => chart.remove() };
  }

  window.TiimChart = { render, ts };
})();
