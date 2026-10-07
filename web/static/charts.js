/* Chart helpers. Two inks only: gold for Quorum's series, grey for the benchmark or baseline. */
(function () {
  const read = (id) => {
    const el = document.getElementById(id);
    return el ? JSON.parse(el.textContent) : null;
  };
  if (!window.Chart) {
    // Chart library blocked or offline: keep the page working, just without charts
    const skip = () => {};
    window.Q = { read, equity: skip, drawdown: skip, mini: skip, calibration: skip };
    return;
  }
  const css = getComputedStyle(document.documentElement);
  const v = (n) => css.getPropertyValue(n).trim();
  const GOLD = v("--gold"), GREY = "#8f9198", LINE = v("--line"), MUTED = v("--muted"), INK = v("--ink");
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const short = (d) => MONTHS[+d.slice(5, 7) - 1] + " " + +d.slice(8, 10);
  const long = (d) => short(d) + ", " + d.slice(0, 4);

  Chart.defaults.font.family = v("--sans");
  Chart.defaults.font.size = 12;
  Chart.defaults.color = MUTED;
  Chart.defaults.animation = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? false : { duration: 400 };

  const tooltip = (fmt) => ({
    backgroundColor: "#ffffff",
    borderColor: v("--line-strong"),
    borderWidth: 1,
    titleColor: INK,
    bodyColor: v("--ink-2"),
    titleFont: { weight: "600" },
    bodyFont: { family: v("--mono") },
    padding: 10,
    boxWidth: 10,
    boxHeight: 2,
    callbacks: { title: (items) => long(items[0].label), label: (c) => ` ${c.dataset.label}: ${fmt(c.parsed.y)}` },
  });

  const axes = (yfmt, yopts = {}) => ({
    x: {
      grid: { display: false },
      border: { color: LINE },
      ticks: { maxTicksLimit: 6, maxRotation: 0, autoSkipPadding: 24, callback: function (val) { return short(this.getLabelForValue(val)); } },
    },
    y: {
      position: "right",
      grid: { color: LINE, drawTicks: false },
      border: { display: false },
      ticks: { padding: 8, maxTicksLimit: 5, callback: yfmt, font: { family: v("--mono"), size: 11 } },
      ...yopts,
    },
  });

  const base = (fmt) => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: { legend: { display: false }, tooltip: tooltip(fmt) },
    elements: { point: { radius: 0, hoverRadius: 4, hitRadius: 8 }, line: { borderWidth: 2, tension: 0 } },
  });

  const sign = (x, d = 1) => (x > 0 ? "+" : x < 0 ? "−" : "") + Math.abs(x).toFixed(d);

  window.Q = {
    read,
    equity(id, s) {
      const opts = base((y) => y.toFixed(1));
      opts.scales = axes((y) => y.toFixed(0));
      new Chart(document.getElementById(id), {
        type: "line",
        data: {
          labels: s.dates,
          datasets: [
            { label: "Quorum", data: s.port, borderColor: GOLD, backgroundColor: GOLD, pointHoverBackgroundColor: GOLD },
            { label: "S&P 500 (SPY)", data: s.bench, borderColor: GREY, backgroundColor: GREY, borderWidth: 1.5 },
          ],
        },
        options: opts,
      });
    },
    drawdown(id, s) {
      const opts = base((y) => sign(y) + "%");
      opts.scales = axes((y) => y.toFixed(0) + "%", { max: 0 });
      new Chart(document.getElementById(id), {
        type: "line",
        data: {
          labels: s.dates,
          datasets: [{ label: "Drawdown", data: s.dd, borderColor: GOLD, borderWidth: 1.5, fill: "origin", backgroundColor: "rgba(169,133,43,0.10)" }],
        },
        options: opts,
      });
    },
    mini(id, dates, ic, baseline, range) {
      const opts = base((y) => (y == null ? "n/a" : sign(y, 3)));
      opts.scales = axes((y) => y.toFixed(2), { min: -range, max: range, ticks: { stepSize: range / 2, callback: (y) => (y === 0 ? "0" : y.toFixed(2)), font: { family: v("--mono"), size: 10 } } });
      opts.scales.x.ticks.maxTicksLimit = 3;
      const ds = [{ label: "Analyst", data: ic, borderColor: GOLD, borderWidth: 1.75, spanGaps: true }];
      if (baseline) ds.push({ label: "Factor", data: baseline, borderColor: GREY, borderWidth: 1.25, spanGaps: true });
      new Chart(document.getElementById(id), { type: "line", data: { labels: dates, datasets: ds }, options: opts });
    },
    calibration(id, rows) {
      const opts = base((y) => y.toFixed(0) + "%");
      opts.interaction = { mode: "index", intersect: false };
      opts.plugins.tooltip.callbacks = {
        title: (items) => "Conviction " + items[0].label,
        label: (c) => ` ${c.dataset.label}: ${c.parsed.y.toFixed(0)}%`,
        afterBody: (items) => ` n = ${rows[items[0].dataIndex].n}`,
      };
      opts.scales = axes((y) => y + "%", { min: 0, max: 100 });
      opts.scales.x.ticks.callback = function (val) { return this.getLabelForValue(val); };
      new Chart(document.getElementById(id), {
        data: {
          labels: rows.map((r) => r.bucket),
          datasets: [
            { type: "bar", label: "Beat the S&P 500", data: rows.map((r) => r.beat_rate), backgroundColor: GOLD, borderRadius: { topLeft: 4, topRight: 4 }, barPercentage: 0.6 },
            { type: "line", label: "Average conviction", data: rows.map((r) => r.avg_conviction), borderColor: GREY, borderWidth: 1.5, pointRadius: 3.5, pointBackgroundColor: "#ffffff", pointBorderColor: GREY, pointBorderWidth: 1.5 },
          ],
        },
        options: opts,
      });
    },
  };
})();
