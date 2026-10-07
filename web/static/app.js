/* Page behavior: sortable tables, table filters, definition popovers, contents highlighting. */
(function () {
  // ---- Sortable tables: every header in a table.sortable becomes a button ----
  const value = (td) => {
    if (!td) return null;
    const raw = (td.dataset.v ?? td.textContent).trim();
    if (raw === "" || raw === "n/a" || raw === "pending" || raw === "-") return null;
    const n = parseFloat(raw.replace(/[−–]/g, "-").replace(/[$,%+x]/g, ""));
    return Number.isNaN(n) || /^[A-Za-z]/.test(raw) ? raw.toLowerCase() : n;
  };
  document.querySelectorAll("table.sortable").forEach((table) => {
    const heads = [...table.tHead.rows[0].cells];
    const body = table.tBodies[0];
    heads.forEach((th, i) => {
      if ("nosort" in th.dataset) return;
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sort";
      btn.innerHTML = th.innerHTML;
      th.textContent = "";
      th.appendChild(btn);
      th.setAttribute("aria-sort", "none");
      btn.addEventListener("click", () => {
        const dir = th.getAttribute("aria-sort") === "descending" ? "ascending" : "descending";
        heads.forEach((h) => h.hasAttribute("aria-sort") && h.setAttribute("aria-sort", "none"));
        th.setAttribute("aria-sort", dir);
        const sign = dir === "ascending" ? 1 : -1;
        const rows = [...body.rows].filter((r) => !r.classList.contains("no-match"));
        rows.sort((a, b) => {
          const x = value(a.cells[i]), y = value(b.cells[i]);
          if (x === null) return 1; // blanks always last
          if (y === null) return -1;
          if (typeof x === typeof y) return (x > y ? 1 : x < y ? -1 : 0) * sign;
          return typeof x === "number" ? -1 : 1;
        });
        rows.forEach((r) => body.appendChild(r));
      });
    });
  });

  // ---- Filters: <input data-filter="table-id"> hides rows that don't match ----
  document.querySelectorAll("input[data-filter]").forEach((input) => {
    const table = document.getElementById(input.dataset.filter);
    const count = document.querySelector(`[data-count="${input.dataset.filter}"]`);
    if (!table) return;
    const body = table.tBodies[0];
    const rows = [...body.rows];
    const cols = table.tHead.rows[0].cells.length;
    const none = document.createElement("tr");
    none.className = "no-match";
    none.innerHTML = `<td colspan="${cols}">No rows match.</td>`;
    const update = () => {
      const q = input.value.trim().toLowerCase();
      let shown = 0;
      rows.forEach((r) => {
        const hit = !q || r.textContent.toLowerCase().includes(q);
        r.hidden = !hit;
        shown += hit;
      });
      if (!shown) body.appendChild(none);
      else none.remove();
      if (count) count.textContent = q ? `${shown} of ${rows.length}` : `${rows.length} rows`;
    };
    input.addEventListener("input", update);
    update();
  });

  // ---- Definitions: one shared popover for every [data-tip] ----
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  tip.id = "tip";
  document.body.appendChild(tip);
  const show = (el) => {
    tip.textContent = el.dataset.tip;
    tip.classList.add("show");
    const r = el.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    let top = r.top - h - 8;
    if (top < 8) top = r.bottom + 8;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
    el.setAttribute("aria-describedby", "tip");
  };
  const hide = (el) => {
    tip.classList.remove("show");
    if (el) el.removeAttribute("aria-describedby");
  };
  document.querySelectorAll("[data-tip]").forEach((el) => {
    if (!el.hasAttribute("tabindex") && !el.closest("button")) el.setAttribute("tabindex", "0");
    el.addEventListener("mouseenter", () => show(el));
    el.addEventListener("mouseleave", () => hide(el));
    el.addEventListener("focus", () => show(el));
    el.addEventListener("blur", () => hide(el));
  });
  document.addEventListener("keydown", (e) => e.key === "Escape" && hide());
  window.addEventListener("scroll", () => hide(), { passive: true });

  // ---- Methodology contents: highlight the section in view ----
  const toc = document.querySelector(".toc");
  if (toc && "IntersectionObserver" in window) {
    const links = new Map([...toc.querySelectorAll("a")].map((a) => [a.getAttribute("href").slice(1), a]));
    const io = new IntersectionObserver(
      (entries) => entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove("on"));
        links.get(e.target.id)?.classList.add("on");
      }),
      { rootMargin: "-20% 0px -70% 0px" }
    );
    links.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }
})();
