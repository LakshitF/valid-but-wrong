/* =========================================================
   Valid but Wrong — interactions
   Progressive enhancement: the page reads fine without JS.
   ========================================================= */

(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  /* ── reading progress ─────────────────────────────────── */

  const progressBar = $("#progressBar");
  let progressTick = false;

  const updateProgress = () => {
    if (!progressBar) return;
    const doc = document.documentElement;
    const max = doc.scrollHeight - doc.clientHeight;
    const value = max > 0 ? (doc.scrollTop / max) * 100 : 0;
    progressBar.style.width = `${Math.min(100, Math.max(0, value))}%`;
    progressTick = false;
  };

  window.addEventListener("scroll", () => {
    if (!progressTick) {
      progressTick = true;
      window.requestAnimationFrame(updateProgress);
    }
  }, { passive: true });
  window.addEventListener("resize", updateProgress, { passive: true });
  updateProgress();

  /* ── reveal on scroll ─────────────────────────────────── */

  const revealTargets = $$(".reveal, .reveal-bars");
  if ("IntersectionObserver" in window) {
    const revealObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-in");
        revealObserver.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
    revealTargets.forEach((el) => revealObserver.observe(el));
  } else {
    revealTargets.forEach((el) => el.classList.add("is-in"));
  }

  /* ── scrollspy ────────────────────────────────────────── */

  const tocLinks = $$("#toc a");
  const linkById = new Map(
    tocLinks.map((link) => [link.getAttribute("href").slice(1), link])
  );

  if ("IntersectionObserver" in window && tocLinks.length) {
    const spy = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        tocLinks.forEach((link) => link.classList.remove("active"));
        const active = linkById.get(entry.target.id);
        if (active) active.classList.add("active");
      });
    }, { rootMargin: "-14% 0px -70% 0px", threshold: 0 });

    $$(".section[id]").forEach((section) => spy.observe(section));
  }

  /* ── abstention / permission arm ──────────────────────── */

  const armPanel = $("#policy-arm");
  const armData = {
    base: {
      silent: 54.8, silentCount: "115 / 210",
      shifted: 34.7, shiftedCount: "25 / 72",
      echo: 38.1, echoCount: "80 / 210",
      clarify: 0, clarifyCount: "0 / 210",
      note: "With the base prompt, the model is licensed to do exactly one thing: emit a call."
    },
    permission: {
      silent: 34.8, silentCount: "73 / 210",
      shifted: 22.2, shiftedCount: "16 / 72",
      echo: 0.5, echoCount: "1 / 210",
      clarify: 64.8, clarifyCount: "136 / 210",
      note: "One added sentence: most of the silence becomes a question. Clean items remain untouched."
    }
  };

  const setArm = (key) => {
    if (!armPanel || !armData[key]) return;
    const data = armData[key];

    $$("[data-arm]", armPanel).forEach((button) => {
      const active = button.dataset.arm === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    armPanel.classList.toggle("show-added", key === "permission");

    ["silent", "shifted", "echo", "clarify"].forEach((metric) => {
      const value = $(`[data-arm-value="${metric}"]`, armPanel);
      const count = $(`[data-arm-count="${metric}"]`, armPanel);
      const bar = $(`[data-arm-bar="${metric}"]`, armPanel);
      if (value) value.textContent = `${data[metric].toFixed(1)}%`;
      if (count) count.textContent = data[`${metric}Count`];
      if (bar) bar.style.setProperty("--w", `${data[metric]}%`);
    });

    const note = $("[data-arm-note]", armPanel);
    if (note) note.textContent = data.note;
  };

  $$("[data-arm]", armPanel || document).forEach((button) => {
    button.addEventListener("click", () => setArm(button.dataset.arm));
  });

  /* ── validation retry ─────────────────────────────────── */

  const retryPanel = $("#retry-arm");
  const retryData = {
    list: {
      "7b": { all: 78.7, allCount: "63 / 80", shifted: 69.2, shiftedCount: "27 / 39" },
      "3b": { all: 72.4, allCount: "131 / 181", shifted: 69.1, shiftedCount: "47 / 68" },
      mistral: { all: 94.5, allCount: "155 / 164", shifted: 94.2, shiftedCount: "65 / 69" }
    },
    nolist: {
      "7b": { all: 86.2, allCount: "69 / 80", shifted: 92.3, shiftedCount: "36 / 39" },
      "3b": { all: 33.7, allCount: "61 / 181", shifted: 23.5, shiftedCount: "16 / 68" },
      mistral: { all: 78.7, allCount: "129 / 164", shifted: 71.0, shiftedCount: "49 / 69" }
    }
  };

  const retryNotes = {
    list: "With the allowed values in the error, most of the loud failures become silent substitutions on the next turn.",
    nolist: "Even with no values named, every model still converts loud failures into silent ones. The 3B converts far fewer; its earlier number leaned on the list."
  };

  const setRetry = (key) => {
    if (!retryPanel || !retryData[key]) return;

    $$("[data-retry-toggle]", retryPanel).forEach((button) => {
      const active = button.dataset.retryToggle === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    Object.entries(retryData[key]).forEach(([model, data]) => {
      const card = $(`[data-retry-card="${model}"]`, retryPanel);
      if (!card) return;

      ["all", "shifted"].forEach((metric) => {
        const value = $(`[data-retry-value="${metric}"]`, card);
        const count = $(`[data-retry-count="${metric}"]`, card);
        const bar = $(`[data-retry-bar="${metric}"]`, card);
        if (value) value.textContent = `${data[metric].toFixed(1)}%`;
        if (count) count.textContent = data[`${metric}Count`];
        if (bar) bar.style.setProperty("--w", `${data[metric]}%`);
      });
    });

    const note = $("[data-retry-note]", retryPanel);
    if (note) note.textContent = retryNotes[key];
  };

  $$("[data-retry-toggle]", retryPanel || document).forEach((button) => {
    button.addEventListener("click", () => setRetry(button.dataset.retryToggle));
  });

  /* ── provider-side enforcement ────────────────────────── */

  const enfPanel = $("#enforcement-arm");
  const enfData = {
    closed: {
      sub: "210 / 210",
      shifted: "72 / 72",
      echo: "0 / 210",
      status: "enum present · provider rejects non-members",
      note: "With the enum enforced, the invalid call is impossible. Every awkward instruction becomes a substitution, and none can raise."
    },
    open: {
      sub: "0 / 210",
      shifted: "0 / 72",
      echo: "210 / 210",
      status: "enum removed · the supplied value is expressible",
      note: "With the enum removed, the same model echoes every awkward item. The ability to represent the value was there all along; the constraint produced the substitution."
    }
  };

  const setEnforcement = (key) => {
    if (!enfPanel || !enfData[key]) return;
    const data = enfData[key];

    $$("[data-enf]", enfPanel).forEach((button) => {
      const active = button.dataset.enf === key;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-pressed", String(active));
    });

    ["sub", "shifted", "echo"].forEach((metric) => {
      const el = $(`[data-enf-value="${metric}"]`, enfPanel);
      if (el) el.textContent = data[metric];
    });

    enfPanel.classList.toggle("enum-removed", key === "open");

    const status = $("[data-enum-status]", enfPanel);
    if (status) status.textContent = data.status;

    const note = $("[data-enf-note]", enfPanel);
    if (note) note.textContent = data.note;
  };

  $$("[data-enf]", enfPanel || document).forEach((button) => {
    button.addEventListener("click", () => setEnforcement(button.dataset.enf));
  });

  /* ── detection-plane legend highlight ─────────────────── */

  $$(".legend-item").forEach((item) => {
    const point = document.getElementById(`pt-${item.dataset.point}`);
    if (!point) return;

    const highlight = () => point.classList.add("is-hot");
    const clear = () => point.classList.remove("is-hot");

    item.addEventListener("mouseenter", highlight);
    item.addEventListener("mouseleave", clear);
    item.addEventListener("focus", highlight);
    item.addEventListener("blur", clear);
  });

  /* ── copy citation ────────────────────────────────────── */

  const copyButton = $("#copyCite");
  const bibtex = $("#bibtex");

  const copyText = async (text) => {
    if (navigator.clipboard && window.isSecureContext) {
      try {
        await navigator.clipboard.writeText(text);
        return true;
      } catch (_) {
        /* fall through to the legacy path */
      }
    }

    const area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.top = "-9999px";
    document.body.appendChild(area);
    area.select();

    let copied = false;
    try {
      copied = document.execCommand("copy");
    } catch (_) {
      copied = false;
    }
    area.remove();
    return copied;
  };

  if (copyButton && bibtex) {
    copyButton.addEventListener("click", async () => {
      const copied = await copyText(bibtex.textContent.trim());
      const original = copyButton.textContent;
      copyButton.textContent = copied ? "Copied ✓" : "Copy failed";
      if (!copied) {
        const details = copyButton.closest(".section")?.querySelector(".cite");
        if (details) details.open = true;
      }
      window.setTimeout(() => {
        copyButton.textContent = original;
      }, 1800);
    });
  }

})();
