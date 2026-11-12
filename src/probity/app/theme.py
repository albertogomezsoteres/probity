"""Visual design of the Probity app: colour tokens, small colour helpers, the stylesheet and the Gradio theme."""

from __future__ import annotations

import importlib.resources
import inspect
from typing import Any

import numpy as np

from ..probe import UNSURE

# ---------- design tokens (one dark system for the whole product) ----------
BG, BG2, SURFACE, SURFACE2, LINE, LINE2 = "#0A0D12", "#0F131A", "#141922", "#1A2029", "#232B36", "#2E3845"
INK, MUTED, FAINT = "#EDF1F5", "#98A3B3", "#65717F"
TRUTH, FALSE_, UNSURE_C = "#5AA9FF", "#FF6A5C", "#8A94A3"
TRUTH_SOFT, FALSE_SOFT, UNSURE_SOFT = "rgba(90,169,255,.16)", "rgba(255,106,92,.16)", "rgba(138,148,163,.16)"
GOOD = "#4ED1A1"


def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))


def _mix(c1, c2, t):
    a, b = _hex(c1), _hex(c2)
    return "#{:02x}{:02x}{:02x}".format(*tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3)))


def diverge(x):
    # x in [0, 1]: 0 = pushes toward false, .5 = neutral, 1 = pushes toward true
    x = float(np.clip(x, 0, 1))
    return _mix(FALSE_, SURFACE2, x / 0.5) if x < 0.5 else _mix(SURFACE2, TRUTH, (x - 0.5) / 0.5)


def col_p(p):
    return TRUTH if p >= UNSURE[1] else FALSE_ if p <= UNSURE[0] else UNSURE_C


def soft_p(p):
    return TRUTH_SOFT if p >= UNSURE[1] else FALSE_SOFT if p <= UNSURE[0] else UNSURE_SOFT


def verdict_words(p):
    return (
        "Gemma believes this is true"
        if p >= UNSURE[1]
        else "Gemma believes this is false"
        if p <= UNSURE[0]
        else "Gemma's hidden states are undecided"
    )


FLAG_SHORT = {
    "unfamiliar": "Unfamiliar sentence",
    "unsure": "Borderline",
    "negation": "Contains a negation",
    "not_claim": "Not a factual claim",
    "long": "Long sentence",
}
FLAG_LONG = {
    "unfamiliar": "Its activations are farther from the training data than 99% of held-out statements, so treat the reading as a guess.",
    "unsure": "The reading falls between 30% and 70%.",
    "negation": "This polygraph learned negations, but a probe trained without them reads negated sentences backwards (AUROC 0.01).",
    "not_claim": "A question, an opinion or small talk. It is shown but not counted in the verdict.",
    "long": "The polygraph was trained on short statements of under 20 words.",
}


def load_css() -> str:
    """The app stylesheet (``static/probity.css``)."""
    return importlib.resources.files("probity.app").joinpath("static/probity.css").read_text(encoding="utf-8")


def build_theme() -> Any:
    """Gradio theme with the Probity palette in both light and dark mode (the product is dark by design)."""
    import gradio as gr

    t = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )
    wanted = dict(
        body_background_fill=BG,
        body_text_color=INK,
        background_fill_primary=SURFACE,
        background_fill_secondary=BG2,
        block_background_fill=SURFACE,
        block_border_color=LINE,
        block_border_width="1px",
        block_shadow="none",
        block_radius="14px",
        block_label_text_color=MUTED,
        block_title_text_color=INK,
        border_color_primary=LINE2,
        color_accent=TRUTH,
        color_accent_soft="rgba(90,169,255,.16)",
        input_background_fill=SURFACE,
        input_border_color=LINE2,
        input_border_color_focus=TRUTH,
        input_shadow="none",
        input_radius="12px",
        input_placeholder_color=FAINT,
        body_text_color_subdued=MUTED,
        button_primary_background_fill=INK,
        button_primary_background_fill_hover="#FFFFFF",
        button_primary_text_color=BG,
        button_primary_border_color=INK,
        button_secondary_background_fill=SURFACE,
        button_secondary_background_fill_hover=SURFACE2,
        button_secondary_text_color=INK,
        button_secondary_border_color=LINE2,
        button_large_radius="12px",
        button_small_radius="10px",
        checkbox_background_color=SURFACE,
        checkbox_background_color_selected=TRUTH,
        checkbox_border_color=LINE2,
        checkbox_border_color_selected=TRUTH,
        checkbox_label_background_fill="transparent",
        checkbox_label_border_color="transparent",
        checkbox_label_text_color=MUTED,
        table_border_color=LINE,
        table_even_background_fill=SURFACE,
        table_odd_background_fill=SURFACE,
    )
    ok = inspect.signature(t.set).parameters
    both = {}
    for k, v in wanted.items():
        if k in ok:
            both[k] = v
        if k + "_dark" in ok:
            both[k + "_dark"] = v  # the same palette in both modes: the product is dark by design
    return t.set(**both)


HEAD = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500;600&display=swap'><meta name='color-scheme' content='dark'>"
)
PAGE_JS = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); window.location.replace(u.href); return; }
  const fit = () => { const w = window.innerWidth; const z = w >= 2300 ? 1.4 : w >= 1760 ? 1.2 : w >= 1500 ? 1.1 : 1;
    document.documentElement.style.setProperty('--pg-zoom', z); };
  fit(); window.addEventListener('resize', fit);
  const entered = () => { try { return sessionStorage.getItem('pgEntered') === '1'; } catch (e) { return false; } };
  const leave = (w) => { window.scrollTo(0, 0); w.classList.add('pg-leave'); setTimeout(() => w.remove(), 650); try { sessionStorage.setItem('pgEntered', '1'); } catch (e) {} };
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-pg-launch]'); if (b) { const w = document.getElementById('pg-welcome'); if (w) leave(w); }
    const c = e.target.closest('[data-copy]');
    if (c) { const pre = c.closest('.pg-codeblock').querySelector('pre'); const t = pre ? pre.innerText : '';
      const done = () => { c.textContent = 'Copied'; setTimeout(() => c.textContent = 'Copy', 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done).catch(done); else done(); }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { const w = document.getElementById('pg-welcome'); if (w) leave(w); } });
  let n = 0; const tick = setInterval(() => {
    const tpl = document.getElementById('pg-welcome-tpl');
    if (tpl && !document.getElementById('pg-welcome') && !entered()) { document.body.appendChild(tpl.content.firstElementChild.cloneNode(true)); tpl.remove(); clearInterval(tick); }
    if (tpl && entered()) { tpl.remove(); clearInterval(tick); }
    if (++n > 80) clearInterval(tick); }, 150);
}"""

HEAD = (
    "<link rel='preconnect' href='https://fonts.googleapis.com'><link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
    "<link rel='stylesheet' href='https://fonts.googleapis.com/css2?family=Sora:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600"
    "&family=IBM+Plex+Mono:wght@400;500;600&display=swap'><meta name='color-scheme' content='dark'>"
)
PAGE_JS = """() => {
  const u = new URL(window.location.href);
  if (u.searchParams.get('__theme') !== 'dark') { u.searchParams.set('__theme', 'dark'); window.location.replace(u.href); return; }
  const fit = () => { const w = window.innerWidth; const z = w >= 2300 ? 1.4 : w >= 1760 ? 1.2 : w >= 1500 ? 1.1 : 1;
    document.documentElement.style.setProperty('--pg-zoom', z); };
  fit(); window.addEventListener('resize', fit);
  const entered = () => { try { return sessionStorage.getItem('pgEntered') === '1'; } catch (e) { return false; } };
  const leave = (w) => { window.scrollTo(0, 0); w.classList.add('pg-leave'); setTimeout(() => w.remove(), 650); try { sessionStorage.setItem('pgEntered', '1'); } catch (e) {} };
  document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-pg-launch]'); if (b) { const w = document.getElementById('pg-welcome'); if (w) leave(w); }
    const c = e.target.closest('[data-copy]');
    if (c) { const pre = c.closest('.pg-codeblock').querySelector('pre'); const t = pre ? pre.innerText : '';
      const done = () => { c.textContent = 'Copied'; setTimeout(() => c.textContent = 'Copy', 1500); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(t).then(done).catch(done); else done(); }
  });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { const w = document.getElementById('pg-welcome'); if (w) leave(w); } });
  let n = 0; const tick = setInterval(() => {
    const tpl = document.getElementById('pg-welcome-tpl');
    if (tpl && !document.getElementById('pg-welcome') && !entered()) { document.body.appendChild(tpl.content.firstElementChild.cloneNode(true)); tpl.remove(); clearInterval(tick); }
    if (tpl && entered()) { tpl.remove(); clearInterval(tick); }
    if (++n > 80) clearInterval(tick); }, 150);
}"""
