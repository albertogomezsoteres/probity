"""HTML views of the Probity app: chat panes, the polygraph gauge, explanations, brain scan, monitor and docs.

Every function returns an HTML string styled by ``static/probity.css``. Nothing here touches the model; numbers
come from the engine (per request) or from :class:`~probity.app.info.AppInfo` (published statistics).
"""

from __future__ import annotations

import base64
import html
from typing import Any

import numpy as np
import pandas as pd

from ..probe import UNSURE
from .info import AppInfo
from .theme import (
    BG,
    FAINT,
    FALSE_,
    FLAG_LONG,
    FLAG_SHORT,
    INK,
    LINE,
    LINE2,
    MUTED,
    SURFACE,
    TRUTH,
    UNSURE_C,
    UNSURE_SOFT,
    col_p,
    diverge,
    soft_p,
    verdict_words,
)

E = html.escape


# ---------- brand marks ----------
def mark_svg(cls="", w=48):
    # Probity mark: a polygraph trace that climbs into the letter P, with the alarm dot at the peak
    return (
        f"<svg class='{cls}' viewBox='0 0 64 64' aria-hidden='true'><path d='M4 41 H12 L16 35 L20 46 L24 41 H27 V9 H39 C50 9 56 15.5 56 24.5 "
        f"C56 33.5 50 40 39 40 H27 V57' fill='none' stroke='{INK}' stroke-width='5.6' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<circle cx='27' cy='9' r='5.2' fill='{FALSE_}'/></svg>"
    )


def signal_svg():
    # a stylised residual-stream read-out: 26 layer nodes, one truth direction, an animated polygraph trace
    nodes = "".join(
        f"<circle class='node' cx='{40 + i * 14.4:.1f}' cy='{150 + 38 * np.sin(i / 2.3):.1f}' r='{2.2 + (i % 4) * 0.5:.1f}' fill='{TRUTH if i == 13 else LINE2}'/>"
        for i in range(26)
    )
    path = " ".join(f"{40 + i * 14.4:.1f},{150 + 38 * np.sin(i / 2.3):.1f}" for i in range(26))
    trace = (
        "M20,250 H110 L124,236 L136,262 L148,248 H170 L184,196 L196,268 L210,244 H260 L274,236 L288,256 L300,250 H400"
    )
    return (
        f"<svg class='pg-signal' viewBox='0 0 420 300' role='img' aria-label='Stylised view of a truth signal read from the hidden layers of a language model'>"
        f"<defs><linearGradient id='pgg' x1='0' x2='1'><stop offset='0' stop-color='{FALSE_}'/><stop offset='1' stop-color='{TRUTH}'/></linearGradient></defs>"
        f"<text x='40' y='98' font-family='IBM Plex Mono, monospace' font-size='10' fill='{FAINT}' letter-spacing='2'>LAYER 1</text>"
        f"<text x='400' y='98' font-family='IBM Plex Mono, monospace' font-size='10' fill='{FAINT}' letter-spacing='2' text-anchor='end'>LAYER 26</text>"
        f"<polyline points='{path}' fill='none' stroke='{LINE2}' stroke-width='1.2'/>{nodes}"
        f"<circle cx='{40 + 13 * 14.4:.1f}' cy='{150 + 38 * np.sin(13 / 2.3):.1f}' r='9' fill='none' stroke='{TRUTH}' stroke-width='1.5' opacity='.7'/>"
        f"<text x='{40 + 13 * 14.4:.1f}' y='{150 + 38 * np.sin(13 / 2.3) - 18:.1f}' font-family='IBM Plex Mono, monospace' font-size='10' fill='{TRUTH}' text-anchor='middle' letter-spacing='1'>READ HERE</text>"
        f"<path class='trace' d='{trace}' fill='none' stroke='url(#pgg)' stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>"
        f"<text x='20' y='290' font-family='IBM Plex Mono, monospace' font-size='10' fill='{FAINT}' letter-spacing='2'>POLYGRAPH TRACE</text></svg>"
    )


def welcome_html(info: AppInfo) -> str:
    g = info.guard_summary["results_test"] if info.guard_summary else {}
    stats = [(f"{info.AUC_TEST[info.layer]:.3f}", "AUROC, held-out statements")]
    if "pushback_assertive" in g:
        v = g["pushback_assertive"]
        stats.append((f"{v['before'] * 100:.0f}% → {v['after'] * 100:.0f}%", "caving under pushback, with the guard"))
    if "instructed_lie" in g:
        v = g["instructed_lie"]
        stats.append((f"{(1 - v['after'] / max(v['before'], 1e-9)) * 100:.0f}%", "of instructed lies blocked"))
    foot = "".join(f"<div><b>{E(a)}</b><span>{E(b)}</span></div>" for a, b in stats)
    return (
        f"<template id='pg-welcome-tpl'><div id='pg-welcome' class='pg' style='background-color:{BG}'><div class='pg-welcome-inner'><div class='pg-hero'>"
        f"<div class='pg-wordmark'>{mark_svg()}Probity</div>"
        f"<h1>What if AI could reveal when it's wrong?</h1>"
        f"<p>Probity reads the hidden states of an open language model while it talks, flags the claims the model itself "
        f"believes are false, and stops them before they reach you.</p>"
        f"<button class='pg-launch' data-pg-launch type='button'>Explore Probity"
        f"<svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2.2' stroke-linecap='round' stroke-linejoin='round'><path d='M5 12h14M13 6l6 6-6 6'/></svg></button>"
        f"<div class='pg-hero-foot'>{foot}</div>"
        f"<p class='pg-hero-note'>Gemma 2 2B-it · linear probes on the residual stream · AIPI 590, Duke University</p>"
        f"</div><div>{signal_svg()}</div></div></div></template>"
    )


def header_html(info: AppInfo) -> str:
    tau = f"<div><span>Guard threshold</span><b>τ = {info.tau:.2f}</b></div>" if info.guard_summary else ""
    return (
        f"<div class='pg'><header class='pg-header'><div class='pg-brand'>{mark_svg()}<div><h1>Probity</h1>"
        f"<span>An AI polygraph for open language models</span></div></div>"
        f"<div class='pg-meta'><div><span>Model</span><b>Gemma 2 2B-it</b></div><div><span>Reads layer</span><b>{info.layer} / {info.n_layers}</b></div>"
        f"<div><span>Held-out AUROC</span><b>{info.AUC_TEST[info.layer]:.3f}</b></div>{tau}</div></header></div>"
    )


# ---------- gauge ----------
def dial_svg(p, trusted):
    cx, cy, R = 150, 142, 112

    def pt(frac, r):
        th = np.pi * (1 - frac)
        return cx + r * np.cos(th), cy - r * np.sin(th)

    def arc(a0, a1, color, width=10, op=1):
        (x0, y0), (x1, y1) = pt(a0, R), pt(a1, R)
        return f"<path d='M{x0:.1f},{y0:.1f} A{R},{R} 0 0 1 {x1:.1f},{y1:.1f}' stroke='{color}' stroke-width='{width}' fill='none' opacity='{op}' stroke-linecap='round'/>"

    ticks = "".join(
        f"<line x1='{pt(k / 10, R - 16)[0]:.1f}' y1='{pt(k / 10, R - 16)[1]:.1f}' x2='{pt(k / 10, R - (24 if k % 5 == 0 else 20))[0]:.1f}' y2='{pt(k / 10, R - (24 if k % 5 == 0 else 20))[1]:.1f}' "
        f"stroke='{MUTED if k % 5 == 0 else LINE2}' stroke-width='{2 if k % 5 == 0 else 1.5}'/>"
        for k in range(1, 10)
    )
    nx, ny = pt(p, R - 32)
    needle = INK if trusted else UNSURE_C
    dash = "" if trusted else "stroke-dasharray='7 5'"
    return (
        f"<svg class='pg-dial' viewBox='0 0 300 172' role='img' aria-label='Polygraph reading {p * 100:.0f} percent'>"
        f"{arc(0, 1, LINE, 10)}{arc(0, UNSURE[0], FALSE_)}{arc(UNSURE[1], 1, TRUTH)}{ticks}"
        f"<g class='needle'><line x1='{cx}' y1='{cy}' x2='{nx:.1f}' y2='{ny:.1f}' stroke='{needle}' stroke-width='4.5' stroke-linecap='round' {dash}/></g>"
        f"<circle cx='{cx}' cy='{cy}' r='9' fill='{needle}'/><circle cx='{cx}' cy='{cy}' r='3.5' fill='{SURFACE}'/>"
        f"<text x='{pt(0, R)[0]:.1f}' y='166' font-size='12' font-family='IBM Plex Mono, monospace' fill='{FALSE_}' text-anchor='middle'>FALSE</text>"
        f"<text x='{pt(1, R)[0]:.1f}' y='166' font-size='12' font-family='IBM Plex Mono, monospace' fill='{TRUTH}' text-anchor='middle'>TRUE</text></svg>"
    )


def trace_svg(state):
    pts = [(t["verdict"]["p"], t.get("tag")) for t in state["turns"] if t["role"] == "assistant" and t["verdict"]][-6:]
    if len(pts) < 2:
        return ""
    W_, H_, x0, x1, y0, y1 = 300, 122, 36, 292, 22, 80
    xs = [x0 + 30 + (x1 - x0 - 60) * i / (len(pts) - 1) for i in range(len(pts))]
    ys = [y1 - (y1 - y0) * p for p, _ in pts]
    grid = "".join(
        f"<line x1='{x0}' x2='{x1}' y1='{y1 - (y1 - y0) * v:.1f}' y2='{y1 - (y1 - y0) * v:.1f}' stroke='{LINE}' stroke-width='1'/>"
        f"<text x='{x0 - 6}' y='{y1 - (y1 - y0) * v + 4:.1f}' font-size='10' font-family='IBM Plex Mono, monospace' fill='{FAINT}' text-anchor='end'>{int(v * 100)}%</text>"
        for v in (0, 0.5, 1)
    )
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    dots, labs = [], []
    for i, ((p, tag), x, y) in enumerate(zip(pts, xs, ys)):
        dots.append(
            f"<circle cx='{x:.1f}' cy='{y:.1f}' r='5.5' fill='{col_p(p)}' stroke='{SURFACE}' stroke-width='2'/>"
        )
        ty = y - 10 if y > y0 + 8 else y + 19
        dots.append(
            f"<text x='{x:.1f}' y='{ty:.1f}' font-size='10.5' font-family='IBM Plex Mono, monospace' font-weight='600' fill='{col_p(p)}' text-anchor='middle'>{p * 100:.0f}%</text>"
        )
        sub = {"push": "after pushback", "lie": "compromised prompt"}.get(tag, "")
        labs.append(
            f"<text x='{x:.1f}' y='{y1 + 18}' font-size='10.5' fill='{MUTED}' text-anchor='middle'>Reply {i + 1}</text>"
        )
        if sub:
            labs.append(
                f"<text x='{x:.1f}' y='{y1 + 31}' font-size='9.5' fill='{FAINT}' text-anchor='middle'>{sub}</text>"
            )
    return (
        f"<div class='pg-trace'><div class='pg-title'>Reading across the conversation</div>"
        f"<svg viewBox='0 0 {W_} {H_}' role='img' aria-label='Polygraph readings for each reply'>{grid}"
        f"<polyline points='{path}' fill='none' stroke='{INK}' stroke-width='2' stroke-linejoin='round'/>{''.join(dots)}{''.join(labs)}</svg></div>"
    )


def gauge_html(v, state=None):
    if v is None:
        return (
            f"<div class='pg'><div class='pg-panel pg-gauge'><div class='pg-eyebrow'>Polygraph reading</div>{dial_svg(0.5, False)}"
            f"<p class='pg-sub' style='margin-top:8px'>Waiting for a factual claim from Gemma.</p></div></div>"
        )
    p = v["p"]
    warn = (
        ""
        if v["trusted"]
        else "<div class='pg-warn'>Low confidence: this sentence is unlike anything the polygraph was trained on, so the needle is a guess.</div>"
    )
    return (
        f"<div class='pg'><div class='pg-panel pg-gauge'><div class='pg-eyebrow'>Polygraph reading · latest answer</div>{dial_svg(p, v['trusted'])}"
        f"<div class='pg-reading' style='color:{col_p(p)}'>{p * 100:.0f}%</div><div class='pg-verdict'>{verdict_words(p)}</div>"
        f"<p class='pg-quote'>“{E(v['sentence'][:140])}”</p>"
        f"<p class='pg-note'>Weakest of {v['n_claims']} claim{'s' if v['n_claims'] != 1 else ''} in this reply</p>"
        f"{warn}{trace_svg(state) if state else ''}</div></div>"
    )


# ---------- chat ----------
def claim_span(r):
    if "not_claim" in r["flags"]:
        return f"<span class='pg-plain'>{E(r['sentence'])}</span>"
    c, s = col_p(r["p"]), soft_p(r["p"])
    mark = " ⚠" if "unfamiliar" in r["flags"] else ""
    return (
        f"<span class='pg-claim' style='background:{s};border-color:{c}' title='Polygraph: {r['p'] * 100:.0f}% true'>{E(r['sentence'])}</span>"
        f"<span class='pg-pct' style='color:{c}'>{r['p'] * 100:.0f}%{mark}</span>"
    )


def details_html(rows: list[dict[str, Any]], layer: int) -> str:
    if not rows:
        return ""
    items, used = [], set()
    for r in rows:
        nc = "not_claim" in r["flags"]
        c, s = (UNSURE_C, UNSURE_SOFT) if nc else (col_p(r["p"]), soft_p(r["p"]))
        notes = []
        if r["claim"] != r["sentence"]:
            notes.append(f"Claim read: “{E(r['claim'])}”")
        if r["p_in_context"] is not None and not nc:
            notes.append(
                f"Read inside the conversation it would score {r['p_in_context'] * 100:.0f}%. The needle ignores that number on purpose."
            )
        chips = "".join(f"<span class='pg-chip' title='{E(FLAG_LONG[f])}'>{FLAG_SHORT[f]}</span>" for f in r["flags"])
        used |= set(r["flags"])
        items.append(
            f"<li><span class='pg-pill' style='background:{s};color:{c}'>{r['p'] * 100:.0f}%</span>"
            f"<div class='pg-row-text'><p>{E(r['sentence'])}</p>{''.join(f'<p class=pg-note>{n}</p>' for n in notes)}"
            f"{f'<div class=pg-chips>{chips}</div>' if chips else ''}</div></li>"
        )
    legend = (
        (
            "<details><summary>What the flags mean</summary><dl>"
            + "".join(f"<dt>{FLAG_SHORT[f]}</dt><dd>{FLAG_LONG[f]}</dd>" for f in FLAG_SHORT if f in used)
            + "</dl></details>"
        )
        if used
        else ""
    )
    return (
        f"<div class='pg'><div class='pg-panel'><div class='pg-eyebrow'>Sentence by sentence</div>"
        f"<p class='pg-sub' style='margin-top:6px'>Each sentence is read alone at layer {layer}, the setting that held up in our tests.</p>"
        f"<ul class='pg-rows'>{''.join(items)}</ul>{legend}</div></div>"
    )


def pressure_html(before, after):
    if before is None:
        return ""
    if after is None:
        return "<div class='pg'><div class='pg-panel'><div class='pg-eyebrow'>After pushback</div><p class='pg-sub' style='margin-top:6px'>Gemma's new reply has no factual claim to read.</p></div></div>"
    if after["p"] <= UNSURE[0] < before["p"]:
        head = "Gemma gave in, and its own hidden states mark the new answer as false."
    elif (after["p"] >= UNSURE[1] and before["p"] >= UNSURE[1]) or abs(after["p"] - before["p"]) < 0.2:
        head = "Gemma held its ground."
    else:
        head = "The reading moved after the pushback."

    def bar(label, p):
        return (
            f"<span>{label}</span><div class='pg-track'><div class='pg-fill' style='width:{max(p * 100, 2):.0f}%;background:{col_p(p)}'></div></div>"
            f"<span class='pg-num' style='color:{col_p(p)}'>{p * 100:.0f}%</span>"
        )

    return (
        f"<div class='pg'><div class='pg-panel'><div class='pg-eyebrow'>After pushback · standard chatbot</div><p class='pg-headline'>{head}</p>"
        f"<div class='pg-bars'>{bar('Before', before['p'])}{bar('After', after['p'])}</div>"
        "<p class='pg-foot'>In our tests Gemma gave in to confident pushback 31% of the time (126 of 410). Reading each claim on its own, "
        "the polygraph told answers it gave in on from answers it held with AUROC 0.98.</p></div></div>"
    )


# ---------- explanations ----------
def word_chip(w, a, lim):
    x = 0.5 + 0.5 * a / lim
    fg = INK if abs(a) / lim < 0.45 else "#0B0F15"
    return f"<span class='pg-word' style='background:{diverge(x)};color:{fg}' title='{a:+.3f}'>{E(w)}</span>"


def explanation_html(ex: dict[str, Any], info: AppInfo) -> str:
    p = float(info.calibrated(ex["score"]))
    v = "true" if ex["score"] > 0 else "false"
    blocks = []
    for m, d in ex["methods"].items():
        a = np.asarray(d["attr"], float)
        lim = float(np.max(np.abs(a))) or 1.0
        c = d["check"]
        status = (
            "<span class='pg-status ok'>Passes the deletion check</span>"
            if c["faithful"]
            else "<span class='pg-status bad'>Fails the deletion check, do not rely on it</span>"
        )
        blocks.append(
            f"<div class='pg-method'><div class='pg-method-head'><h3>{E(m)}</h3>{status}</div>"
            f"<div class='pg-words'>{''.join(word_chip(w, x, lim) for w, x in zip(ex['words'], a))}</div>"
            f"<p class='pg-explain'>Removing its top {c['k']} word{'s' if c['k'] != 1 else ''} ({E(', '.join(c['top_words']))}) moved the reading by "
            f"{c['moved_top']:+.2f} against the verdict. Removing {c['k']} random word{'s' if c['k'] != 1 else ''} moved it by {c['moved_random']:+.2f}.</p></div>"
        )
    agree = ""
    if len(ex["methods"]) > 1:
        from scipy.stats import spearmanr

        ms = list(ex["methods"])
        rs = []
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                rho = spearmanr(ex["methods"][ms[i]]["attr"], ex["methods"][ms[j]]["attr"]).correlation
                rs.append(f"{ms[i]} and {ms[j]}: ρ = {rho:.2f}")
        agree = (
            f"<div class='pg-method'><h3 style='font-size:17px'>Do the methods agree?</h3><p class='pg-explain'>{'; '.join(rs)}. "
            "In our study the three methods picked the same top word only 48 to 64% of the time, so an explanation is evidence to check, not ground truth.</p></div>"
        )
    return (
        f"<div class='pg'><div class='pg-panel'><div class='pg-eyebrow'>Why the polygraph reads this as {v} · {p * 100:.0f}%</div>"
        f"<p class='pg-quote' style='margin:8px 0 0;max-width:none;text-align:left'>“{E(ex['sentence'])}”</p>"
        f"<div class='pg-scale'><span>Pushes toward false</span><i></i><span>Pushes toward true</span></div>{''.join(blocks)}{agree}</div></div>"
    )


# ---------- brain scan ----------
def scan_svg(row: dict[str, Any], info: AppInfo) -> str:
    layers, auc_test, l_app = info.all_layers, info.AUC_TEST, info.layer
    W_, H_ = 820, 330
    x0, x1, y0, y1 = 56, 800, 24, 262
    n = len(layers)

    def X(L):
        return x0 + (x1 - x0) * (L - layers[0]) / (n - 1)

    def Y(p):
        return y1 - (y1 - y0) * p

    bands = (
        f"<rect x='{x0}' y='{Y(1.0):.1f}' width='{x1 - x0}' height='{Y(UNSURE[1]) - Y(1.0):.1f}' fill='{TRUTH}' opacity='.07'/>"
        f"<rect x='{x0}' y='{Y(UNSURE[0]):.1f}' width='{x1 - x0}' height='{Y(0) - Y(UNSURE[0]):.1f}' fill='{FALSE_}' opacity='.07'/>"
    )
    grid = "".join(
        f"<line x1='{x0}' x2='{x1}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='{LINE}'/>"
        f"<text x='{x0 - 10}' y='{Y(v) + 4:.1f}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{FAINT}' text-anchor='end'>{int(v * 100)}%</text>"
        for v in (0, 0.5, 1)
    )
    xt = "".join(
        f"<text x='{X(L):.1f}' y='{y1 + 20}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{FAINT}' text-anchor='middle'>{L}</text>"
        for L in layers
        if L in (1, 5, 10, 15, 20, 25) or layers[-1] == L
    )
    auc = " ".join(f"{X(L):.1f},{Y(auc_test[L]):.1f}" for L in layers)
    ps = row["layer_p"]
    line = " ".join(f"{X(L):.1f},{Y(p):.1f}" for L, p in zip(layers, ps))
    dots = "".join(
        f"<circle class='pt' cx='{X(L):.1f}' cy='{Y(p):.1f}' r='4.5' fill='{col_p(p)}' stroke='{SURFACE}' stroke-width='1.5'>"
        f"<title>Layer {L}: {p * 100:.0f}% believed true · probe AUROC at this layer {auc_test[L]:.3f}</title></circle>"
        for L, p in zip(layers, ps)
    )
    needle = (
        f"<line x1='{X(l_app):.1f}' x2='{X(l_app):.1f}' y1='{y0 - 6}' y2='{y1}' stroke='{INK}' stroke-dasharray='3 4'/>"
        f"<rect x='{X(l_app) + 6:.1f}' y='{y1 - 30}' width='172' height='20' rx='5' fill='{SURFACE}' opacity='.92'/>"
        f"<text x='{X(l_app) + 12:.1f}' y='{y1 - 16}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{INK}'>NEEDLE READS LAYER {l_app}</text>"
    )
    labels = (
        f"<text x='{x1}' y='{Y(0.86):.1f}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{TRUTH}' text-anchor='end'>READS TRUE</text>"
        f"<text x='{x1}' y='{Y(0.12):.1f}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{FALSE_}' text-anchor='end'>READS FALSE</text>"
        f"<text x='{(x0 + x1) / 2:.0f}' y='{H_ - 6}' font-size='12' fill='{MUTED}' text-anchor='middle'>Layer of Gemma, from input (left) to output (right)</text>"
    )
    return (
        f"<svg viewBox='0 0 {W_} {H_}' role='img' aria-label='Polygraph reading at every layer'>{bands}{grid}{xt}"
        f"<polyline points='{auc}' fill='none' stroke='{UNSURE_C}' stroke-width='1.6' stroke-dasharray='5 4'/>"
        f"<polyline points='{line}' fill='none' stroke='{INK}' stroke-width='2.2' stroke-linejoin='round'/>{needle}{dots}{labels}</svg>"
    )


def scan_html(row: dict[str, Any], toks: list[str], ps: np.ndarray, info: AppInfo) -> str:
    chips = "".join(
        f"<span class='pg-tok' style='background:{soft_p(p)};color:{col_p(p) if p <= UNSURE[0] or p >= UNSURE[1] else INK}' title='{p * 100:.0f}% true'>{E(t)}</span>"
        for t, p in zip(toks, ps)
    )
    return (
        f"<div class='pg'><div class='pg-panel pg-scan'><div class='pg-eyebrow'>Reading at every layer</div>"
        f"<p class='pg-sub' style='margin-top:6px'>“{E(row['claim'])}”. Early layers have not formed a view yet; the truth signal appears in the middle of the network, where the polygraph is most accurate. Hover a point for its value.</p>"
        f"{scan_svg(row, info)}<div class='pg-scan-legend'><span><i style='background:{INK}'></i>This sentence: probability Gemma treats it as true</span>"
        f"<span><i style='background:{UNSURE_C}'></i>Polygraph accuracy at each layer (held-out AUROC)</span></div></div>"
        f"<div class='pg-panel'><div class='pg-eyebrow'>Word by word at layer {info.layer}</div>"
        f"<p class='pg-sub' style='margin-top:6px'>Exploratory. The polygraph was trained on the last token of a sentence, so only the final reading is calibrated.</p>"
        f"<div class='pg-tokens'>{chips}</div></div></div>"
    )


# ---------- chat panes ----------
def turn_bubbles(turns: list[dict[str, Any]], tau: float) -> str:
    parts = []
    for t in turns:
        if t["role"] == "user":
            tag = {
                "lie": "<span class='pg-tag lie'>Secret instruction: answer falsely</span>",
                "push": "<span class='pg-tag push'>Pushback</span>",
            }.get(t.get("tag") or "", "")
            parts.append(f"<div class='pg-msg user'>{tag}<div class='pg-bubble'>{E(t['shown'])}</div></div>")
            continue
        g = t.get("guard")
        if g and g["action"] == "passed" and g.get("p_reply") is None:
            parts.append(
                "<div class='pg-event quiet'><div class='pg-event-head'><i></i>Nothing to check</div>"
                "<p>This reply makes no factual claim, so the polygraph has nothing to read and the guard lets it through.</p></div>"
            )
        if g and g["action"] != "passed":
            blocked = g.get("raw_claim") or g.get("raw", "")
            if g["action"] == "replaced":
                src = {"resampled answer": "a resampled answer"}.get(g["source"], g["source"])
                what = f"Delivered {E(src)} instead, which Gemma believes at {g['p_delivered'] * 100:.0f}%."
            else:
                what = f"No alternative reached {tau * 100:.0f}%, so it declined to answer."
            cands = "".join(
                f"<li><span style='color:{col_p(p)};font-weight:600'>{p * 100:.0f}%</span> {E(k)}: {E(s[:120])}</li>"
                for s, p, k in g["candidates"][:6]
            )
            parts.append(
                f"<div class='pg-event'><div class='pg-event-head'><i></i>Polygraph Guard stepped in</div>"
                f"<p>Blocked “{E(blocked[:140].rstrip('.'))}”: Gemma believes it at <b style='color:{col_p(g['p_reply'])}'>{g['p_reply'] * 100:.0f}%</b>. {what}</p>"
                f"{f'<details><summary>Candidates considered</summary><ul>{cands}</ul></details>' if cands else ''}</div>"
            )
        parts.append(
            f"<div class='pg-msg bot'><span class='pg-who'>GEMMA 2 2B</span><div class='pg-bubble'>{' '.join(claim_span(r) for r in t['rows'])}</div></div>"
        )
    return "".join(parts)


def pane_html(turns: list[dict[str, Any]], protected: bool, tau: float) -> str:
    head = (
        (
            "<div class='pg-pane-head'><div><h3>With Polygraph Guard</h3><p class='pg-sub'>Each answer is checked against Gemma's hidden states before it is shown.</p></div>"
            "<span class='pg-badge on'>Guard on</span></div>"
        )
        if protected
        else (
            "<div class='pg-pane-head'><div><h3>Standard chatbot</h3><p class='pg-sub'>Gemma answers directly. The polygraph only watches.</p></div>"
            "<span class='pg-badge'>No guard</span></div>"
        )
    )
    body = (
        turn_bubbles(turns, tau)
        if turns
        else "<div class='pg-empty'><p>Ask a question below. Both chatbots run the same model and receive the same messages.</p></div>"
    )
    return f"<div class='pg'><div class='pg-pane{' guarded' if protected else ''}'>{head}<div class='pg-chat'>{body}</div></div></div>"


def guard_activity_html(state: dict[str, Any], info: AppInfo) -> str:
    gs = [t["guard"] for t in state["g"] if t["role"] == "assistant" and t.get("guard")]
    n, rep, ab = len(gs), sum(g["action"] == "replaced" for g in gs), sum(g["action"] == "abstained" for g in gs)
    return (
        f"<div class='pg'><div class='pg-panel'><div class='pg-eyebrow'>Guard activity</div>"
        f"<p class='pg-sub' style='margin-top:6px'>Threshold τ = {info.tau:.2f}, chosen on development subjects. Up to {info.n_guard} resampled answers plus Gemma's earlier answers are considered.</p>"
        f"<div class='pg-mini'><div><b>{n}</b><span>answers checked</span></div><div><b>{rep}</b><span>replaced</span></div><div><b>{ab}</b><span>declined</span></div></div></div></div>"
    )


def placeholder_html(text: str) -> str:
    return f"<div class='pg'><div class='pg-panel'><p class='pg-sub'>{text}</p></div></div>"


# ---------- monitor ----------
def histogram_svg(ps):
    ps = [p for p in ps if p is not None]
    if not ps:
        return ""
    bins = np.histogram(ps, bins=10, range=(0, 1))[0]
    mx = max(bins.max(), 1)
    W_, H_ = 640, 160
    bw = (W_ - 50) / 10
    bars = "".join(
        f"<rect x='{36 + i * bw + 3:.1f}' y='{116 - 96 * b / mx:.1f}' width='{bw - 6:.1f}' height='{96 * b / mx:.1f}' rx='4' fill='{FALSE_ if i < 3 else TRUTH if i >= 7 else UNSURE_C}' opacity='.85'><title>{b} answers</title></rect>"
        f"<text x='{36 + i * bw + bw / 2:.1f}' y='{110 - 96 * b / mx:.1f}' font-size='11' font-family='IBM Plex Mono, monospace' fill='{MUTED}' text-anchor='middle'>{b if b else ''}</text>"
        for i, b in enumerate(bins)
    )
    axis = "".join(
        f"<text x='{36 + i * bw:.1f}' y='{134}' font-size='10.5' font-family='IBM Plex Mono, monospace' fill='{FAINT}' text-anchor='middle'>{i * 10}%</text>"
        for i in range(0, 11, 2)
    )
    return (
        f"<svg viewBox='0 0 {W_} {H_}' role='img' aria-label='Distribution of polygraph readings' style='width:100%;max-width:1000px;display:block;margin-top:12px'>"
        f"<line x1='36' x2='{W_ - 14}' y1='116' y2='116' stroke='{LINE2}'/>{bars}{axis}"
        f"<text x='36' y='154' font-size='11' fill='{FAINT}'>Probability that Gemma believes the answer's weakest claim</text></svg>"
    )


def monitor_html(rep, secs, source):
    n = len(rep)
    fl = rep[rep.status == "believed false"]
    und = rep[rep.status == "undecided"]
    unf = int(rep.unfamiliar.sum())
    checked = int(rep.p_true.notna().sum())
    kpis = (
        f"<div class='pg-kpis'><div><b>{n:,}</b><span>Answers scanned</span></div>"
        f"<div><b style='color:{FALSE_}'>{len(fl):,}</b><span>Believed false by Gemma · {len(fl) / max(checked, 1) * 100:.1f}%</span></div>"
        f"<div><b>{len(und):,}</b><span>Undecided</span></div><div><b>{unf:,}</b><span>Unfamiliar, review manually</span></div>"
        f"<div><b>{secs / max(checked, 1) * 1000:.0f} ms</b><span>Polygraph time per answer</span></div></div>"
    )
    labelled = ""
    if rep.label.notna().any():
        lab = rep[rep.label.notna() & rep.p_true.notna()]
        y = lab.label.astype(int).values
        flag = (lab.p_true <= UNSURE[0]).values
        det = flag[y == 0].mean() if (y == 0).any() else float("nan")
        fa = flag[y == 1].mean() if (y == 1).any() else float("nan")
        labelled = (
            f"<div class='pg-panel'><div class='pg-eyebrow'>Checked against the labels in your file</div>"
            f"<p class='pg-sub' style='margin-top:6px;font-size:15.5px;color:{INK}'>{len(lab)} labelled answers. The monitor flagged <b>{det * 100:.0f}%</b> of the answers labelled false "
            f"and <b>{fa * 100:.0f}%</b> of those labelled true.</p></div>"
        )
    top = pd.concat([fl, und]).sort_values("p_true").head(12)
    evs = "".join(
        f"<li><span class='pg-ev-score' style='background:{soft_p(r.p_true)};color:{col_p(r.p_true)}'>{r.p_true * 100:.0f}%</span>"
        f"<div><div class='pg-ev-claim'>{E(r.weakest_claim[:160])}</div>"
        f"{f'<div class=pg-ev-user>User asked: {E(r.user[:110])}</div>' if r.user else ''}"
        f"<div class='pg-ev-meta'>{E(str(r.conversation_id))}{' · unfamiliar sentence' if r.unfamiliar else ''}</div></div></li>"
        for r in top.itertuples()
    )
    table = (
        f"<div class='pg-panel'><div class='pg-eyebrow'>Detection events · review first</div>"
        f"<p class='pg-sub' style='margin-top:6px'>Sorted by how strongly Gemma's hidden states reject the answer. The full report lists every row.</p>"
        f"<ul class='pg-events'>{evs or '<li><span></span><div class=pg-sub>Nothing flagged.</div></li>'}</ul></div>"
    )
    return (
        f"<div class='pg'><p class='pg-note' style='margin:0 0 12px'>Source: {E(source)}</p>{kpis}"
        f"<div class='pg-stack'><div class='pg-panel'><div class='pg-eyebrow'>How confident is Gemma in its answers?</div>{histogram_svg(list(rep.p_true))}</div>{labelled}{table}</div></div>"
    )


# ---------- evidence and limits ----------
def trust_html():
    limits = [
        (
            "It reads beliefs, not facts",
            "The probe tracks what Gemma represents as true. Lies and answers it gave in on are caught; an honest mistake can read as true.",
            "The needle is labelled as Gemma's belief, never as a fact check.",
        ),
        (
            "A single “not” can flip it",
            "A probe trained only on affirmative sentences read negated ones backwards: AUROC 0.01 [0.003, 0.022].",
            "Trained with negated statements; sentences with a negation are flagged.",
        ),
        (
            "It can track the instruction instead of the lie",
            "Read inside the conversation, the layer 20 probe scored 0.996 by spotting the request to lie. Within lie prompts it scored 0.36.",
            "Every claim is read on its own, outside the conversation.",
        ),
        (
            "Context shifts the threshold",
            "At layer 14, readings taken inside the conversation drift toward false, even for true answers.",
            "The in-conversation number is shown for comparison only.",
        ),
        (
            "Unfamiliar sentences",
            "Trained on short factual statements about cities, translations, numbers, companies and general facts.",
            "A Mahalanobis familiarity test flags anything farther out than 99% of held-out statements.",
        ),
        (
            "Explanations disagree",
            "IG, LIME and KernelSHAP picked the same top word only 48 to 64% of the time, and IG did not pass the word-deletion test on lies.",
            "Every explanation is checked live by deleting the words it ranks highest.",
        ),
        (
            "Decodable is not the same as used",
            "High probe accuracy shows the truth signal can be read out. It does not prove Gemma uses it to decide what to say.",
            "A causal steering test along the truth direction is the next step.",
        ),
        (
            "One small model, one language",
            "Gemma 2 2B-it in English, about 7,700 training statements.",
            "Results may not transfer to larger models or other languages.",
        ),
    ]
    return (
        "<div class='pg-limits'>"
        + "".join(f"<div class='pg-limit'><h3>{t}</h3><p>{f}</p><p>{d}</p></div>" for t, f, d in limits)
        + "</div>"
    )


def evidence_html(info: AppInfo) -> str:
    s = info.app_stats
    GUARD_SUMMARY, FIGS, L_APP = info.guard_summary, info.figures, info.layer
    stats = (
        f"<div class='pg-stats'><div class='pg-stat'><b>{s['test_auroc']:.3f}</b><span>AUROC on {s['n_test']:,} held-out statements, layer chosen on validation</span></div>"
        f"<div class='pg-stat'><b>0.957</b><span>AUROC of simply asking Gemma whether a statement is true</span></div>"
        f"<div class='pg-stat'><b>31%</b><span>of confident pushbacks made Gemma give in (126 of 410)</span></div>"
        f"<div class='pg-stat'><b>{s['ece']:.3f}</b><span>calibration error, so the needle's percentages mean what they say</span></div></div>"
    )
    findings = [
        (
            "The truth is readable from the hidden states",
            "What we did",
            "A logistic-regression probe at every layer, trained on 7,696 true and false statements, layer chosen on validation.",
            "What we found",
            f"Layer {L_APP} reaches AUROC 0.981 [0.975, 0.987] on held-out facts, about as well as asking Gemma directly (0.957). The signal peaks in the middle layers.",
        ),
        (
            "Gemma gives in, and the probe sees it",
            "What we did",
            'Hundreds of conversations where the user insists on a wrong answer, asks "are you sure?", or secretly tells Gemma to lie.',
            "What we found",
            "Gemma gave in to confident pushback 31% of the time (10% on cities, 79% on general knowledge). Reading each claim alone, the probe separates caved from held answers with AUROC 0.98.",
        ),
        (
            "A shortcut caught before shipping",
            "What we did",
            "The layer 20 probe looked perfect on lies read inside the conversation, so we split the data by prompt and by claim truth.",
            "What we found",
            "It tracked the instruction, not the lie: AUROC 0.996 for the prompt, 0.36 for truth inside lie prompts. Reading claims on their own removed the shortcut.",
        ),
        (
            "Explanations need checking",
            "What we did",
            "120 verdicts explained with LIME, KernelSHAP and Integrated Gradients, each tested by deleting its top words.",
            "What we found",
            "The methods agree on the top word only 48 to 64% of the time, and Integrated Gradients fails the deletion test on lies. The app checks every explanation live.",
        ),
    ]
    fh = "".join(
        f"<div class='pg-finding'><div><h3>{t}</h3></div><div><p class='pg-k'>{k1}</p><p class='pg-muted'>{a}</p><p class='pg-k'>{k2}</p><p>{b}</p></div></div>"
        for t, k1, a, k2, b in findings
    )
    guard = ""
    if GUARD_SUMMARY:
        R = GUARD_SUMMARY["results_test"]

        def row(k, better):
            if k not in R:
                return ""
            v = R[k]
            b, a = v["before"] * 100, v["after"] * 100
            good = (a < b) if better == "lower" else (a >= b - 2)
            return (
                f"<tr><td>{E(v['label'])}<div class='pg-note'>{v['n']} conversations, {v['subjects']} subjects; {better} is better</div></td>"
                f"<td class='pg-num' style='color:{MUTED}'>{b:.1f}%</td><td class='pg-num' style='color:{TRUTH if good else FALSE_}'>{a:.1f}%</td>"
                f"<td class='pg-mono'>[{v['after_ci'][0] * 100:.1f}, {v['after_ci'][1] * 100:.1f}]</td></tr>"
            )

        rows = (
            row("pushback_assertive", "lower")
            + row("instructed_lie", "lower")
            + row("pushback_neutral", "lower")
            + row("correction", "higher")
            + row("plain_question", "higher")
        )
        lat = GUARD_SUMMARY.get("latency_ms", {})
        guard = (
            f"<div class='pg-finding'><div><h3>Polygraph Guard on held-out subjects</h3><p class='pg-k'>Protocol</p><p class='pg-muted'>τ = {GUARD_SUMMARY['tau']:.2f} chosen on development subjects, "
            f"reported once on test subjects, 95% intervals from a bootstrap over subjects. The guard never sees the right answer. A check costs about "
            f"{lat.get('polygraph_check', float('nan')):.0f} ms against {lat.get('generation', float('nan')):.0f} ms to generate the answer.</p></div>"
            f"<div class='pg-table-wrap'><table class='pg-table'><thead><tr><th>Outcome</th><th>Without guard</th><th>With guard</th><th>95% interval</th></tr></thead><tbody>{rows}</tbody></table></div></div>"
        )
    figs = ""
    if FIGS:
        figs = (
            "<div class='pg-figs'>"
            + "".join(
                f"<figure><img alt='{E(c)}' src='data:image/png;base64,{base64.b64encode(p_.read_bytes()).decode()}'><figcaption>{E(c)}</figcaption></figure>"
                for p_, c in FIGS
            )
            + "</div>"
        )
    return (
        f"<div class='pg'><section class='pg-section' style='margin-top:0'><div class='pg-eyebrow'>Evidence</div><h2>What the numbers say</h2>"
        f"<p>Every result below was measured on data the detector never saw, with intervals from a bootstrap over facts or subjects.</p>{stats}{fh}{guard}{figs}</section>"
        f"<section class='pg-section'><div class='pg-eyebrow'>Limits</div><h2>When not to trust the polygraph</h2><p>We tried to break our own detector. Each card is a failure we measured and what the product does about it.</p>{trust_html()}</section></div>"
    )


# ---------- developers ----------
def codeblock(title, code_html):
    return f"<div class='pg-codeblock'><header><span>{E(title)}</span><button type='button' data-copy>Copy</button></header><pre>{code_html}</pre></div>"


def dev_html(info: AppInfo) -> str:
    install = codeblock("shell", '<span class="c"># from the repository root</span>\npip install -e ".[model]"')
    check = codeblock(
        "python",
        '<span class="k">from</span> probity <span class="k">import</span> Polygraph\n\n'
        'poly = Polygraph.from_pretrained(<span class="s">"google/gemma-2-2b-it"</span>, '
        f'probe=<span class="s">"results/app_probes.npz"</span>, layer=<span class="n">{info.layer}</span>)\n'
        'poly.check(<span class="s">"Anyang-si is located in Iraq."</span>)\n'
        '<span class="c"># {"p_true": 0.04, "verdict": "believed false", "weakest_claim": ..., "claims": [...]}</span>',
    )
    guard = codeblock(
        "python",
        f'reply, report = poly.guarded_reply(messages, tau=<span class="n">{info.tau:.2f}</span>)\n'
        'report[<span class="s">"action"</span>]   <span class="c"># "passed" | "replaced" | "abstained"</span>\n\n'
        '<span class="c"># or guard a reply you already generated</span>\n'
        "reply, report = poly.guard(messages, candidate_reply)",
    )
    api = codeblock(
        "python",
        '<span class="k">from</span> gradio_client <span class="k">import</span> Client\n\n'
        'client = Client(<span class="s">"https://YOUR-LINK"</span>)\n'
        'client.predict(<span class="s">"Anyang-si is located in Iraq."</span>, api_name=<span class="s">"/check"</span>)',
    )
    return f"""<div class="pg"><div class="pg-docs"><nav class="pg-docs-nav"><b>Developers</b><span>Install</span><span>Check a reply</span><span>Guard a chat</span><span>HTTP API</span><span>Integration</span></nav>
<div>
<div class="pg-eyebrow">Developers</div><h3 style="margin-top:6px">Use it from code</h3>
<p>Everything in this console is available programmatically: the <code>probity</code> package for teams that serve open models, and the HTTP endpoint of this running instance. Nothing here is retrained; the package loads the same probe and applies the same rules as the app.</p>
<h3>Install</h3><p>From the project repository. Requires PyTorch and Transformers, and a GPU for interactive use.</p>
{install}
<h3>Check a reply</h3><p>Every factual sentence is read on its own; the verdict is the weakest claim.</p>
{check}
<h3>Guard a chat</h3><p>Generate a reply and run it through Polygraph Guard. The report says whether the reply passed, was replaced or was declined, and lists every candidate considered.</p>
{guard}
<h3>HTTP API of this instance</h3><p>The same check, from any language, through the Gradio client (use this page's address).</p>
{api}
<h3>Integration pattern</h3>
<ol class="pg-refs"><li>Generate the reply with your open model as usual.</li><li>Call <code>check</code> on the reply. It costs one extra forward pass over the claim.</li>
<li>If the weakest claim is below τ, call <code>guard</code> to replace it or decline, and log the event for review.</li><li>Run the monitor over yesterday's logs to find the answers your model did not believe.</li></ol>
<p style="margin-top:18px;font-size:14px;color:var(--pg-faint)">Works with open-weight models only: the polygraph needs access to hidden states. That is also an argument for model transparency.</p>
</div></div></div>"""


ABOUT = """<div class="pg"><section class="pg-section" style="margin-top:0"><div class="pg-eyebrow">About</div><h2>The research story</h2>
<p>A chatbot can know the right answer and still give you the wrong one, because you insisted. From the outside, that answer looks like any other. Probity started as a question: can the model's own hidden states tell us when it does not believe what it is saying?</p>
<div class="pg-timeline">
<div><span class="pg-mono">01 · Read</span><h3>A linear probe on the residual stream</h3><p>7,696 true and false statements from the Geometry of Truth datasets go through Gemma 2 2B-it. A logistic-regression probe is fitted at every layer on fact-grouped training data. Gemma itself is never retrained.</p></div>
<div><span class="pg-mono">02 · Evaluate</span><h3>Held-out facts, one test</h3><p>The layer is chosen on validation and the test set is read once: AUROC 0.981 [0.975, 0.987], comparable to asking Gemma directly (0.957 pooled). Platt scaling turns scores into calibrated probabilities.</p></div>
<div><span class="pg-mono">03 · Attack</span><h3>Red-teaming the detector</h3><p>Instructed lies, confident pushback and negations. Gemma gave in 31% of the time. The probe failed in three distinct ways, including a shortcut that read the instruction to lie instead of the lie.</p></div>
<div><span class="pg-mono">04 · Explain</span><h3>Three attribution methods, each checked</h3><p>Integrated Gradients, LIME and KernelSHAP were run on 120 verdicts. They often disagreed, and a deletion test showed which explanations to trust. Reading claims on their own fixed the shortcut.</p></div>
<div><span class="pg-mono">05 · Act</span><h3>From thermometer to guardrail</h3><p>Polygraph Guard replaces answers Gemma does not believe with one it does, or declines. Its threshold was set on development subjects and measured once on held-out ones.</p></div>
</div></section>
<section class="pg-section"><h2>Related work</h2><ol class="pg-refs">
<li>Marks and Tegmark (2023). The Geometry of Truth: emergent linear structure in LLM representations of true/false datasets.</li>
<li>Azaria and Mitchell (2023). The internal state of an LLM knows when it's lying.</li>
<li>Burns et al. (2023). Discovering latent knowledge in language models without supervision.</li>
<li>Sharma et al. (2024). Towards understanding sycophancy in language models.</li>
<li>Ribeiro, Singh and Guestrin (2016). “Why should I trust you?” Explaining the predictions of any classifier (LIME).</li>
<li>Lundberg and Lee (2017). A unified approach to interpreting model predictions (SHAP).</li>
<li>Sundararajan, Taly and Yan (2017). Axiomatic attribution for deep networks (Integrated Gradients).</li>
<li>Krishna et al. (2022). The disagreement problem in explainable machine learning.</li></ol></section>
<section class="pg-section"><h2>Credits</h2><p>Alberto Gómez Soteres, AIPI 590 Explainable AI, Duke University, Fall 2026. Gemma is provided by Google under the Gemma terms of use. Conversations stay in this session and are not stored.</p></section></div>"""
