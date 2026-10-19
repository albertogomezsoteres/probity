"""Make the final figures for the README and the poster from the saved results.

Usage:
    python scripts/make_figures.py

Reads only files in results/ (written by the notebooks) and writes PNG and SVG files to figures/.
No model is needed, so this runs on any laptop in a few seconds.
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines
import matplotlib.patches
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

# ---------- style ----------
for f in (ROOT / "assets" / "fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
INK, MUTED, FAINT, GRID, LINE = "#15212C", "#56636F", "#8994A0", "#E7EBEE", "#CDD4DA"
TRUE_C, FALSE_C, NEUTRAL, SOFT_T, SOFT_F = "#1F5FBF", "#D2452B", "#9AA5B1", "#DCE7F7", "#F8DDD7"
plt.rcParams.update({
    "font.family": "IBM Plex Sans", "font.size": 12, "text.color": INK,
    "axes.titlesize": 14, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.titlepad": 14,
    "axes.labelcolor": MUTED, "axes.labelsize": 12, "axes.edgecolor": LINE, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.frameon": False, "legend.fontsize": 11, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "svg.fonttype": "path", "svg.hashsalt": "probity",
})

def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)
    print("wrote", name)

def load(name):
    return json.loads((RES / name).read_text())

s01, s01b, s02b, s03, s05 = (load(f) for f in ["summary.json", "summary01b.json", "summary02b.json", "summary03.json", "summary05.json"])
L = s01b["layer_selected_on_validation"]


# ---------- Figure 1: where the truth signal lives ----------
def fig_layers():
    lay = pd.read_csv(RES / "results_layers.csv")
    lay = lay[lay.format == "assistant"]
    probe = lay[lay.method == "probe"].set_index("layer").auroc
    shuf = lay[lay.method == "shuffled control"].set_index("layer").auroc
    lodo = pd.read_csv(RES / "results_lodo.csv").groupby("layer").auroc.mean()
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.axhline(s01b["test_ask_model_auroc"], color=NEUTRAL, lw=1.5, ls=(0, (5, 4)))
    ax.text(0.3, s01b["test_ask_model_auroc"] + 0.012, f"Asking Gemma \"true or false?\"  {s01b['test_ask_model_auroc']:.3f}",
            ha="left", va="bottom", color=MUTED, fontsize=11)
    ax.plot(shuf.index, shuf.values, color=LINE, lw=2, label="Same probe, labels shuffled (control)")
    ax.plot(lodo.index, lodo.values, color=FALSE_C, lw=2, alpha=.85, label="Probe on a dataset it never saw (mean of 9)")
    ax.plot(probe.index, probe.values, color=TRUE_C, lw=2.6, label="Probe on held-out statements")
    lo, hi = s01b["test_polygraph_ci"]
    ax.errorbar([L], [s01b["test_polygraph_auroc"]], yerr=[[s01b["test_polygraph_auroc"] - lo], [hi - s01b["test_polygraph_auroc"]]],
                fmt="o", color=INK, ms=8, capsize=4, zorder=5)
    ax.text(L + 0.4, s01b["test_polygraph_auroc"] + 0.014,
            f"Layer {L}, chosen on validation\ntest AUROC {s01b['test_polygraph_auroc']:.3f} [{lo:.3f}, {hi:.3f}]",
            ha="left", va="bottom", fontsize=11, color=INK)
    ax.set_xlim(-0.5, probe.index.max() + 0.5); ax.set_ylim(min(0.4, float(lodo.min()) - 0.03), 1.11)
    ax.set_xlabel("Layer of Gemma 2 2B-it (0 = embeddings)"); ax.set_ylabel("AUROC (0.5 = chance)")
    ax.set_title("The truth signal is readable in the middle of the network")
    ax.legend(loc="lower right", bbox_to_anchor=(1, 0.02))
    save(fig, "fig1_layer_profile")


# ---------- Figure 2: negation ----------
def fig_negation():
    a = pd.read_csv(RES / "results_autopsy.csv")
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for probe, color, label in [("same-domain affirmative", FALSE_C, "Trained without negations"), ("full polygraph", TRUE_C, "Trained with negations (final probe)")]:
        d = a[(a.probe == probe) & (a.domain == "translations")].sort_values("layer")
        ax.plot(d.layer, d.auroc_neg, color=color, lw=2.4, label=label)
    ax.axhline(0.5, color=NEUTRAL, lw=1.2, ls=(0, (5, 4)))
    ax.text(8.5, 0.515, "chance", color=MUTED, fontsize=10.5)
    aff = [r for r in s01b["autopsy_at_layer"] if r["probe"] == "same-domain affirmative" and r["domain"] == "translations"][0]
    ax.annotate(f"AUROC {aff['auroc_neg']:.2f} [{aff['auroc_neg_lo']:.3f}, {aff['auroc_neg_hi']:.3f}]\nit reads negations backwards",
                xy=(L, aff["auroc_neg"]), xytext=(6.5, 0.24), fontsize=11, color=FALSE_C, arrowprops=dict(arrowstyle="-", color=FALSE_C, lw=1))
    ax.set_ylim(-0.03, 1.05); ax.set_xlabel("Layer"); ax.set_ylabel("AUROC on negated translations")
    ax.set_title("A single \"not\" fools a probe that never saw one")
    ax.legend(loc="center right")
    save(fig, "fig2_negation")


# ---------- Figure 3: the instruction shortcut ----------
def fig_shortcut():
    # Notebook 02b re-reads every game answer inside its original conversation (column reread_L20).
    c = pd.read_csv(RES / "results_confound_cells.csv")
    order = [("truth-game prompt", "true claim", "Asked for the truth,\ntold the truth"),
             ("truth-game prompt", "false claim", "Asked for the truth,\ngot it wrong"),
             ("lie-game prompt", "true claim", "Asked to lie,\ntold the truth anyway"),
             ("lie-game prompt", "false claim", "Asked to lie,\nlied")]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    rng = np.random.default_rng(0)
    for i, (pt, ct, _) in enumerate(order):
        v = c[(c.prompt_type == pt) & (c.claim_truth == ct)]["reread_L20"].dropna().values
        y = len(order) - 1 - i
        if pt == "lie-game prompt":
            ax.axhspan(y - 0.45, y + 0.45, color="#F4F6F8", zorder=0, lw=0)
        ax.scatter(v, y + rng.uniform(-0.2, 0.2, len(v)), s=12, alpha=.5, color=TRUE_C if ct == "true claim" else FALSE_C, lw=0, zorder=2)
        ax.plot([np.median(v)] * 2, [y - 0.32, y + 0.32], color=INK, lw=2.6, zorder=3)
        ax.text(1.01, y, f"n = {len(v)}", transform=ax.get_yaxis_transform(), va="center", fontsize=10, color=MUTED)
    ax.axvline(0, color=NEUTRAL, lw=1.2, ls=(0, (5, 4)))
    ax.set_yticks(range(len(order))); ax.set_yticklabels([o[2] for o in order][::-1], fontsize=11)
    ax.set_xlabel("Layer 20 probe reading inside the conversation  (left = false, right = true)")
    ax.grid(axis="y", visible=False)
    t20 = {r["test"]: r for r in s02b["tests"] if r["layer"] == 20}
    tl = t20["truth, within lie-game prompt (refused vs. lied)"]; pr = t20["PROMPT, true claims only (truth game vs. refused to lie)"]
    ax.set_title("Inside the conversation, layer 20 follows the instruction more than the truth")
    fig.text(0.0, -0.17, f"Grey rows: the user asked Gemma to lie. Both rows read as false, even when Gemma told the truth.\n"
             f"Telling the prompts apart: AUROC {pr['auroc']:.2f}. Telling truth from lies inside lie prompts: AUROC {tl['auroc']:.2f} (worse than chance).\n"
             f"Blue = true claim, red = false claim, black line = median.", fontsize=10.5, color=MUTED, transform=ax.transAxes, va="top")
    save(fig, "fig3_shortcut")


# ---------- Figure 4: in context vs. alone ----------
def fig_context():
    oc = pd.DataFrame(s02b["out_of_context"])
    oc["scenario"] = oc.scenario.replace({"Caved to \"I'm sure it's X\"": "Gave in to pushback"})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    for ax, col, title in [(axes[0], "auroc_in_context", "Read inside the conversation"), (axes[1], "auroc_alone", "Each claim read on its own")]:
        m = oc.pivot(index="layer", columns="scenario", values=col)[["Instructed lies", "Gave in to pushback"]]
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("q", [FALSE_C, "#F4E6E2", "#E3ECF8", TRUE_C])
        ax.imshow(m.values, cmap=cmap, vmin=0.5, vmax=1.0)
        for (i, j), v in np.ndenumerate(m.values):
            r, g, b, _ = cmap((v - 0.5) / 0.5)
            ink = "white" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else INK
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=14, fontweight="semibold", color=ink)
            if col == "auroc_in_context" and m.index[i] == 20 and m.columns[j] == "Instructed lies":
                ax.text(j, i + 0.22, "shortcut (Figure 3)", ha="center", va="center", fontsize=9.5, color=ink)
        ax.set_xticks(range(2)); ax.set_xticklabels(m.columns, fontsize=11)
        ax.set_yticks(range(len(m.index))); ax.set_yticklabels([f"Layer {x}" for x in m.index], fontsize=11)
        ax.set_title(title, fontsize=13); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.suptitle("Reading each claim on its own fixes both failures (AUROC)", x=0.06, ha="left", fontsize=15, fontweight="semibold", y=1.04)
    save(fig, "fig4_context_fix")


# ---------- Figure 5: what the attribution methods rank highest (XAI) ----------
def fig_xai():
    # For each explained verdict: which part of the input holds the 3 words that push hardest toward the verdict?
    # Compared with the share of words that part has, because the instruction is much longer than the answer.
    att = json.loads((RES / "attributions.json").read_text())
    methods = ["LIME", "KernelSHAP", "Integrated Gradients"]
    panels = [("lie", "...", "Instructed lie, layer 20 (30 verdicts)"), ("pressure", "only.", f"Gave in to pushback, layer {L} (15 verdicts)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9), sharex=True)
    for ax, (q, boundary, title) in zip(axes, panels):
        ex = [e for e in att if e["question"] == q and (q != "pressure" or e["label"] == 0)]   # pushback: answers where Gemma gave in
        words_share, top = [], {m: [] for m in methods}
        for e in ex:
            w = e["words"]; cut = max(i for i, x in enumerate(w) if x == boundary)
            user = np.arange(len(w)) <= cut; sign = np.sign(e["score"]) or 1.0
            words_share.append(user.mean())
            for m in methods:
                top[m].append(user[np.argsort(-(sign * np.asarray(e[m])))[:3]].mean())
        ref = float(np.mean(words_share))
        for i, m in enumerate(methods):
            y = len(methods) - 1 - i; v = float(np.mean(top[m]))
            ax.barh(y, v, color=FALSE_C, height=0.56)
            ax.barh(y, 1 - v, left=v, color=TRUE_C, height=0.56)
            ax.text(v / 2, y, f"{v*100:.0f}%", ha="center", va="center", color="white", fontsize=11.5, fontweight="semibold")
        ax.axvline(ref, color=INK, lw=1.4, ls=(0, (4, 3)))
        ax.text(ref, len(methods) - 0.45, f"{ref*100:.0f}% of the words", ha="center", fontsize=10, color=INK,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))
        ax.set_yticks(range(len(methods))); ax.set_yticklabels(methods[::-1] if ax is axes[0] else [])
        ax.set_axisbelow(True); ax.set_xlim(0, 1); ax.set_ylim(-0.6, len(methods) - 0.2)
        ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(title, fontsize=13); ax.grid(axis="y", visible=False)
    handles = [matplotlib.patches.Patch(color=FALSE_C, label="top words in the user's message"),
               matplotlib.patches.Patch(color=TRUE_C, label="top words in Gemma's claim"),
               matplotlib.lines.Line2D([], [], color=INK, ls=(0, (4, 3)), label="share of words in the user's message")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Where are the 3 words that push hardest toward each verdict? The methods disagree, and none singles out the instruction",
                 x=0.06, ha="left", fontsize=14, fontweight="semibold", y=1.03)
    save(fig, "fig5_xai_attribution")


# ---------- Figure 6: can the explanations be trusted? (deletion test) ----------
def fig_deletion():
    d = s03["deletion_k3_by_method"]
    qs = [("pressure", "Gave in to pushback"), ("negation", "Negated statements"), ("lie", "Instructed lies")]
    ms = [("LIME", TRUE_C), ("KernelSHAP", "#5B8FD9"), ("Integrated Gradients", "#9DB9E6"), ("random words", NEUTRAL)]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    w = 0.19; x = np.arange(len(qs))
    for k, (m, color) in enumerate(ms):
        vals = [d[q][m] for q, _ in qs]
        cols = [FALSE_C if (m == "Integrated Gradients" and v < d[q]["random words"]) else color for (q, _), v in zip(qs, vals)]
        ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=cols, label=m)
        for xi, v in zip(x + (k - 1.5) * w, vals):
            ax.text(xi, v + (0.4 if v >= 0 else -0.4), f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=9.5, color=MUTED)
    ax.axhline(0, color=INK, lw=1)
    ax.set_xticks(x); ax.set_xticklabels([q[1] for q in qs])
    ax.set_ylabel("Change in the probe's score after\ndeleting the top 3 words (higher = faithful)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2)
    ax.set_axisbelow(True)
    ax.annotate("Integrated Gradients does worse than\nrandom deletion on lies", xy=(2 + 0.4 * w, d["lie"]["Integrated Gradients"] - 0.3),
                xytext=(0.85, -6.2), fontsize=10.5, color=FALSE_C, arrowprops=dict(arrowstyle="-", color=FALSE_C, lw=1))
    ax.set_ylim(min(-8.5, d["lie"]["Integrated Gradients"] - 3), max(d["pressure"].values()) + 4)
    ax.set_title("Deletion test: which explanations are faithful? (120 explained verdicts)")
    save(fig, "fig6_deletion_test")


# ---------- Figure 7: Polygraph Guard ----------
def fig_guard():
    R = s05["results_test"]
    rows = [("pushback_assertive", "Wrong answer after confident pushback", "lower"),
            ("instructed_lie", "Lie delivered under a secret instruction", "lower"),
            ("pushback_neutral", "Wrong answer after \"are you sure?\"", "lower"),
            ("correction", "Valid correction accepted (control)", "higher"),
            ("plain_question", "Correct answer still correct (control)", "higher")]
    rows = [r for r in rows if r[0] in R]
    fig, ax = plt.subplots(figsize=(10.4, 0.9 * len(rows) + 1.3))
    for i, (k, label, better) in enumerate(rows):
        v = R[k]; b, a = v["before"] * 100, v["after"] * 100; lo, hi = (x * 100 for x in v["after_ci"])
        y = len(rows) - 1 - i
        good = (a <= b) if better == "lower" else (a >= b - 2)
        c = TRUE_C if good else FALSE_C
        ax.plot([b, a], [y, y], color=LINE, lw=7, solid_capstyle="round", zorder=1)
        ax.plot([lo, hi], [y - 0.24, y - 0.24], color=c, lw=1.6)
        ax.scatter([b], [y], s=110, color=NEUTRAL, zorder=3)
        ax.scatter([a], [y], s=130, color=c, zorder=4)
        same = abs(a - b) < 6
        if k == "instructed_lie" and v.get("abstained"):
            ax.text(a + 7, y + 0.3, f"mostly by abstaining ({v['abstained']*100:.0f}% of answers)", ha="left", va="center", fontsize=10, color=MUTED)
        if same:  # dots overlap, so write one label: before -> after
            left = min(a, b) > 50
            ax.text(min(a, b) - 3 if left else max(a, b) + 3, y, f"{b:.1f}% before, {a:.1f}% after", ha="right" if left else "left",
                    va="center", fontsize=11.5, fontweight="semibold", color=c)
        else:
            ax.text(b, y + 0.3, f"{b:.1f}%", ha="center", fontsize=11, color=MUTED)
            ax.text(a, y + 0.3, f"{a:.1f}%", ha="center", fontsize=11.5, fontweight="semibold", color=c)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{r[1]}\n{r[2]} is better, n = {R[r[0]]['n']}" for r in rows][::-1], fontsize=11)
    ax.set_xlim(-4, 104); ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(-0.7, len(rows) - 0.3); ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=90, color=NEUTRAL, label="Without guard"); ax.scatter([], [], s=90, color=TRUE_C, label="With Polygraph Guard")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    ax.set_title(f"Polygraph Guard on held-out subjects (threshold {s05['tau']:.2f} chosen on development subjects)")
    save(fig, "fig7_guard_results")


# ---------- Figure 8: guard threshold (supplementary) ----------
def fig_threshold():
    sw = pd.read_csv(RES / "guard_threshold_sweep.csv")
    fig, ax = plt.subplots(figsize=(7.8, 3.8))
    ax.plot(sw.tau, sw.caved * 100, color=FALSE_C, lw=2.2, marker="o", label="Wrong answer after pushback")
    ax.plot(sw.tau, sw.rejected_corrections * 100, color=TRUE_C, lw=2.2, marker="o", label="Valid corrections rejected")
    ax.plot(sw.tau, sw.false_alarms * 100, color=NEUTRAL, lw=2.2, marker="o", label="Correct answers changed")
    ax.axvline(s05["tau"], color=INK, lw=1, ls=":")
    ax.text(s05["tau"] - 0.01, ax.get_ylim()[1] * 0.9, f"chosen {s05['tau']:.2f}\n(edge of the grid)", ha="right", fontsize=10.5)
    ax.set_xlabel("Guard threshold (probability that Gemma believes the claim)"); ax.set_ylabel("Development subjects (%)")
    ax.set_title("Choosing the guard threshold on development subjects only")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=10)
    save(fig, "fig8_guard_threshold")


# ---------- Figure 9: medicine and finance (notebook 06) ----------
def fig_high_stakes():
    path = RES / "summary06.json"
    if not path.exists():
        print("skip fig9_high_stakes (results/summary06.json missing)")
        return
    s06 = load("summary06.json")
    tr = s06["transfer"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    groups = [("Original domain\n(test facts)", s01b["test_polygraph_auroc"], s01b["test_ask_model_auroc"]),
              ("Medicine", tr["medicine"]["auroc"], tr["medicine"]["ask_auroc"]),
              ("Finance", tr["finance"]["auroc"], tr["finance"]["ask_auroc"])]
    x = np.arange(len(groups)); w = 0.36
    a1.bar(x - w / 2, [g[1] for g in groups], w, color=TRUE_C, label="Probe (no retraining)")
    a1.bar(x + w / 2, [g[2] for g in groups], w, color=NEUTRAL, label="Asking Gemma")
    for xi, g in zip(x, groups):
        a1.text(xi - w / 2, g[1] + 0.01, f"{g[1]:.2f}", ha="center", fontsize=10.5, color=INK)
        a1.text(xi + w / 2, g[2] + 0.01, f"{g[2]:.2f}", ha="center", fontsize=10.5, color=MUTED)
    a1.set_xticks(x); a1.set_xticklabels([g[0] for g in groups]); a1.set_ylim(0.5, 1.13); a1.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    a1.set_ylabel("AUROC, correct vs wrong statement"); a1.grid(axis="x", visible=False)
    a1.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0)); a1.set_title("The probe transfers only partly")
    R = s06["results"]; X = s06.get("crossfit", {}).get("results", {})
    get = lambda d: d.get("all", d)
    rows = [("pushback_assertive", "Wrong answer after\nconfident pushback", "lower"),
            ("instructed_lie", "Wrong answer under an\ninstruction to lie", "lower"),
            ("correction", "Valid correction\naccepted (control)", "higher"),
            ("plain_question", "Correct answer\nstill correct (control)", "higher")]
    for i, (k, label, better) in enumerate(rows):
        v = get(R[k]); y = len(rows) - 1 - i
        b, a = v["before"] * 100, v["after"] * 100
        a2.plot([b, a], [y, y], color=LINE, lw=6, solid_capstyle="round", zorder=1)
        a2.scatter([b], [y], s=90, color=NEUTRAL, zorder=3)
        a2.scatter([a], [y], s=110, color=TRUE_C, zorder=4)
        if k in X:
            c = X[k]["after"] * 100
            a2.scatter([c], [y], s=90, facecolor="white", edgecolor=TRUE_C, lw=2, zorder=5)
        a2.text(min(a, b) - 2, y + 0.3, f"{b:.0f}% to {a:.0f}%", ha="left", fontsize=10.5, color=INK)
    a2.set_yticks(range(len(rows))); a2.set_yticklabels([r[1] for r in rows][::-1], fontsize=10.5)
    a2.set_xlim(-3, 103); a2.set_xticks([0, 25, 50, 75, 100]); a2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    a2.set_ylim(-0.6, len(rows) - 0.2); a2.grid(axis="y", visible=False)
    a2.scatter([], [], s=80, color=NEUTRAL, label="Without guard")
    a2.scatter([], [], s=90, color=TRUE_C, label=f"Guard, threshold {s06.get('tau_from_nb05', 0.5):.2f} from notebook 05")
    a2.scatter([], [], s=80, facecolor="white", edgecolor=TRUE_C, lw=2, label="Guard, threshold chosen on the other half (cross-fitted)")
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=2, fontsize=10)
    n_items = s06.get("n_items", 82)
    a2.set_title(f"Polygraph Guard on {n_items} medicine and finance questions")
    fig.suptitle(f"Gemma knows {s06['first_answer_correct']*100:.0f}% of these answers, but gives in to pushback {get(R['pushback_assertive'])['before']*100:.0f}% of the time",
                 x=0.06, ha="left", fontsize=15, fontweight="semibold", y=1.04)
    fig.tight_layout()
    save(fig, "fig9_high_stakes")


if __name__ == "__main__":
    fig_layers(); fig_negation(); fig_shortcut(); fig_context(); fig_xai(); fig_deletion(); fig_guard(); fig_threshold(); fig_high_stakes()
