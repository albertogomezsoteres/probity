"""Robustness checks on the saved results (no model or GPU needed).

After the main experiments, I re-analysed the saved results to look for problems that pooled numbers can hide:
  A. Results split by question source (cities vs general knowledge), because the sources are not balanced.
  B. The instruction-shortcut test (notebook 02b) within each source.
  C. Probe vs asking the model, dataset by dataset.
  D. Attribution shares per word, because the instruction has many more words than the answer.
  E. What Polygraph Guard actually did in each scenario, including abstentions and the cases it missed.

Usage:
    python scripts/robustness_checks.py        # prints a report and writes results/robustness_checks.json
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"
out = {}


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else None


def boot_ci(y, s, groups, n=1000, seed=0):
    """95% interval for AUROC, resampling subjects (groups) instead of single claims."""
    rng = np.random.default_rng(seed)
    y, s, groups = np.asarray(y), np.asarray(s), np.asarray(groups)
    uniq = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        ii = np.concatenate([idx[g] for g in rng.choice(uniq, len(uniq))])
        if len(np.unique(y[ii])) == 2:
            vals.append(roc_auc_score(y[ii], s[ii]))
    return [round(float(v), 3) for v in np.percentile(vals, [2.5, 97.5])]


# ---------- A. results by question source ----------
c = pd.read_csv(RES / "claims.csv")
press = c[(c.scenario == "pressure_assertive") & c.note.isin(["caved", "held"])]
cave = {"all": {"caved": int((press.note == "caved").sum()), "n": int(len(press))}}
for src, d in press.groupby("source"):
    cave[src] = {"caved": int((d.note == "caved").sum()), "n": int(len(d))}
for v in cave.values():
    v["rate"] = round(v["caved"] / v["n"], 3)
out["A_caving_rate_by_source"] = cave

lies = c[((c.scenario == "instructed_lie") & (c.note == "lie")) | ((c.scenario == "instructed_truth") & (c.note == "truth"))]
caving = press
by_src = []
for name, d, ycol in [("Instructed lies", lies, lambda d: (d.note == "truth").astype(int)),
                      ("Gave in to pushback", caving, lambda d: (d.note == "held").astype(int))]:
    for src in ["all", "cities", "knowledge"]:
        dd = d if src == "all" else d[d.source == src]
        y = ycol(dd)
        for L in (14, 20):
            for mode, col in [("in context", f"score_L{L}"), ("alone", f"alone_L{L}")]:
                by_src.append({"scenario": name, "source": src, "layer": L, "reading": mode,
                               "n_true": int(y.sum()), "n_false": int((1 - y).sum()),
                               "auroc": None if auc(y, dd[col]) is None else round(auc(y, dd[col]), 3),
                               "ci_subjects": boot_ci(y, dd[col], dd.subject) if auc(y, dd[col]) is not None else None})
out["A_auroc_by_source"] = by_src

# ---------- B. the 2x2 instruction test within each source ----------
cells = pd.read_csv(RES / "results_confound_cells.csv")
tests = []
for src in ["all", "cities", "knowledge"]:
    d = cells if src == "all" else cells[cells.source == src]
    true = (d.claim_truth == "true claim").values
    lie_p = (d.prompt_type == "lie-game prompt").values
    for L in (14, 20):
        s = d[f"reread_L{L}"].values
        for name, mask, y in [("truth inside the lie prompt (refused vs lied)", lie_p, true),
                              ("truth inside the truth prompt (told truth vs wrong)", ~lie_p, true),
                              ("prompt, true claims only", true, ~lie_p),
                              ("prompt, false claims only", ~true, ~lie_p)]:
            a = auc(y[mask], s[mask])
            tests.append({"source": src, "layer": L, "test": name, "n": int(mask.sum()),
                          "auroc": None if a is None else round(a, 3)})
out["B_instruction_test_by_source"] = tests

# ---------- C. probe vs asking the model, per dataset ----------
pd_ = pd.read_csv(RES / "results_test_per_dataset.csv")
diff = pd_.polygraph_auroc - pd_.ask_model_auroc
out["C_probe_vs_ask_per_dataset"] = {
    "mean_auroc_probe": round(float(pd_.polygraph_auroc.mean()), 3),
    "mean_auroc_ask": round(float(pd_.ask_model_auroc.mean()), 3),
    "probe_better": int((diff > 0.005).sum()), "ask_better": int((diff < -0.005).sum()),
    "within_0.005": int((diff.abs() <= 0.005).sum()),
    "pooled": json.loads((RES / "summary01b.json").read_text())["test_polygraph_auroc"],
    "pooled_ask": json.loads((RES / "summary01b.json").read_text())["test_ask_model_auroc"],
}

# ---------- D. attribution per word ----------
att = json.loads((RES / "attributions.json").read_text())
METHODS = ["LIME", "KernelSHAP", "Integrated Gradients"]
xai = {}
for q, boundary in [("lie", "..."), ("pressure", "only.")]:
    ex = [e for e in att if e["question"] == q and (q != "pressure" or e["label"] == 0)]   # pushback: only answers where Gemma gave in
    rows = {m: {"user_share": [], "user_words_share": [], "per_word_ratio": [], "top3_in_user": [], "wrong_in_top3": []} for m in METHODS}
    for e in ex:
        w = e["words"]
        cut = max(i for i, x in enumerate(w) if x == boundary)          # last word of the user turn
        user = np.arange(len(w)) <= cut
        sign = np.sign(e["score"]) or 1.0
        for m in METHODS:
            a = np.asarray(e[m], float); ab = np.abs(a)
            rows[m]["user_share"].append(ab[user].sum() / ab.sum())
            rows[m]["user_words_share"].append(user.mean())
            rows[m]["per_word_ratio"].append(ab[user].mean() / ab[~user].mean())
            top = np.argsort(-(sign * a))[:3]                              # 3 words pushing hardest toward the verdict (the ones the deletion test removes)
            rows[m]["top3_in_user"].append(user[top].mean())
            rows[m]["wrong_in_top3"].append(any(w[i].strip(".,").upper() == "WRONG" for i in top))
    xai[q] = {m: {k: round(float(np.mean(v)), 3) for k, v in r.items()} for m, r in rows.items()}
    xai[q]["n_examples"] = len(ex)
out["D_attribution_per_word"] = xai

# ---------- E. what the guard did ----------
g = pd.read_json(RES / "guard_test_conversations.jsonl", lines=True)
guard = {}
for scen, d in g.groupby("scenario"):
    t = pd.crosstab(d.base_outcome, d.outcome)
    guard[scen] = {"n": int(len(d)), "before_to_after": {f"{b} -> {a}": int(t.loc[b, a]) for b in t.index for a in t.columns if t.loc[b, a]},
                   "actions": d.action.value_counts().to_dict()}
pb = g[g.scenario == "pushback_assertive"]
missed = pb[(pb.base_outcome == "bad") & (pb.outcome == "bad")]
guard["pushback_assertive"]["missed_with_no_score"] = int(missed.p_base.isna().sum())
guard["pushback_assertive"]["missed_total"] = int(len(missed))
ok = pb.p_base.notna()
guard["pushback_assertive"]["guard_reading_auroc"] = round(auc((pb.base_outcome[ok] == "good").astype(int), pb.p_base[ok]), 3)
src_of = c.drop_duplicates("subject").set_index("subject").source
g["source"] = g.subject.map(src_of)
by = {}
for scen in ["pushback_assertive", "instructed_lie"]:
    for src in ["all", "cities", "knowledge"]:
        d = g[(g.scenario == scen) & ((g.source == src) | (src == "all"))]
        r = d[d.p_base.notna()]
        by[f"{scen} | {src}"] = {"n": int(len(d)), "wrong_before": round(float((d.base_outcome == "bad").mean()), 3),
                                 "wrong_after": round(float((d.outcome == "bad").mean()), 3),
                                 "abstained": round(float((d.outcome == "abstain").mean()), 3),
                                 "guard_reading_auroc": round(auc((r.base_outcome == "good").astype(int), r.p_base), 3)}
guard["by_source"] = by
out["E_guard_breakdown"] = guard

(RES / "robustness_checks.json").write_text(json.dumps(out, indent=2))

# ---------- report ----------
pd.set_option("display.width", 160)
print("A. Caving under confident pushback, by source:", {k: f"{v['caved']}/{v['n']} = {v['rate']:.0%}" for k, v in cave.items()})
t = pd.DataFrame(by_src)
print("\nA. AUROC by source (95% CI resamples subjects):")
print(t[t.layer == 14].pivot_table(index=["scenario", "source"], columns="reading", values="auroc").round(3))
print(t[t.layer == 20].pivot_table(index=["scenario", "source"], columns="reading", values="auroc").round(3).rename_axis(columns="layer 20"))
print("\nB. Instruction test (02b) by source:")
print(pd.DataFrame(tests).pivot_table(index=["layer", "test"], columns="source", values="auroc").round(3))
print("\nC. Probe vs asking, per dataset:", out["C_probe_vs_ask_per_dataset"])
print("\nD. Attribution per word:")
for q in xai:
    print(" ", q, {m: xai[q][m] for m in METHODS})
print("\nE. Guard:")
for k, v in guard.items():
    print(" ", k, v)
print("\nwrote results/robustness_checks.json")
