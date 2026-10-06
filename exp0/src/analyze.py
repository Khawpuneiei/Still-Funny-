"""Make the Exp. 0 figures, tables and feasibility numbers for the one-page summary.

    python src/analyze.py --run-id pilot1

Needs: parse.py done; label_categories.py done (for Figure 1 and kappa);
hand_labels.csv / refusal_check.csv filled in (optional; adds kappa and refusal-rule error).
Writes results/<run-id>/fig1_categories.png, fig2_refusals.png, fig3_spearman.png,
spearman.csv, pairwise_summary.csv, normqa_summary.csv, feasibility.csv and summary.md.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, RESULTS, RUNS, load_config

# Categorical palette (fixed order, colorblind-checked).
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
FIG_GROUPS = {  # 12 labeler categories folded into 7 groups for Figure 1
    "pun": "pun / wordplay", "observational": "observational", "dark": "dark", "political": "political",
    "other": "other", "refusal": "refusal",
    **{c: "identity-targeted" for c in ["ethnic", "gender", "sexual_orientation", "disability",
                                         "appearance", "religion", "age"]},
}
GROUP_ORDER = ["pun / wordplay", "observational", "identity-targeted", "dark", "political", "other", "refusal"]


def style(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.set_axisbelow(True)


def kappa(a, b):
    a, b = pd.Series(a).astype(str), pd.Series(b).astype(str)
    cats = sorted(set(a) | set(b))
    po = (a.values == b.values).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def md_table(df):
    df = df.reset_index() if df.index.name else df
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:.3g}" if isinstance(v, float) else str(v))
                                       for v in r) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    args = ap.parse_args()
    out = RESULTS / args.run_id
    cfg = load_config()
    order = [m["name"] for m in sorted(cfg["models"], key=lambda m: (m["family"], m["generation"]))]
    calls = pd.read_csv(out / "calls.csv", low_memory=False)
    order = [m for m in order if m in set(calls["model"])]
    jp = DATA / "pilot_jokes.csv"
    jokes = (pd.read_csv(jp) if jp.exists() else pd.DataFrame(columns=["joke_id", "humor_rating", "offense_rating"])).set_index("joke_id")
    md = [f"# Exp. 0 results: run {args.run_id}", "",
          "Pilot: local models are 4-bit (Q4_K_M) via Ollama; small n; human answer key is HaHackathon "
          "(Meaney et al., 2021); no new human data.", ""]

    # ---------- Figure 1: category distribution of zero-shot jokes ----------
    lab_path = out / "labels.jsonl"
    labels = None
    if lab_path.exists():
        labels = pd.DataFrame([json.loads(l) for l in open(lab_path, encoding="utf-8")])
        labels = labels[labels.get("category").notna()].drop_duplicates(["model", "prompt_id"], keep="last")
        gen = pd.read_csv(out / "generations.csv").merge(labels[["model", "prompt_id", "category", "mechanism"]],
                                                         on=["model", "prompt_id"], how="left",
                                                         suffixes=("_asked", ""))
        z = gen[gen["part"] == "exp1_zero_shot"].copy()
        z.loc[z["refusal_kw"], "category"] = "refusal"
        z["fig_group"] = z["category"].map(FIG_GROUPS).fillna("other")
        dist = pd.crosstab(z["model"], z["fig_group"], normalize="index").reindex(index=order, columns=GROUP_ORDER,
                                                                                    fill_value=0)
        fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.4))
        left = np.zeros(len(order))
        for i, g in enumerate(GROUP_ORDER):
            ax.barh(range(len(order)), dist[g], left=left, color=C[i], height=0.6, label=g,
                    edgecolor="white", linewidth=1.5)
            left += dist[g].values
        ax.set_yticks(range(len(order)), order)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("share of 100 zero-shot jokes")
        ax.set_title("Joke categories when asked for 'a joke', older to newer within each family",
                     fontsize=9, color=INK, loc="left")
        ax.legend(ncol=4, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.18))
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "fig1_categories.png", dpi=200)
        plt.close(fig)
        dist.round(3).to_csv(out / "category_distribution.csv")
        md += ["## Figure 1: zero-shot joke categories (labeler: provisional)", "",
               "![](fig1_categories.png)", "", md_table(dist.round(2)), ""]
    else:
        print("No labels.jsonl yet: run src/label_categories.py for Figure 1.")

    # ---------- Figure 2: refusal rate per identity-swap side ----------
    gen_all = pd.read_csv(out / "generations.csv")
    sw = gen_all[gen_all["part"] == "exp1_identity_swap"]
    if not sw.empty:
        rr = sw.groupby(["model", "axis", "group_name"])["refusal_kw"].mean().reset_index()
        axes_names = [a for a in ["gender", "religion", "race"] if a in set(sw["axis"])]
        axes_names += [a for a in dict.fromkeys(sw["axis"]) if a not in axes_names]
        fig, axs = plt.subplots(1, len(axes_names), figsize=(10, 3.2), sharey=True)
        axs = np.atleast_1d(axs)
        for ax, axis in zip(axs, axes_names):
            sub = rr[rr["axis"] == axis]
            sides = list(dict.fromkeys(sw[sw["axis"] == axis].sort_values("side")["group_name"]))
            x = np.arange(len(order))
            for i, side in enumerate(sides):
                vals = sub[sub["group_name"] == side].set_index("model")["refusal_kw"].reindex(order)
                ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=C[i], label=side)
            ax.set_xticks(x, order, rotation=60, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(" vs ".join(sides), fontsize=9, color=INK, loc="left")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            style(ax)
        axs[0].set_ylabel("refusal rate (keyword rule)")
        fig.tight_layout()
        fig.savefig(out / "fig2_refusals.png", dpi=200)
        plt.close(fig)
        tab = rr.pivot_table(index="model", columns="group_name", values="refusal_kw").reindex(order)
        tab.round(3).to_csv(out / "refusal_by_side.csv")
        cat_ref = gen_all[gen_all["part"] == "exp1_category"].pivot_table(
            index="model", columns="category", values="refusal_kw").reindex(order)
        md += ["## Figure 2: refusal rate by identity-swap side", "", "![](fig2_refusals.png)", "",
               md_table(tab.round(2)), "", "Refusal rate for category-conditioned prompts:", "",
               md_table(cat_ref.round(2)), ""]

    # ---------- Figure 3: Spearman with HaHackathon ----------
    lk = pd.read_csv(out / "likert.csv")
    if not lk.empty:
        lk = lk[lk["group"] != "old"].join(jokes[["humor_rating", "offense_rating"]], on="joke_id")
        rows = []
        for (m, g), d in lk.groupby(["model", "group"]):
            d = d.dropna(subset=["funniness"])
            rf = spearmanr(d["funniness"], d["humor_rating"]) if len(d) > 2 else (np.nan, np.nan)
            ra = spearmanr(d["acceptability"], -d["offense_rating"]) if len(d) > 2 else (np.nan, np.nan)
            rows.append({"model": m, "group": g, "n": len(d), "rho_funny": rf[0], "p_funny": rf[1],
                         "rho_accept_vs_offense": ra[0], "p_accept": ra[1]})
        sp = pd.DataFrame(rows)
        sp.to_csv(out / "spearman.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        x = np.arange(len(order))
        for i, g in enumerate(["stable", "shifted"]):
            vals = sp[sp["group"] == g].set_index("model")["rho_funny"].reindex(order)
            ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=C[i], label=f"norm-{g} jokes")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.set_xticks(x, order, rotation=45, ha="right")
        ax.set_ylabel("Spearman rho vs humans")
        ax.set_title("Agreement with HaHackathon humor ratings (n = 60 per group)", fontsize=9, color=INK,
                     loc="left")
        ax.legend(fontsize=7, frameon=False)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        style(ax)
        fig.tight_layout()
        fig.savefig(out / "fig3_spearman.png", dpi=200)
        plt.close(fig)
        wide = sp.pivot_table(index="model", columns="group",
                              values=["rho_funny", "rho_accept_vs_offense"]).reindex(order)
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        md += ["## Figure 3: Spearman rho with human ratings", "", "![](fig3_spearman.png)", "",
               "rho_accept_vs_offense correlates model acceptability with minus the human offense rating.", "",
               md_table(wide.round(2)), ""]

    # ---------- Binary, pairwise, norm QA ----------
    bi = pd.read_csv(out / "binary.csv")
    if not bi.empty:
        b = bi.pivot_table(index="model", columns="group", values="answer",
                           aggfunc=lambda s: (s == "yes").mean()).reindex(order)
        b.round(3).to_csv(out / "binary_summary.csv")
        md += ["## Binary: share judged funny", "", md_table(b.round(2)), ""]
    pw = pd.read_csv(out / "pairwise.csv")
    if not pw.empty:
        pairs = pd.read_csv(DATA / "pilot_pairs.csv").set_index("pair_id")
        hr = jokes["humor_rating"]
        pairs["human_funnier"] = np.where(hr.reindex(pairs["shifted_id"]).values >
                                          hr.reindex(pairs["stable_id"]).values, "shifted", "stable")
        pw = pw.dropna(subset=["funnier"])
        pw["chose"] = np.where(pw["funnier"].str.upper() == "A", pw["a_is"],
                               np.where(pw["a_is"] == "shifted", "stable", "shifted"))
        pw["agree_human"] = pw["chose"] == pairs["human_funnier"].reindex(pw["pair_id"]).values
        cons = pw.pivot_table(index=["model", "pair_id"], columns="order", values="chose", aggfunc="first")
        consistent = (cons[0] == cons[1]).groupby("model").mean() if {0, 1} <= set(cons.columns) else None
        ps = pw.groupby("model").agg(chose_shifted=("chose", lambda s: (s == "shifted").mean()),
                                     agree_with_human=("agree_human", "mean"),
                                     chose_A=("funnier", lambda s: (s.str.upper() == "A").mean()))
        if consistent is not None:
            ps["order_consistent"] = consistent
        ps = ps.reindex(order)
        ps.round(3).to_csv(out / "pairwise_summary.csv")
        md += ["## Pairwise (shifted vs stable, matched on human rating, both orders)", "", md_table(ps.round(2)), ""]
    nq = pd.read_csv(out / "normqa.csv")
    if not nq.empty:
        cq = nq[nq["part"] == "exp3_category"]
        if not cq.empty:
            t = cq.pivot_table(index="model", columns=["category", "question"], values="answer", aggfunc="first")
            t.reindex(order).to_csv(out / "normqa_category_answers.csv")
        jq = nq[nq["part"] == "exp3_joke"]
        if not jq.empty and "acceptable_today" in jq:
            jq = jq.assign(drop=jq["acceptable_1990s"] - jq["acceptable_today"])
            s = jq.pivot_table(index="model", columns="group", values="drop").reindex(order)
            s.round(3).to_csv(out / "normqa_summary.csv")
            md += ["## Norm QA: mean drop in acceptability, 1990s minus today (joke level)", "",
                   md_table(s.round(2)), "", "Category-level answers: normqa_category_answers.csv", ""]

    # ---------- Batch condition (one prompt per part) vs one call per item ----------
    bc_path = out / "batch_conditions.csv"
    if bc_path.exists() and bc_path.stat().st_size > 1:
        bc = pd.read_csv(bc_path)
        md += ["## Batch condition: one prompt per part vs one call per item", "",
               "Each part was also asked as a single prompt (e.g. 'Tell me 100 jokes.'), repeated "
               f"{int(bc['repeat'].max()) + 1} times per model.", ""]
        d = bc.groupby(["model", "part"]).agg(asked=("n_requested", "sum"), returned=("n_returned", "sum"),
                                             cut_off=("truncated", "sum")).reset_index()
        d["delivered"] = d["returned"] / d["asked"]
        uq = bc.dropna(subset=["n_unique"]).groupby(["model", "part"])[["n_unique", "n_returned"]].sum()
        d = d.join((uq["n_unique"] / uq["n_returned"].where(uq["n_returned"] > 0)).rename("unique_share"),
                   on=["model", "part"])
        dw = d.pivot_table(index="model", columns="part", values="delivered").reindex(order)
        d.round(3).to_csv(out / "batch_delivery.csv", index=False)
        md += ["Share of requested items actually delivered (1 = all):", "", md_table(dw.round(2)), "",
               "Per-part counts, duplicates and cut-off replies: batch_delivery.csv", ""]

        bg = pd.read_csv(out / "batch_generations.csv") if (out / "batch_generations.csv").stat().st_size > 1 \
            else pd.DataFrame()
        if labels is not None and not bg.empty:
            bz = bg[bg["part"] == "exp1_zero_shot"].drop(columns=["category"], errors="ignore").merge(labels[["model", "prompt_id", "category"]],
                                                         on=["model", "prompt_id"], how="left")
            bz.loc[bz["refusal_kw"], "category"] = "refusal"
            bz["fig_group"] = bz["category"].map(FIG_GROUPS).fillna("other")
            bdist = pd.crosstab(bz["model"], bz["fig_group"], normalize="index").reindex(
                index=order, columns=GROUP_ORDER, fill_value=0)
            if "dist" in locals():
                comp = pd.DataFrame({"identity_single": dist["identity-targeted"],
                                     "identity_batch": bdist["identity-targeted"],
                                     "pun_single": dist["pun / wordplay"], "pun_batch": bdist["pun / wordplay"]})
                comp.round(3).to_csv(out / "batch_vs_single_categories.csv")
                md += ["Zero-shot joke categories, one joke per call vs 'Tell me 100 jokes.' (share of jokes):", "",
                       md_table(comp.round(2)), ""]
        sw_b = bc[bc["part"] == "exp1_identity_swap"]
        if not sw_b.empty and "tab" in locals():
            sw_b = sw_b.assign(not_delivered=1 - (sw_b["n_returned"] - sw_b["n_refusal_items"]) / sw_b["n_requested"])
            bt = sw_b.pivot_table(index="model", columns="group_name", values="not_delivered").reindex(order)
            both = pd.concat({"single: refusal rate": tab, "batch: share not delivered": bt}, axis=1)
            both.columns = [f"{a} | {b}" for a, b in both.columns]
            both.round(3).to_csv(out / "batch_vs_single_refusals.csv")
            md += ["Identity pairs: refusal rate with one call per joke vs share of the 10 requested jokes "
                   "not delivered (missing or refusing) in the batch prompt: batch_vs_single_refusals.csv", ""]
        bl = pd.read_csv(out / "batch_likert.csv") if (out / "batch_likert.csv").stat().st_size > 1 else pd.DataFrame()
        if not bl.empty and "sp" in locals():
            bl = bl.groupby(["model", "joke_id"])[["funniness", "acceptability"]].mean().reset_index()
            bl = bl.join(jokes[["humor_rating", "group"]], on="joke_id") if "group" in jokes else \
                bl.join(jokes[["humor_rating"]], on="joke_id").join(lk.drop_duplicates("joke_id")
                                                                    .set_index("joke_id")["group"], on="joke_id")
            rows = []
            for (m, g), dd in bl.dropna(subset=["funniness", "humor_rating"]).groupby(["model", "group"]):
                rows.append({"model": m, "group": g, "n": len(dd),
                             "rho_funny_batch": spearmanr(dd["funniness"], dd["humor_rating"])[0] if len(dd) > 2 else np.nan})
            bsp = pd.DataFrame(rows)
            if not bsp.empty:
                cmp = sp[["model", "group", "rho_funny"]].merge(bsp, on=["model", "group"], how="outer")
                cmp = cmp.pivot_table(index="model", columns="group", values=["rho_funny", "rho_funny_batch"]).reindex(order)
                cmp.columns = [f"{a}_{b}".replace("rho_funny_batch", "batch").replace("rho_funny", "single")
                               for a, b in cmp.columns]
                cmp.round(3).to_csv(out / "batch_vs_single_spearman.csv")
                md += ["Spearman rho with HaHackathon funniness, one joke per call vs all 120 in one prompt "
                       "(batch averaged over repeats):", "", md_table(cmp.round(2)), ""]
        bq = pd.read_csv(out / "batch_normqa.csv") if (out / "batch_normqa.csv").stat().st_size > 1 else pd.DataFrame()
        nq_c = nq[nq["part"] == "exp3_category"] if "nq" in locals() and not nq.empty else pd.DataFrame()
        if not bq.empty and not nq_c.empty:
            single = nq_c.assign(answer=nq_c["answer"].astype(str).str.lower().str.replace(r"\.0$", "", regex=True))
            bq0 = bq[bq["repeat"] == 0].assign(answer=lambda x: x["answer"].astype(str).str.replace(r"\.0$", "", regex=True))
            mm = bq0.merge(single[["model", "category", "question", "answer"]], on=["model", "category", "question"],
                           suffixes=("_batch", "_single"))
            agree = (mm["answer_batch"] == mm["answer_single"]).groupby(mm["model"]).mean().reindex(order)
            agree.round(3).to_csv(out / "batch_vs_single_normqa.csv")
            md += ["Norm questions: share of the 18 answers that are the same in batch and single form:", "",
                   md_table(agree.round(2).to_frame("same_answer")), ""]

    # ---------- Feasibility ----------
    f = calls.groupby("model").agg(calls=("prompt_id", "size"), input_tokens=("input_tokens", "sum"),
                                   output_tokens=("output_tokens", "sum"), hours=("latency_s", lambda s: s.sum() / 3600),
                                   mean_latency_s=("latency_s", "mean"))
    js = calls[calls["parse_ok"].notna()]
    f["unparsed_rate"] = js.groupby("model")["parse_ok"].apply(lambda s: 1 - s.astype(bool).mean())
    f["retry_rate"] = js.groupby("model")["attempts"].apply(lambda s: (s == 2).mean())
    by = cfg["by_name"]
    f["provider"] = [by[m]["provider"] for m in f.index]
    f["cost_usd"] = [(r.input_tokens * by[m].get("price_in", 0) + r.output_tokens * by[m].get("price_out", 0)) / 1e6
                     for m, r in f.iterrows()]
    f = f.reindex(order)
    notes = []
    if labels is not None and "input_tokens" in labels:
        lm = by[cfg["labeler"]]
        lab_cost = (labels["input_tokens"].sum() * lm.get("price_in", 0)
                    + labels["output_tokens"].sum() * lm.get("price_out", 0)) / 1e6
        notes.append(f"Labeling cost ({cfg['labeler']}): ${lab_cost:.2f}")
    hl = out / "hand_labels.csv"
    if hl.exists() and labels is not None:
        h = pd.read_csv(hl, encoding_errors="replace").dropna(subset=["category"])  # survives a non-UTF-8 Excel save
        h = h[h["category"].astype(str).str.strip() != ""]
        if len(h):
            m = h.merge(labels, on=["model", "prompt_id"], suffixes=("_human", "_model"))
            notes.append(f"Provisional category kappa (labeler vs your {len(m)} hand labels): "
                         f"{kappa(m['category_human'].str.strip().str.lower(), m['category_model']):.2f}")
            mm = m.dropna(subset=["mechanism_human"])
            if len(mm):
                notes.append(f"Mechanism kappa: {kappa(mm['mechanism_human'].str.strip().str.lower(), mm['mechanism_model']):.2f}")
    rc = out / "refusal_check.csv"
    if rc.exists():
        r = pd.read_csv(rc, encoding_errors="replace").dropna(subset=["refusal_human"])
        if len(r):
            hum, kw = r["refusal_human"].astype(int).astype(bool), r["refusal_kw"].astype(bool)
            notes.append(f"Refusal keyword rule on {len(r)} hand-checked outputs: accuracy {(hum == kw).mean():.2f}, "
                         f"precision {(hum & kw).sum() / max(kw.sum(), 1):.2f}, recall {(hum & kw).sum() / max(hum.sum(), 1):.2f}")
    f.round(4).to_csv(out / "feasibility.csv")
    bdir = RUNS / f"{args.run_id}-batch"
    if bdir.exists():
        brs = [json.loads(l) for p in bdir.glob("*.jsonl") for l in open(p, encoding="utf-8") if l.strip()]
        brs = pd.DataFrame([r for r in brs if not r.get("error")])
        if not brs.empty:
            bf = brs.groupby("model").agg(calls=("prompt_id", "size"), input_tokens=("input_tokens", "sum"),
                                          output_tokens=("output_tokens", "sum"),
                                          hours=("latency_s", lambda s: s.sum() / 3600))
            bf["cost_usd"] = [(r.input_tokens * by[m].get("price_in", 0) + r.output_tokens * by[m].get("price_out", 0)) / 1e6
                              for m, r in bf.iterrows()]
            bf = bf.reindex([m for m in order if m in bf.index])
            bf.round(4).to_csv(out / "batch_feasibility.csv")
            notes.append(f"Batch condition: {int(bf['calls'].sum())} calls, ${bf['cost_usd'].sum():.2f} OpenAI cost, "
                         f"{bf['hours'].sum():.1f} h total model time (batch_feasibility.csv)")
    tot = f["cost_usd"].sum()
    md += ["## Feasibility", "", md_table(f.round(3)), "",
           f"Total OpenAI cost for model calls: ${tot:.2f}. Local GPU hours: "
           f"{f.loc[f.provider == 'ollama', 'hours'].sum():.1f}.", ""] + [f"- {n}" for n in notes]
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(notes))
    print(f"Wrote figures and summary.md to {out}")


if __name__ == "__main__":
    main()
