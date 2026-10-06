"""Turn the JSONL logs of one run into tidy tables, and make the hand-checking sheets.

    python src/parse.py --run-id pilot1

Writes results/<run-id>/:
    calls.csv           one row per prompt per model (last successful attempt)
    generations.csv     Exp. 1 outputs with the keyword refusal flag
    likert.csv, binary.csv, pairwise.csv, normqa.csv
    hand_labels.csv     50 generated jokes for you to label (category, mechanism)   } created once,
    refusal_check.csv   30 outputs for you to mark as refusal or not (1/0)           } never overwritten
"""
import argparse
import json

import pandas as pd

from common import RESULTS, RUNS, is_refusal

SEED = 7


def load_run(run_id):
    rows = []
    for path in sorted((RUNS / run_id).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not rows:
        raise SystemExit(f"No logs in runs/{run_id}/")
    df = pd.DataFrame(rows)
    n_err = df["error"].notna().sum()
    df = df[df["error"].isna()].drop_duplicates(["model", "prompt_id"], keep="last")
    meta = pd.json_normalize(df["meta"]).set_index(df.index)
    df = pd.concat([df.drop(columns=["meta"]), meta], axis=1)
    return df, n_err


def expand(df, part, prefix=""):
    sub = df[df["part"] == part].copy()
    if sub.empty:
        return sub
    parsed = pd.json_normalize(sub["parsed"].apply(lambda x: x or {})).set_index(sub.index)
    parsed.columns = [prefix + c for c in parsed.columns]
    return pd.concat([sub, parsed], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    args = ap.parse_args()
    out = RESULTS / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    df, n_err = load_run(args.run_id)
    df["refusal_kw"] = df["raw_output"].apply(is_refusal)
    df.to_csv(out / "calls.csv", index=False)

    gen = df[df["part"].str.startswith("exp1")].copy()
    gen_cols = [c for c in ["model", "prompt_id", "part", "prompt", "paraphrase", "category", "axis", "side",
                            "group_name", "raw_output", "refusal_kw", "finish_reason"] if c in gen.columns]
    gen[gen_cols].to_csv(out / "generations.csv", index=False)

    for part, name in [("exp2_likert", "likert"), ("exp2_binary", "binary"), ("exp2_pairwise", "pairwise")]:
        expand(df, part).to_csv(out / f"{name}.csv", index=False)
    pd.concat([expand(df, "exp3_category"), expand(df, "exp3_joke")]).to_csv(out / "normqa.csv", index=False)

    # Hand-check sheets (made once so your labels are never overwritten). The BOM makes Excel
    # read them as UTF-8 instead of the Windows code page (emojis, curly quotes).
    hl = out / "hand_labels.csv"
    if not hl.exists():
        pool = gen[~gen["refusal_kw"]]
        s = pool.sample(min(50, len(pool)), random_state=SEED)
        s[["model", "prompt_id", "raw_output"]].assign(category="", mechanism="").to_csv(hl, index=False, encoding="utf-8-sig")
        print(f"Created {hl.name}: fill 'category' (pun, observational, ethnic, gender, sexual_orientation, "
              f"disability, appearance, religion, dark, political, age, other) and 'mechanism' "
              f"(wordplay, incongruity, disparagement).")
    rc = out / "refusal_check.csv"
    if not rc.exists():
        flagged, clean = gen[gen["refusal_kw"]], gen[~gen["refusal_kw"]]
        k = min(15, len(flagged))
        s = pd.concat([flagged.sample(k, random_state=SEED),
                       clean.sample(min(30 - k, len(clean)), random_state=SEED)]).sample(frac=1, random_state=SEED)
        s[["model", "prompt_id", "raw_output", "refusal_kw"]].assign(refusal_human="").to_csv(rc, index=False, encoding="utf-8-sig")
        print(f"Created {rc.name}: put 1 in 'refusal_human' if the output refuses or dodges the joke, else 0.")

    # Console summary
    js = df[df["parse_ok"].notna()]
    summ = df.groupby("model").agg(calls=("prompt_id", "size"))
    summ["unparsed"] = js.groupby("model")["parse_ok"].apply(lambda s: (~s.astype(bool)).sum())
    summ["retried"] = df.groupby("model")["attempts"].apply(lambda s: (s == 2).sum())
    summ["gen_refusal_kw"] = gen.groupby("model")["refusal_kw"].mean().round(3)
    print(summ.fillna(0).to_string())
    if n_err:
        print(f"\n{n_err} failed calls were skipped; re-run src/run.py with the same --run-id to fill them in.")
    print(f"\nTables written to {out}")


if __name__ == "__main__":
    main()
