"""Build the Exp. 0 joke set from HaHackathon (SemEval-2021 Task 7).

Input : the HaHackathon train CSV (columns: id, text, is_humor, humor_rating,
        humor_controversy, offense_rating).
Output: data/pilot_jokes.csv      120 jokes (60 norm-shifted, 60 norm-stable)
        data/pilot_pairs.csv      30 shifted-vs-stable pairs matched on human humor rating
        data/pilot_candidates.csv spare jokes per category, for swapping bad picks by hand

Categories come from keyword rules, so READ pilot_jokes.csv before running models
(Day 2) and swap any joke whose category is wrong with one from pilot_candidates.csv.

Usage:
    python src/build_pilot_set.py path/to/hahackathon_train.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 2026

# Norm-shifted categories: identity targets whose joke acceptability changed.
SHIFTED = {
    "gender": r"wom[ae]n|wife|wives|girlfriend|girls?|blondes?|feminis\w*|housewi\w*|mother[- ]in[- ]law",
    "ethnicity": r"black (?:man|men|guy|people|kid|woman)|blacks|asians?|chinese|mexicans?|jews?|jewish|african|"
                 r"indians?|arabs?|irish(?:man)?|polish|pakistani|ethiopian|white people|immigrants?",
    "disability": r"blind|deaf|wheelchair|autis\w*|retard\w*|disabled|down'?s syndrome|dwarfs?|midgets?|"
                  r"amputee|paralys\w*|stutter\w*|cripple\w*",
    "sexual_orientation": r"gay|gays|lesbians?|homosexual\w*|queer|bisexual|trans(?:gender)?|transsexual",
    "appearance": r"fat|fatty|obese|ugly|bald|overweight|short people|skinny",
}
# Sensitive topics that are neither clearly stable nor in the 5 shifted categories;
# jokes matching these are kept out of the stable set.
OTHER_SENSITIVE = r"muslims?|islam|christians?|catholic|priests?|god|jesus|religio\w*|dead|death|die[sd]?|kill\w*|" \
                  r"rape\w*|suicide|abortion|cancer|nazis?|hitler|terror\w*|bomb\w*|trump|obama|sex\w*|porn\w*"

PER_SHIFTED_CAT = 12          # 5 x 12 = 60
N_STABLE = 60
N_PAIRS = 30
N_NORMQA_EACH = 15            # 15 shifted + 15 stable = 30 joke-level norm questions
STABLE_MAX_OFFENSE = 0.5      # HaHackathon offense_rating is on 0-5


def rx(pattern):
    return re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE)


def stratified_sample(df, n, rng, col="humor_rating", bins=4):
    """Sample n rows spread across humor-rating quartiles, so the Spearman test has range."""
    if len(df) <= n:
        return df
    q = pd.qcut(df[col].rank(method="first"), bins, labels=False)
    per_bin = [n // bins + (1 if i < n % bins else 0) for i in range(bins)]
    parts = []
    for b, k in zip(range(bins), per_bin):
        part = df[q == b]
        parts.append(part.sample(min(k, len(part)), random_state=rng.integers(1e9)))
    out = pd.concat(parts)
    if len(out) < n:  # top up if a bin was short
        rest = df.drop(out.index)
        out = pd.concat([out, rest.sample(n - len(out), random_state=rng.integers(1e9))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hahackathon_csv")
    ap.add_argument("--max-words", type=int, default=60, help="skip very long texts")
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    df = pd.read_csv(args.hahackathon_csv)
    need = {"id", "text", "is_humor", "humor_rating", "offense_rating"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns {missing}; found {list(df.columns)}")

    df = df[(df["is_humor"] == 1) & df["humor_rating"].notna()].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.split().str.len() <= args.max_words]
    df = df.drop_duplicates("text")
    print(f"{len(df)} humorous texts with ratings after filtering")

    # Assign each joke to at most one shifted category (first match wins, ambiguous ones dropped).
    hits = {cat: df["text"].str.contains(rx(p)) for cat, p in SHIFTED.items()}
    hit_df = pd.DataFrame(hits)
    n_hits = hit_df.sum(axis=1)
    df["category"] = None
    for cat in SHIFTED:
        df.loc[hit_df[cat] & (n_hits == 1), "category"] = cat

    sensitive = df["text"].str.contains(rx(OTHER_SENSITIVE))
    stable_pool = df[(n_hits == 0) & ~sensitive & (df["offense_rating"] <= STABLE_MAX_OFFENSE)]

    chosen, spares = [], []
    for cat in SHIFTED:
        pool = df[df["category"] == cat]
        print(f"  {cat:20s} {len(pool):5d} candidates")
        if len(pool) < PER_SHIFTED_CAT:
            print(f"  WARNING: only {len(pool)} {cat} jokes; widen the keyword list")
        pick = stratified_sample(pool, PER_SHIFTED_CAT, rng)
        chosen.append(pick.assign(group="shifted"))
        spares.append(pool.drop(pick.index).head(3 * PER_SHIFTED_CAT).assign(group="shifted"))
    print(f"  {'stable':20s} {len(stable_pool):5d} candidates")
    pick = stratified_sample(stable_pool, N_STABLE, rng)
    chosen.append(pick.assign(group="stable", category="stable"))
    spares.append(stable_pool.drop(pick.index).sample(min(3 * N_STABLE, len(stable_pool) - len(pick)),
                                                      random_state=SEED).assign(group="stable", category="stable"))

    jokes = pd.concat(chosen).reset_index(drop=True)
    jokes["joke_id"] = [f"j{i:03d}" for i in range(len(jokes))]
    jokes["source"] = "hahackathon"
    jokes["era"] = "2010s"

    # Joke-level norm questions: 15 shifted (3 per category) + 15 stable.
    jokes["normqa"] = 0
    sh = jokes[jokes.group == "shifted"].groupby("category").head(N_NORMQA_EACH // len(SHIFTED))
    st = jokes[jokes.group == "stable"].head(N_NORMQA_EACH)
    jokes.loc[sh.index.union(st.index), "normqa"] = 1

    # Pairs: 6 shifted per category, each matched to the stable joke with the closest human humor rating.
    shifted_for_pairs = jokes[jokes.group == "shifted"].groupby("category").head(N_PAIRS // len(SHIFTED))
    stable_left = jokes[jokes.group == "stable"].copy()
    pairs = []
    for _, s in shifted_for_pairs.iterrows():
        j = (stable_left["humor_rating"] - s["humor_rating"]).abs().idxmin()
        t = stable_left.loc[j]
        stable_left = stable_left.drop(j)
        pairs.append({"pair_id": f"p{len(pairs):02d}", "shifted_id": s.joke_id, "stable_id": t.joke_id,
                      "shifted_humor": s.humor_rating, "stable_humor": t.humor_rating,
                      "human_funnier": "shifted" if s.humor_rating > t.humor_rating else "stable"})

    cols = ["joke_id", "text", "group", "category", "humor_rating", "offense_rating",
            "humor_controversy", "normqa", "source", "era", "id"]
    cols = [c for c in cols if c in jokes.columns]
    DATA.mkdir(exist_ok=True)
    jokes[cols].rename(columns={"id": "source_id"}).to_csv(DATA / "pilot_jokes.csv", index=False)
    pd.DataFrame(pairs).to_csv(DATA / "pilot_pairs.csv", index=False)
    sp = pd.concat(spares)
    sp[[c for c in cols if c in sp.columns and c != "normqa"]].rename(columns={"id": "source_id"}) \
        .to_csv(DATA / "pilot_candidates.csv", index=False)

    print(f"\nWrote {len(jokes)} jokes, {len(pairs)} pairs, {len(sp)} spare candidates to {DATA}")
    print(jokes.groupby(["group", "category"]).size().to_string())
    print("\nNext: read data/pilot_jokes.csv. To replace a joke whose category is wrong, paste a row's "
          "text and ratings from data/pilot_candidates.csv over it but KEEP its joke_id, so the pairs "
          "file still points to it. Re-running this script draws a fresh sample and overwrites all three files.")


if __name__ == "__main__":
    main()
