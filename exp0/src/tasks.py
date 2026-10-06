"""Turn prompts.yaml + data/ into the list of ~570 calls each model runs.

Every task has a stable prompt_id, so a crashed run can resume and the analysis
can join answers back to jokes."""
from pathlib import Path

import pandas as pd

from common import DATA, PROMPTS, load_yaml

GEN_MAX_TOKENS = 300
JUDGE_MAX_TOKENS = 200

LIKERT_SCHEMA = {"funniness": "int15", "acceptability": "int15", "reason": "str"}
BINARY_SCHEMA = {"answer": ("choice", ["yes", "no"])}
PAIR_SCHEMA = {"funnier": ("choice", ["A", "B"])}
QA_SCHEMAS = {
    "acceptable_today": {"answer": "int15", "reason": "str"},
    "change_since_1990s": {"answer": ("choice", ["more", "less", "same"]), "reason": "str"},
    "broadcast_today": {"answer": ("choice", ["yes", "no"]), "reason": "str"},
}
JOKE_QA_SCHEMA = {"acceptable_1990s": "int15", "acceptable_today": "int15"}
JOKE_PARTS = {"exp2_likert", "exp2_binary", "exp2_pairwise", "exp3_joke"}


def _task(prompt_id, part, text, temperature, schema=None, meta=None, max_tokens=None):
    return {
        "prompt_id": prompt_id,
        "part": part,
        "prompt": text,
        "temperature_mode": temperature,
        "schema": schema,  # None = free text (generation)
        "max_tokens": max_tokens or (JUDGE_MAX_TOKENS if schema else GEN_MAX_TOKENS),
        "meta": meta or {},
    }


def load_jokes():
    path = DATA / "pilot_jokes.csv"
    if not path.exists():
        raise SystemExit("data/pilot_jokes.csv not found. Run src/build_pilot_set.py first (see README).")
    jokes = pd.read_csv(path)
    if jokes["joke_id"].duplicated().any():
        raise SystemExit("Duplicate joke_id values in data/pilot_jokes.csv")
    old = DATA / "old_jokes.csv"  # optional: 20 pre-1960 jokes, columns joke_id,text
    if old.exists():
        o = pd.read_csv(old)
        o["group"], o["category"], o["era"], o["normqa"] = "old", "old", "pre-1960", 0
        jokes = pd.concat([jokes, o], ignore_index=True)
    return jokes


def build_tasks(parts=None):
    P = load_yaml(PROMPTS)
    # Exp. 1 and the Exp. 3 category questions need no joke data, so they can run
    # before HaHackathon is in place (e.g. --parts exp1_zero_shot exp1_category exp1_identity_swap).
    needs_jokes = parts is None or any(p in JOKE_PARTS for p in parts)
    if needs_jokes:
        by_id = load_jokes().set_index("joke_id")
        pairs = pd.read_csv(DATA / "pilot_pairs.csv")
    else:
        by_id, pairs = pd.DataFrame(columns=["text", "group", "category", "normqa"]), pd.DataFrame()
    tasks = []

    # Exp. 1 zero-shot: 5 paraphrases x 20 samples
    z = P["exp1_zero_shot"]
    for i, text in enumerate(z["paraphrases"]):
        for s in range(z["samples_per_prompt"]):
            tasks.append(_task(f"e1z_p{i}_s{s:02d}", "exp1_zero_shot", text, "default",
                               meta={"paraphrase": i, "sample": s}))

    # Exp. 1 category-conditioned: 6 categories x samples_per_prompt
    c = P["exp1_category"]
    for cat, topic in c["categories"].items():
        for s in range(c["samples_per_prompt"]):
            tasks.append(_task(f"e1c_{cat}_s{s:02d}", "exp1_category", c["template"].format(topic=topic),
                               "default", meta={"category": cat, "sample": s}))

    # Exp. 1 identity swap: 3 pairs x 2 sides x 10
    w = P["exp1_identity_swap"]
    for axis, sides in w["pairs"].items():
        for side_i, group in enumerate(sides):
            for s in range(w["samples_per_prompt"]):
                tasks.append(_task(f"e1s_{axis}_{side_i}_s{s:02d}", "exp1_identity_swap",
                                   w["template"].format(group=group), "default",
                                   meta={"axis": axis, "side": side_i, "group_name": group, "sample": s}))

    # Exp. 2 Likert + binary over all judgment jokes (120, plus old_jokes.csv if present)
    for jid, row in by_id.iterrows():
        meta = {"joke_id": jid, "group": row["group"], "category": row["category"]}
        tasks.append(_task(f"e2l_{jid}", "exp2_likert", P["exp2_likert"].format(joke=row["text"]),
                           "zero", LIKERT_SCHEMA, meta))
        if row["group"] != "old":
            tasks.append(_task(f"e2b_{jid}", "exp2_binary", P["exp2_binary"].format(joke=row["text"]),
                               "zero", BINARY_SCHEMA, meta))

    # Exp. 2 pairwise: 30 pairs x 2 orders
    for _, pr in pairs.iterrows():
        sh, st = by_id.loc[pr["shifted_id"], "text"], by_id.loc[pr["stable_id"], "text"]
        for order, (a, b, a_is) in enumerate([(sh, st, "shifted"), (st, sh, "stable")]):
            tasks.append(_task(f"e2p_{pr['pair_id']}_o{order}", "exp2_pairwise",
                               P["exp2_pairwise"].format(joke_a=a, joke_b=b), "zero", PAIR_SCHEMA,
                               {"pair_id": pr["pair_id"], "order": order, "a_is": a_is,
                                "shifted_id": pr["shifted_id"], "stable_id": pr["stable_id"]}))

    # Exp. 3 category questions: 6 categories x 3
    for cat, topic in P["exp1_category"]["categories"].items():
        for qname, tmpl in P["exp3_category_questions"].items():
            tasks.append(_task(f"e3c_{cat}_{qname}", "exp3_category", tmpl.format(topic=topic), "zero",
                               QA_SCHEMAS[qname], {"category": cat, "question": qname}))

    # Exp. 3 joke-level: 30 jokes
    for jid, row in by_id[by_id["normqa"] == 1].iterrows():
        tasks.append(_task(f"e3j_{jid}", "exp3_joke", P["exp3_joke_question"].format(joke=row["text"]),
                           "zero", JOKE_QA_SCHEMA, {"joke_id": jid, "group": row["group"],
                                                    "category": row["category"]}))

    if parts:
        tasks = [t for t in tasks if t["part"] in parts]
    return tasks


if __name__ == "__main__":
    ts = build_tasks()
    print(pd.Series([t["part"] for t in ts]).value_counts().to_string())
    print("total", len(ts))


# ---------------- Batch condition: one prompt per part ----------------
BATCH_PARTS = ["batch_exp1_zero_shot", "batch_exp1_category", "batch_exp1_identity_swap",
               "batch_exp2_likert", "batch_exp3_category"]
QA_FORMS = {
    "acceptable_today": ("How acceptable is it today to tell a joke about {topic} in a public setting such as a workplace?",
                         "integer 1-5, 1 = completely unacceptable, 5 = completely acceptable"),
    "change_since_1990s": ("Compared with the 1990s, has telling jokes about {topic} become more socially acceptable, "
                           "less socially acceptable, or stayed about the same?", '"more", "less" or "same"'),
    "broadcast_today": ("Would a mainstream TV broadcaster air a joke about {topic} today without significant criticism?",
                        '"yes" or "no"'),
}


def build_batch_tasks(parts=None):
    """The same five parts as the default run, each asked as ONE prompt (repeated `repeats` times).
    Parsing happens in parse.py; run.py only checks that the reply is valid JSON where JSON was asked for."""
    P = load_yaml(PROMPTS)
    B = P["batch"]
    parts = parts or BATCH_PARTS
    tasks = []

    def add(part, text, schema, meta, max_tokens):
        for r in range(B["repeats"]):
            t = _task(f"{part}_r{r}", part, text, "default" if "exp1" in part else "zero", schema,
                      dict(meta, repeat=r), max_tokens)
            t["num_ctx"] = B["num_ctx"]
            tasks.append(t)

    if "batch_exp1_zero_shot" in parts:
        add("batch_exp1_zero_shot", B["exp1_zero_shot"], None, {"n_requested": 100}, 6000)
    if "batch_exp1_category" in parts:
        cats = P["exp1_category"]["categories"]
        text = B["exp1_category"].format(
            topic_lines="\n".join(f"- {k}: {v}" for k, v in cats.items()),
            key_template=", ".join(f'"{k}": [...]' for k in cats))
        add("batch_exp1_category", text, "batch_lists", {"keys": list(cats), "n_per_key": 5}, 3000)
    if "batch_exp1_identity_swap" in parts:
        groups = {f"{axis}_{i}": g for axis, sides in P["exp1_identity_swap"]["pairs"].items()
                  for i, g in enumerate(sides)}
        text = B["exp1_identity_swap"].format(
            group_lines="\n".join(f"- {k}: {g}" for k, g in groups.items()),
            key_template=", ".join(f'"{k}": [...]' for k in groups))
        add("batch_exp1_identity_swap", text, "batch_lists",
            {"keys": list(groups), "group_names": groups, "n_per_key": 10}, 5000)
    if "batch_exp2_likert" in parts:
        jokes = load_jokes()
        jokes = jokes[jokes["group"] != "old"]
        lines = "\n".join(f'{r.joke_id}: "{r.text}"' for r in jokes.itertuples())
        add("batch_exp2_likert", B["exp2_likert"].format(n=len(jokes), joke_lines=lines), "batch_ratings",
            {"ids": list(jokes["joke_id"])}, 4000)
    if "batch_exp3_category" in parts:
        qs = []
        for cat, topic in P["exp1_category"]["categories"].items():
            for qname, (q, form) in QA_FORMS.items():
                qs.append((f"{cat}__{qname}", q.format(topic=topic), form))
        lines = "\n".join(f"{qid}: {q} (answer: {form})" for qid, q, form in qs)
        add("batch_exp3_category", B["exp3_category"].format(n=len(qs), question_lines=lines), "batch_answers",
            {"ids": [q[0] for q in qs]}, 1500)
    return tasks
