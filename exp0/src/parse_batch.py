"""Turn the batch-condition logs (one prompt per part) into tables that line up with the single-call ones.

    python src/parse_batch.py --run-id pilot1        # reads runs/pilot1-batch/

Writes results/<run-id>/:
    batch_generations.csv   one row per joke pulled out of a batch reply (same columns as generations.csv)
    batch_conditions.csv    per model x repeat x condition: jokes asked for, returned, unique, refused, cut off
    batch_likert.csv        one row per joke rating from the batch rating prompt
    batch_normqa.csv        one row per answer from the batch norm-question prompt
"""
import argparse
import json
import re

import pandas as pd

from common import RESULTS, RUNS, is_refusal

ITEM_RE = re.compile(r"^\s*(?:\d{1,3}\s*[.):]|[-*•])\s+", re.MULTILINE)


def split_list(text):
    """Split a free-text list of jokes ('1. ...', '2) ...', '- ...') into items."""
    text = (text or "").strip()
    starts = [m.start() for m in ITEM_RE.finditer(text)]
    if len(starts) >= 2:
        parts = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
        items = [ITEM_RE.sub("", p, count=1).strip() for p in parts]
    else:
        items = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [i for i in items if len(i) > 5]


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def int15(v):
    try:
        v = int(round(float(v)))
        return v if 1 <= v <= 5 else None
    except (TypeError, ValueError):
        return None


def load_batch_run(run_id):
    rows = []
    for path in sorted((RUNS / run_id).glob("*.jsonl")):
        for line in open(path, encoding="utf-8"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not rows:
        raise SystemExit(f"No logs in runs/{run_id}/ (run src/run.py --batch first)")
    n_err = sum(bool(r.get("error")) for r in rows)
    last = {}
    for r in rows:
        if not r.get("error"):
            last[(r["model"], r["prompt_id"])] = r
    out = []
    for r in last.values():
        m = r.pop("meta")
        out.append(type("Rec", (), {**r, **m}))
    return out, n_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    args = ap.parse_args()
    recs, n_err = load_batch_run(f"{args.run_id}-batch")
    out = RESULTS / args.run_id
    out.mkdir(parents=True, exist_ok=True)

    gens, conds, likert, qa = [], [], [], []
    for r in recs:
        base = {"model": r.model, "repeat": r.repeat}
        truncated = getattr(r, "finish_reason", None) == "length"
        part = r.part.removeprefix("batch_")
        if r.part == "batch_exp1_zero_shot":
            items = split_list(r.raw_output)
            groups = {"all": items}
        elif r.part in ("batch_exp1_category", "batch_exp1_identity_swap"):
            obj = getattr(r, "parsed", None)
            obj = obj if isinstance(obj, dict) else {}
            groups = {}
            for k in r.keys:
                v = obj.get(k, [])
                v = [v] if isinstance(v, str) else v if isinstance(v, list) else []
                groups[k] = [str(x).strip() for x in v if str(x).strip()]
        else:
            groups = None

        if groups is not None:
            n_req = r.n_requested if part == "exp1_zero_shot" else r.n_per_key
            for key, items in groups.items():
                row = {"part": part, "condition": key}
                if part == "exp1_identity_swap":
                    axis, side = key.rsplit("_", 1)
                    row.update(axis=axis, side=int(side), group_name=r.group_names[key])
                elif part == "exp1_category":
                    row["category"] = key
                for i, text in enumerate(items):
                    gens.append({**base, **row, "prompt_id": f"{r.prompt_id}#{key}#{i:03d}", "position": i,
                                 "raw_output": text, "refusal_kw": is_refusal(text)})
                n_ref = sum(is_refusal(t) for t in items)
                conds.append({**base, **row, "n_requested": n_req, "n_returned": len(items),
                              "n_unique": len({norm(t) for t in items}), "n_refusal_items": n_ref,
                              "refused": len(items) == 0 or n_ref == len(items),
                              "whole_reply_refusal": len(items) <= 1 and is_refusal(r.raw_output),
                              "truncated": truncated})
        elif r.part == "batch_exp2_likert":
            p = getattr(r, "parsed", None)
            got = p.get("ratings", []) if isinstance(p, dict) else []
            seen = set()
            for it in got:
                if not isinstance(it, dict) or it.get("id") not in r.ids or it["id"] in seen:
                    continue
                seen.add(it["id"])
                likert.append({**base, "joke_id": it["id"], "funniness": int15(it.get("funniness")),
                               "acceptability": int15(it.get("acceptability"))})
            conds.append({**base, "part": "exp2_likert", "condition": "all", "n_requested": len(r.ids),
                          "n_returned": len(seen), "truncated": truncated,
                          "whole_reply_refusal": not seen and is_refusal(r.raw_output)})
        elif r.part == "batch_exp3_category":
            p = getattr(r, "parsed", None)
            got = p.get("answers", []) if isinstance(p, dict) else []
            seen = set()
            for it in got:
                if not isinstance(it, dict) or it.get("id") not in r.ids or it["id"] in seen:
                    continue
                seen.add(it["id"])
                cat, q = it["id"].split("__")
                a = it.get("answer")
                a = int15(a) if q == "acceptable_today" else str(a).strip().strip(".").lower()
                qa.append({**base, "category": cat, "question": q, "answer": a})
            conds.append({**base, "part": "exp3_category", "condition": "all", "n_requested": len(r.ids),
                          "n_returned": len(seen), "truncated": truncated,
                          "whole_reply_refusal": not seen and is_refusal(r.raw_output)})

    pd.DataFrame(gens).to_csv(out / "batch_generations.csv", index=False)
    c = pd.DataFrame(conds)
    c.to_csv(out / "batch_conditions.csv", index=False)
    pd.DataFrame(likert).to_csv(out / "batch_likert.csv", index=False)
    pd.DataFrame(qa).to_csv(out / "batch_normqa.csv", index=False)

    s = c.groupby(["model", "part"]).agg(asked=("n_requested", "sum"), returned=("n_returned", "sum"),
                                         cut_off=("truncated", "sum")).reset_index()
    print(s.to_string(index=False))
    if n_err:
        print(f"\n{n_err} failed calls skipped; re-run src/run.py --batch to fill them in.")
    print(f"\nBatch tables written to {out}")


if __name__ == "__main__":
    main()
