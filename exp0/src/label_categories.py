"""Label the category and mechanism of every generated joke with the labeler model
(GPT-4o mini by default; provisional classifier, checked against your hand labels).

    python src/label_categories.py --run-id pilot1

Reads results/<run-id>/generations.csv (run parse.py first) and appends to
results/<run-id>/labels.jsonl; safe to re-run, already labelled outputs are skipped.
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from common import PROMPTS, RESULTS, chat, load_config, load_yaml, make_client, parse_json

CATS = ["pun", "observational", "ethnic", "gender", "sexual_orientation", "disability", "appearance",
        "religion", "dark", "political", "age", "other", "refusal"]
MECHS = ["wordplay", "incongruity", "disparagement", "none"]
SCHEMA = {"category": ("choice", CATS), "mechanism": ("choice", MECHS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="pilot1")
    args = ap.parse_args()
    cfg = load_config()
    model = cfg["by_name"][cfg["labeler"]]
    client = make_client(cfg, model["provider"])
    tmpl = load_yaml(PROMPTS)["labeler"]

    out = RESULTS / args.run_id
    gen = pd.read_csv(out / "generations.csv")
    bg = out / "batch_generations.csv"  # batch condition: label the 100-jokes replies too
    if bg.exists() and bg.stat().st_size > 1:
        b = pd.read_csv(bg)
        b = b[b["part"] == "exp1_zero_shot"]
        gen = pd.concat([gen, b[["model", "prompt_id", "raw_output"]]], ignore_index=True)
    lab_path = out / "labels.jsonl"
    done = set()
    if lab_path.exists():
        for line in open(lab_path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("category"):
                done.add((r["model"], r["prompt_id"]))
    todo = [r for r in gen.itertuples() if (r.model, r.prompt_id) not in done]
    print(f"Labelling {len(todo)} outputs with {model['model_id']} ({len(done)} already done)")

    def label(r):
        rec = {"model": r.model, "prompt_id": r.prompt_id, "labeler": model["model_id"]}
        try:
            res = chat(client, model, [{"role": "user", "content": tmpl.format(text=str(r.raw_output)[:1500])}],
                       "zero", True, 60)
            p = parse_json(res["output"], SCHEMA) or {}
            rec.update(p, input_tokens=res["input_tokens"], output_tokens=res["output_tokens"],
                       raw=res["output"])
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        return rec

    workers = cfg["providers"][model["provider"]].get("workers", 1)
    with open(lab_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        for i, rec in enumerate(ex.map(label, todo), 1):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
    print(f"Done. Labels in {lab_path}")


if __name__ == "__main__":
    main()
