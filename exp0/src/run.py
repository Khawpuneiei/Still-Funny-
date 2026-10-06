"""Run Exp. 0 tasks on one or more models and log every call to JSONL.

Examples (from the exp0 folder):
    python src/run.py --smoke --models gpt-4o-mini llama3.1-8b    # 5 test prompts each, timed (Day 1)
    python src/run.py --provider openai                           # all OpenAI models (Day 3)
    python src/run.py --provider ollama                           # all local models, overnight
    python src/run.py --models qwen3-8b --parts exp2_likert       # one model, one part

    python src/run.py --batch --provider openai                   # batch condition: one prompt per part

Logs go to runs/<run-id>/<model>.jsonl and are only ever appended to. Re-running the
same command with the same --run-id skips calls that already succeeded, so a crash or
a closed laptop just means running it again.
"""
import argparse
import json
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from common import CONFIG, DATA, PROMPTS, RUNS, chat, file_hash, load_config, load_yaml, make_client, parse_json
from tasks import BATCH_PARTS, build_batch_tasks, build_tasks

SMOKE_IDS = ["e1z_p0_s00", "e1c_gender_s00", "e1s_gender_1_s00", "e2l_j000", "e3c_gender_acceptable_today"]


def done_ids(path):
    ids = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not r.get("error"):
                    ids.add(r["prompt_id"])
    return ids


def run_one(client, model, task, retry_text, run_id):
    rec = {
        "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": model["provider"], "model": model["name"], "model_id": model["model_id"],
        "prompt_id": task["prompt_id"], "part": task["part"], "prompt": task["prompt"],
        "meta": task["meta"], "error": None,
    }
    messages = [{"role": "user", "content": task["prompt"]}]
    try:
        r = chat(client, model, messages, task["temperature_mode"], task["schema"] is not None, task["max_tokens"],
                 task.get("num_ctx"))
        rec.update({k: r[k] for k in ("temperature", "latency_s", "input_tokens", "output_tokens",
                                       "reasoning_tokens", "finish_reason")})
        rec["raw_output"], rec["attempts"], rec["parsed"] = r["output"], 1, None
        if task["schema"] is not None:
            parsed = parse_json(r["output"], task["schema"])
            if parsed is None:  # retry once, telling the model its answer was not valid JSON
                messages += [{"role": "assistant", "content": r["output"]},
                             {"role": "user", "content": retry_text}]
                r2 = chat(client, model, messages, task["temperature_mode"], True, task["max_tokens"], task.get("num_ctx"))
                rec["first_output"], rec["raw_output"], rec["attempts"] = r["output"], r2["output"], 2
                rec["latency_s"] = round(rec["latency_s"] + r2["latency_s"], 3)
                for k in ("input_tokens", "output_tokens", "reasoning_tokens"):
                    if r2[k] is not None:
                        rec[k] = (rec[k] or 0) + r2[k]
                parsed = parse_json(r2["output"], task["schema"])
            rec["parsed"] = parsed
            rec["parse_ok"] = parsed is not None
    except Exception as e:  # logged and retried on the next run
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", help="model names from configs/models.yaml")
    ap.add_argument("--provider", choices=["openai", "ollama"], help="run every model of this provider")
    ap.add_argument("--parts", nargs="*", help="parts to run (default: default_parts in prompts.yaml)")
    ap.add_argument("--run-id", default="pilot1")
    ap.add_argument("--smoke", action="store_true", help="5 test prompts per model, logged to runs/smoke/")
    ap.add_argument("--batch", action="store_true",
                    help="batch condition: one prompt per part (\"Tell me 100 jokes.\"), logged to runs/<run-id>-batch/")
    ap.add_argument("--limit", type=int, help="at most N new calls per model (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="print the task counts and exit")
    args = ap.parse_args()

    cfg = load_config()
    names = args.models or [m["name"] for m in cfg["models"] if m.get("enabled", True)
                            and (not args.provider or m["provider"] == args.provider)]
    unknown = [n for n in names if n not in cfg["by_name"]]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}. Known: {list(cfg['by_name'])}")
    if args.provider:
        names = [n for n in names if cfg["by_name"][n]["provider"] == args.provider]

    if args.batch:
        tasks = build_batch_tasks([p if p.startswith("batch_") else "batch_" + p for p in args.parts] if args.parts
                                  else BATCH_PARTS)
        run_id = f"{args.run_id}-batch"
    else:
        parts = args.parts or load_yaml(PROMPTS).get("default_parts")
        tasks = build_tasks(parts)
        run_id = "smoke" if args.smoke else args.run_id
    if args.smoke:
        tasks = [t for t in tasks if t["prompt_id"] in SMOKE_IDS]
    print(f"{len(tasks)} tasks per model, models: {', '.join(names)}")
    if args.dry_run:
        from collections import Counter
        for part, n in Counter(t["part"] for t in tasks).items():
            print(f"  {part:20s} {n}")
        return

    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    hashes = {"prompts": file_hash(PROMPTS), "config": file_hash(CONFIG),
              "pilot_jokes": file_hash(DATA / "pilot_jokes.csv"), "pilot_pairs": file_hash(DATA / "pilot_pairs.csv")}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        changed = [k for k in hashes if old["hashes"].get(k) != hashes[k] and k != "config"]
        if changed and not args.smoke:
            print(f"WARNING: {changed} changed since this run started. Results across models may not be "
                  f"comparable. Use a new --run-id if the change is intended.")
    else:
        manifest_path.write_text(json.dumps({"run_id": run_id, "started": datetime.now(timezone.utc).isoformat(),
                                             "hashes": hashes}, indent=2))

    retry_text = load_yaml(PROMPTS)["json_retry"]
    for name in names:
        model = cfg["by_name"][name]
        client = make_client(cfg, model["provider"])
        out_path = run_dir / f"{name}.jsonl"
        skip = done_ids(out_path)
        todo = [t for t in tasks if t["prompt_id"] not in skip]
        random.Random(0).shuffle(todo)  # spread parts over time so a partial run still covers every part
        if args.limit:
            todo = todo[: args.limit]
        print(f"\n== {name} ({model['model_id']}): {len(skip)} done, {len(todo)} to run")
        if not todo:
            continue
        lock = threading.Lock()
        t0, n_err, n_bad = time.time(), 0, 0
        workers = cfg["providers"][model["provider"]].get("workers", 1)
        with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
            futs = [ex.submit(run_one, client, model, t, retry_text, run_id) for t in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                n_err += bool(rec["error"])
                n_bad += rec.get("parse_ok") is False
                if args.smoke:
                    out = rec["error"] or (rec.get("raw_output") or "").replace("\n", " ")[:120]
                    print(f"  {rec['prompt_id']:28s} {rec.get('latency_s', 0) or 0:6.1f}s "
                          f"out_tok={rec.get('output_tokens')}  {out}")
                elif i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el/60:.1f} min elapsed, ~{el/i*(len(todo)-i)/60:.0f} min left, "
                          f"errors={n_err}, unparsed={n_bad}")
        if n_err:
            print(f"  {n_err} calls failed (see 'error' in the log). Run the same command again to retry them.")


if __name__ == "__main__":
    main()
