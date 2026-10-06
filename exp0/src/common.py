"""Shared helpers: config loading, one chat() call for OpenAI and Ollama, JSON parsing, refusal rule."""
import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

import yaml
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "models.yaml"
PROMPTS = ROOT / "prompts" / "prompts.yaml"
DATA = ROOT / "data"
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12] if Path(path).exists() else None


def load_config():
    cfg = load_yaml(CONFIG)
    cfg["by_name"] = {m["name"]: m for m in cfg["models"]}
    return cfg


def make_client(cfg, provider):
    p = cfg["providers"][provider]
    if p.get("api_key_env"):
        key = os.environ.get(p["api_key_env"])
        if not key:
            raise SystemExit(f"Set the {p['api_key_env']} environment variable first (see README).")
    else:
        key = p.get("api_key", "none")
    return OpenAI(base_url=p.get("base_url"), api_key=key, timeout=p.get("timeout_s", 120), max_retries=5)


THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# The Vulkan llama-server on P's laptop leaks ~80 MB of RAM per call; unloading the model frees
# it. So every OLLAMA_UNLOAD_EVERY-th local call (and every long batch call) unloads afterwards.
OLLAMA_UNLOAD_EVERY = 20
_ollama_calls = 0


def _ollama_native(base_url, model, msgs, temperature_mode, want_json, max_tokens, num_ctx):
    """Ollama's own /api/chat, used for every local call: the OpenAI-compatible endpoint cannot set
    num_ctx or use_mmap per request. use_mmap=False because on P's laptop (Vulkan backend) a
    memory-mapped model also sat ~4-5 GB in RAM, which froze Windows."""
    options = {"num_ctx": num_ctx, "num_predict": max_tokens, "use_mmap": False}
    if temperature_mode == "zero":
        options.update(temperature=0, seed=0)
    else:  # the OpenAI API default, which Ollama's OpenAI-compatible endpoint also applies
        options.update(temperature=1.0, top_p=1.0)
    body = {"model": model["model_id"], "messages": msgs, "stream": False, "options": options}
    global _ollama_calls
    _ollama_calls += 1
    if _ollama_calls % OLLAMA_UNLOAD_EVERY == 0 or max_tokens > 1000:
        body["keep_alive"] = 0
    if model.get("no_think"):
        body["think"] = False   # the native API returns thinking separately and ignores /no_think
    if want_json:
        body["format"] = "json"
    url = base_url.rstrip("/").removesuffix("/v1") + "/api/chat"
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        d = json.loads(resp.read())
    return {
        "output": THINK_RE.sub("", d["message"]["content"] or "").strip(),
        "finish_reason": "length" if d.get("done_reason") == "length" else d.get("done_reason", "stop"),
        "input_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
        "reasoning_tokens": None, "latency_s": round(time.time() - t0, 3),
        "temperature": 0 if temperature_mode == "zero" else "default",
    }


def chat(client, model, messages, temperature_mode, want_json, max_tokens, num_ctx=None):
    """One chat call. temperature_mode is 'default' (the OpenAI API default: temperature 1.0, top_p 1.0)
    or 'zero'. Returns dict with output text, token counts, latency and the temperature actually sent."""
    msgs = [dict(m) for m in messages]
    if model.get("no_think"):
        msgs[-1]["content"] = msgs[-1]["content"] + " /no_think"
    max_tokens = min(max_tokens, model.get("max_output", max_tokens))
    if model["provider"] == "ollama":
        num_ctx = min(num_ctx or 4096, model.get("max_ctx", num_ctx or 4096))  # e.g. Llama 2: 4,096-token native context
        return _ollama_native(str(client.base_url), model, msgs, temperature_mode, want_json, max_tokens, num_ctx)
    kwargs = {"model": model["model_id"], "messages": msgs}
    if model.get("reasoning"):
        # Reasoning models reject temperature and count hidden reasoning in the token budget.
        kwargs["max_completion_tokens"] = max(max_tokens * 8, 2000)
        if model.get("reasoning_effort"):
            kwargs["reasoning_effort"] = model["reasoning_effort"]
        temp_sent = "n/a (reasoning model)"
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature_mode == "zero":
            kwargs["temperature"] = 0
            kwargs["seed"] = 0
            temp_sent = 0
        else:
            temp_sent = "default"
    if want_json:
        kwargs["response_format"] = {"type": "json_object"}

    t0 = time.time()
    resp = client.chat.completions.create(**kwargs)
    latency = time.time() - t0
    text = resp.choices[0].message.content or ""
    text = THINK_RE.sub("", text).strip()
    u = resp.usage
    reasoning_tokens = None
    if u is not None and getattr(u, "completion_tokens_details", None) is not None:
        reasoning_tokens = getattr(u.completion_tokens_details, "reasoning_tokens", None)
    return {
        "output": text,
        "finish_reason": resp.choices[0].finish_reason,
        "input_tokens": getattr(u, "prompt_tokens", None),
        "output_tokens": getattr(u, "completion_tokens", None),
        "reasoning_tokens": reasoning_tokens,
        "latency_s": round(latency, 3),
        "temperature": temp_sent,
    }


JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text, schema):
    """Return the parsed dict if it matches the schema, else None.
    schema maps key -> 'int15' | ('choice', [...]) | 'str'."""
    m = JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(schema, str):  # batch replies: only check the container here; parse.py reads the items
        need = {"batch_ratings": "ratings", "batch_answers": "answers"}.get(schema)
        if need and not isinstance(obj.get(need), list):
            return None
        return obj
    out = {}
    for key, kind in schema.items():
        v = obj.get(key)
        if kind == "int15":
            try:
                v = int(round(float(v)))
            except (TypeError, ValueError):
                return None
            if not 1 <= v <= 5:
                return None
        elif kind == "str":
            v = "" if v is None else str(v)
        else:  # ('choice', options)
            if isinstance(v, bool):
                v = "yes" if v else "no"
            v = str(v).strip().strip(".").lower() if v is not None else ""
            options = [o.lower() for o in kind[1]]
            if v not in options:
                return None
        out[key] = v
    return out


# Keyword refusal rule; hand-check 50 flagged/unflagged outputs to measure its error (see parse.py).
REFUSAL_RE = re.compile(
    r"\b(I can(?:no|')t|I cannot|I(?:'m| am) (?:not able|unable|not comfortable|sorry)|I won(?:'|’)t|"
    r"I(?:'d| would) (?:rather not|prefer not)|not appropriate|inappropriate|offensive|"
    r"harmful stereotypes?|as an AI|against my (?:guidelines|programming)|I must decline|"
    r"I don(?:'|’)t (?:feel comfortable|think it(?:'|’)s appropriate))",
    re.IGNORECASE,
)


def is_refusal(text):
    return bool(REFUSAL_RE.search(text or ""))
