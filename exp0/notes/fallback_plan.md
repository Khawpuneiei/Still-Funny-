# Fallback: the smallest run that still makes Monday 8 am SGT

Written 4 Oct 2026, 00:30 UTC (08:30 SGT Sun), while the laptop was offline. Times are SGT.
The deadline is Mon 5 Oct, 08:00 SGT (00:00 UTC).

## Planning numbers (estimates, not yet measured on this laptop)

The smoke test replaces these with real timings. Re-plan right after it.

- **Local 7-8B, 4-bit, RTX 4060 Laptop:** about 35 tokens/s.
  - Joke call: ~4 s (Llama 2 is chattier, up to ~8 s).
  - Rating or question call: ~1.5 s.
  - Batch call ("Tell me 100 jokes."): 1-3 min.
  - So one model's single-call run takes ~25 min, and its batch run (3 repeats) ~28 min.
- **OpenAI:** ~1.5 s per call, 4 in parallel. gpt-6-luna is assumed to be a reasoning model at ~10-15 s per call.
  - All 3 GPT models take ~20 min single-call and ~15 min batch.
- **Labeling (GPT-4o mini):** ~10 min for the single-call jokes, plus ~15 min for the batch jokes.
- **Your hand-check:** ~50 min, possible once the single-call run is parsed.
- **One-pager:** ~1 h, including your read-through.

## Options, in the order to cut

| Option | What runs | Total to one-pager | Done by 23:00 Sun if laptop back by | Done by 06:00 Mon if back by |
|---|---|---|---|---|
| **A. Full** | 7 models, all parts, batch × 3 repeats | ~5.5 h | 17:30 Sun | 00:30 Mon |
| **B. Batch × 1** | A, but 1 batch repeat | ~4 h | 19:00 Sun | 02:00 Mon |
| **C. 5 models** | B without the Qwen pair | ~3.25 h | 19:45 Sun | 02:45 Mon |
| **D. Minimum** | 5 models, single-call only, 120 prompts per model, no batch | ~2.5 h | 20:30 Sun | 03:30 Mon |

"06:00 Mon" keeps 2 h of slack before the deadline for one failed step and a re-run.
The human steps (hand-check and one-pager, ~2 h) dominate C and D. So below option C, cutting machine time buys little.

Why these cuts, in this order:
1. **Batch repeats 3 → 1.** This keeps your batch comparison and only loses its repeat-to-repeat variance.
2. **Drop the Qwen pair.** Llama 2 → 3.1 spans 12 months, against 7 for Qwen2.5 → 3. With the 3 GPT generations, the
   time axis survives.
3. **Drop the batch for the local models.** The GPT batch takes minutes, so keep it if there is any slack.
4. **Cap each model at 120 prompts.** run.py shuffles each model's prompts with a fixed seed, so a capped run still
   covers every part.

Never cut:
- zero-shot jokes and identity-swap pairs (Figures 1 and 2);
- HaHackathon ratings when the CSV is there (Figure 3);
- the hand-check sheets, which are the only check on the labeler and the refusal rule.

## How to apply each option on the laptop

- **B:** in `prompts\prompts.yaml`, under `batch:`, set `repeats: 1` before the batch step starts. This is the only
  prompts.yaml change allowed, and it changes no prompt text. run.py may print a "prompts changed" warning, which is
  expected.
- **C:** in `configs\models.yaml`, set `enabled: false` on `qwen2.5-7b` and `qwen3-8b`. Config changes don't trigger
  the warning.
- **D:** remove the batch steps from run_all.ps1, and add `--limit 120` to the single-call `run.py` calls.

## Speed-ups that cost nothing (do these in every option)

- **Run OpenAI and local in parallel.** Run the OpenAI steps (single-call, then batch) as their own background job
  next to the local steps, since they use the network and the GPU respectively. That saves ~35 min. Labeling waits
  for both.
- **Hand-check during the batch run.** Start it as soon as `results\pilot1\hand_labels.csv` exists, while the batch
  condition runs.
- **Skip the manual skim of the 60 HaHackathon jokes** (README step 5), or let the laptop session do it, and list it
  as a limit.
- **Llama 2 batch context:** cap it at its native 4,096 tokens. At the batch default of 8,192, its KV cache
  (no grouped-query attention) overflows 8 GB VRAM, so layers spill to the CPU. It also runs past its training
  length. This is `max_ctx: 4096` in models.yaml plus one line in `src/common.py` `chat()`, changed in the cloud
  copy on 4 Oct; the laptop copy needs the same edit.
