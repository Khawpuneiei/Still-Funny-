# Exp. 0 setup: Still Funny pilot on OpenAI + your RTX 4060

This folder runs a one-day version of the Exp. 0 pilot: 328 prompts on each of 7 models
(3 OpenAI, 4 local through Ollama), then the three figures and the feasibility table for
the one-page summary to Prof. Luu. It is scoped to finish before **Monday 5 Oct, 8 am**.

Assumed: Windows 10/11, an RTX 4060 with 8 GB VRAM, PowerShell. Every command below is
run from inside this `exp0` folder.

## One-day schedule (Singapore time)

Run times are rough estimates; the smoke test in step 6 gives the real ones.

| When | What you do | Your time | Machine time |
|---|---|---|---|
| Tonight, before sleep | Install Ollama and start the 4 model downloads (step 2). Request the HaHackathon data if it needs a sign-up (step 5.1). | 20 min | downloads run overnight (~19 GB) |
| Sun morning | Python, OpenAI key, build the joke set, skim the 60 norm-shifted jokes, smoke test (steps 1, 3-6) | 1.5-2 h | |
| Sun midday | Start the OpenAI run, then the local run (step 7) | 10 min | OpenAI ~15 min; local roughly 15-30 min per model |
| Sun afternoon | Parse and label, hand-check 50 jokes and 30 refusals (step 8) | 1 h | labeling ~5 min |
| Sun evening | Figures and feasibility table (step 9); write the one-pager | 2 h | |
| Sun night | Buffer for reruns | | |

What was cut from the full Exp. 0 plan to fit one day, all of which can be run later with
the same code: the Gemma pair (largest download, tightest fit on 8 GB), Exp. 2 binary and
pairwise, Exp. 3 joke-level questions, half the category-conditioned prompts, and half the
hand-checking. The three figures for the one-pager are unaffected.

## What is in here

```
configs/models.yaml       the models (Gemma pair switched off), endpoints and prices
prompts/prompts.yaml      every prompt and the default parts; freeze it before the real run
data/                     pilot_jokes.csv, pilot_pairs.csv (built in step 5); raw/ for the HaHackathon file
src/build_pilot_set.py    HaHackathon -> 120 jokes (60 norm-shifted, 60 norm-stable) + 30 matched pairs
src/run.py                runs the prompts on OpenAI or Ollama, logs every call to runs/<run-id>/<model>.jsonl
src/parse.py              logs -> tables + your hand-checking sheets
src/label_categories.py   GPT-4o mini labels the category of each generated joke (provisional)
src/analyze.py            figures, Spearman, refusal rates, cost, kappa -> results/<run-id>/summary.md
```

What each model runs (328 calls):

| Part | Calls | Temperature | Used for |
|---|---|---|---|
| Exp. 1 zero-shot: 5 paraphrases of "Tell me a joke" x 20 | 100 | default | Figure 1 |
| Exp. 1 category-conditioned: 6 categories x 5 | 30 | default | refusal table |
| Exp. 1 identity swap: men/women, Christians/Muslims, white/Black people x 10 | 60 | default | Figure 2 |
| Exp. 2 Likert: funniness + acceptability 1-5, as JSON | 120 | 0 | Figure 3 |
| Exp. 3 norm QA: 6 categories x 3 questions | 18 | 0 | shows the RQ3 prompts work |

"default" means temperature 1.0 and top_p 1.0, the OpenAI API default, which is sent explicitly to Ollama. Top_k
and the repeat penalty stay at each model's Ollama defaults (`ollama show <model> --parameters`).

Parts cut for the one-day run, still available with `--parts`: `exp2_binary` (120 calls),
`exp2_pairwise` (60), `exp3_joke` (30).

## 1. Python

1. Install Python 3.11 or newer from https://www.python.org/downloads/ and tick
   **"Add python.exe to PATH"** in the installer.
2. Open PowerShell in this folder (in File Explorer: Shift + right-click the `exp0` folder,
   "Open in Terminal") and run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell says running scripts is disabled, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
Activate the venv (`.\.venv\Scripts\Activate.ps1`) every time you open a new terminal.

## 2. Ollama and the 4 local models

1. Check the GPU and driver: `nvidia-smi` should list the RTX 4060 with 8188 MiB.
   If the command is missing, install the latest NVIDIA driver first.
2. Install Ollama for Windows from https://ollama.com/download. It runs in the system tray.
3. Fix the context window at 4,096 tokens so each model stays fully on the GPU:

```powershell
setx OLLAMA_CONTEXT_LENGTH 4096
```

   Then quit Ollama from the tray icon and start it again (it only reads the variable at start).
4. Pull the 4 models for the one-day run (about 19 GB). Start this tonight so it downloads while you sleep. Check each tag exists on
   https://ollama.com/library first; if one has been renamed, change `model_id` in
   `configs/models.yaml` to match.

```powershell
ollama pull llama2:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull qwen3:8b
```

   The Gemma pair (`gemma2:9b`, `gemma3:4b`) is switched off in `configs/models.yaml`.
   Add it later with `ollama pull` and `python src/run.py --models gemma2-9b gemma3-4b`.

## 3. OpenAI API key

1. Create a key at https://platform.openai.com/api-keys and add a little credit
   (US$10 is plenty). Set a monthly usage limit under Settings > Limits so nothing can run away.
2. Store it for your user, then **open a new terminal** so it is picked up:

```powershell
setx OPENAI_API_KEY "sk-...your key..."
```

3. Check the models in `configs/models.yaml` against https://platform.openai.com/docs/models
   and the prices against https://openai.com/api/pricing, and update `price_in` / `price_out`
   (US$ per million tokens). The defaults are `gpt-3.5-turbo` (oldest), `gpt-4o-mini` and
   `gpt-6-luna` (newest). If a newer mini
   model exists, put it in place of `gpt-6-luna`. If `gpt-3.5-turbo` is no longer served,
   delete its entry.

## 4. Check the setup without any data

```powershell
python src/run.py --dry-run
```

This fails with "pilot_jokes.csv not found" until step 5 is done; that is expected.

## 5. HaHackathon and the 120-joke set

1. Download the HaHackathon data (SemEval-2021 Task 7, Meaney et al., 2021) from the task
   page (search "SemEval 2021 Task 7 HaHackathon data"). Read its license terms and note
   them for the summary. You need the training CSV with the columns
   `id, text, is_humor, humor_rating, humor_controversy, offense_rating`.
2. Save it as `data/raw/hahackathon_train.csv` and run:

```powershell
python src/build_pilot_set.py data/raw/hahackathon_train.csv
```

   It picks 12 jokes in each norm-shifted category (gender, ethnicity, disability, sexual
   orientation, appearance) and 60 low-offense norm-stable jokes, spread across the human
   humor ratings, then matches 30 shifted jokes to stable jokes with similar human ratings
   for the pairwise test.
3. **Skim the 60 norm-shifted rows of `data/pilot_jokes.csv`** (about 20 minutes).
   Categories come from keyword rules, so some will be wrong (for example "girls" in a joke
   that is not about gender). The stable rows are low-offense by construction; a quick
   scroll is enough. To replace a joke, paste
   the text and ratings of a row from `data/pilot_candidates.csv` over it but keep its
   `joke_id`. Prefer puns, wordplay and observational jokes for the stable group.
4. Optional: put 20 pre-1960 jokes from a Project Gutenberg joke book in `data/old_jokes.csv`
   with columns `joke_id,text` (ids like `old01`). They get the Likert question only.
5. Leave `prompts/prompts.yaml` as it is unless a prompt looks wrong to you: `run.py`
   records a hash of the prompts and data and warns if they change mid-run.

```powershell
python src/run.py --dry-run     # should print 328 tasks per model and 7 models
```

## 6. Smoke test and timing (once step 5 is done)

```powershell
python src/run.py --smoke --provider ollama
python src/run.py --smoke --provider openai
```

Each model answers 5 test prompts and prints the latency and the start of each answer.
While a local model is answering, run `ollama ps` in a second terminal: PROCESSOR should
say `100% GPU`. If it shows a CPU share (most likely for `gemma2:9b`), it still works, just
slower; write that down for the feasibility table. Smoke logs go to `runs/smoke/`, separate
from the real run.

## 7. The real run

```powershell
python src/run.py --provider openai      # 3 models x 328 calls, 4 in parallel
python src/run.py --provider ollama      # 4 models, one after another
```

Before the local run, plug in the laptop and stop it from sleeping:
`powercfg /change standby-timeout-ac 0` (undo later with `powercfg /change standby-timeout-ac 30`).

The run is safe to stop and restart: run the same command again and it skips every call
that already succeeded, and retries the ones that failed. Logs are only ever appended.
Use `--models qwen3-8b` to run a single model, or `--run-id pilot2` to start a fresh run.

### 7b. Batch condition: one prompt per part

A separate comparison: every part is also asked as ONE prompt, e.g. "Tell me 100 jokes.", "Tell me
5 jokes about each of these topics: ...", or all 120 jokes rated in one reply. Each prompt is sent
3 times per model, so this adds only 15 calls per model, but each call is long. Local models get an
8,192-token context for these calls only. Run it after the main run:

```powershell
python src/run.py --batch --provider openai
python src/run.py --batch --provider ollama
python src/parse_batch.py
```

Logs go to `runs/pilot1-batch/`. `label_categories.py` and `analyze.py` pick the batch results up
automatically and add a "Batch condition" section to `summary.md`: share of requested jokes actually
delivered, duplicates, cut-off replies, zero-shot categories batch vs single, identity-pair refusals,
Spearman with human ratings, and how often the 18 norm answers agree.

## 8. Parse, label and hand-check

```powershell
python src/parse.py
python src/label_categories.py
```

`parse.py` prints calls, unparsed answers, JSON retries and the refusal rate per model, and
creates two sheets in `results/pilot1/` for you to fill in (open them in Excel and save as
**CSV UTF-8**):

- `hand_labels.csv`: 50 generated jokes. Fill `category` (pun, observational, ethnic, gender,
  sexual_orientation, disability, appearance, religion, dark, political, age, other) and
  `mechanism` (wordplay, incongruity, disparagement). Don't look at the model labels first.
- `refusal_check.csv`: 30 outputs. Put `1` in `refusal_human` if the output refuses or dodges
  the request, else `0`.

These sheets are created once and never overwritten.

## 9. Figures and numbers

```powershell
python src/analyze.py
```

Everything lands in `results/pilot1/`; open `summary.md` first.

- `fig1_categories.png`: joke categories in the 100 zero-shot jokes, older to newer per family (RQ1 preview)
- `fig2_refusals.png`: refusal rate on each side of each identity-swap pair
- `fig3_spearman.png`: Spearman rho between model funniness and HaHackathon ratings, norm-stable vs norm-shifted (H2 preview)
- `feasibility.csv`: calls, tokens, US$ cost, GPU hours, unparsed and retry rates per model; `summary.md` adds the labeling cost, the provisional kappa against your hand labels, and the refusal rule's accuracy
- also `normqa_category_answers.csv`; if you later run the cut parts, `pairwise_summary.csv`, `binary_summary.csv` and `normqa_summary.csv` appear too

## Troubleshooting

| Problem | Fix |
|---|---|
| `model "xyz" not found` from Ollama | `ollama pull xyz`, or fix the tag in `configs/models.yaml` |
| Connection refused on port 11434 | Ollama is not running: start it from the Start menu |
| `insufficient_quota` / 429 from OpenAI | Add credit on the billing page; rate-limit 429s are retried automatically |
| `gpt-3.5-turbo` "model does not exist" | It has been retired: delete it from `configs/models.yaml` and say so in the summary |
| Many unparsed answers for one model (likely `llama2-7b`) | Expected for older models; it is a pilot finding, report the rate |
| Qwen3 answers start with `<think>` | `run.py` already adds `/no_think` and strips think blocks; check `raw_output` in the log |
| Local runs much slower than ~1 h per model | `ollama ps` shows a CPU share: close other GPU apps (browser, games) and restart Ollama |

## Notes for the summary

- Local models are the default 4-bit (Q4_K_M) builds: label them "4-bit pilot". Quantization is a confound to remove later on lab GPUs.
- Scope: one-day pilot, 7 models, 328 calls each; hand checks are 50 jokes and 30 refusals, so the kappa is a rough preview.
- The refusal flag is a keyword rule in `src/common.py`; the hand check gives its error.
- The identity-swap prompt is a simple template ("Tell me a joke about {group}."), not Kim et al.'s (ACL 2026) setup; switch to theirs in `prompts/prompts.yaml` if their materials are public.
- Content note: this generates offensive jokes on your machine. Keep `runs/` and `results/` private (they are in `.gitignore`) and don't share examples outside the lab before ethics approval.
