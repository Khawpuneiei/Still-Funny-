# Still Funny? Exp. 0 pilot: first results

Panjapong Poobanchuen · NTU CCDS (exchange) · {{DATE}} · pilot for "Still Funny? Language Models, Jokes, and Shifting Social Norms"

**Question.** Do language models' jokes and joke judgments follow the way social norms about humour have shifted?
Exp. 0 is a one-day pilot. It tests the whole pipeline across model generations before the main experiments, and gives
a first look at three of the signals the study will measure.

**Setup.** {{N_MODELS}} models, older to newer within each family:
- {{GPT_LINE: GPT-3.5 Turbo → GPT-4o mini → GPT-6 Luna}} through the OpenAI API;
- Llama 2 7B → Llama 3.1 8B and Qwen2.5 7B → Qwen3 8B, open weights, 4-bit, on one RTX 4060 laptop GPU.

Each model answered {{CALLS_PER_MODEL}} separate prompts:
- 100 open requests for a joke (5 paraphrases × 20) and 30 requests on given topics;
- 60 identity-swapped requests (men/women, Christians/Muslims, white/Black people);
- {{IF_DATA: 120 HaHackathon jokes rated for funniness and acceptability, 60 on norm-shifted topics and 60 norm-stable;}}
- 18 questions on whether joke categories are acceptable today.

Settings were temperature 0 with a fixed seed where the API allows, and every call is logged.
A second condition sent each part as one prompt ("Tell me 100 jokes."), 3 repeats per model.

**1. What models joke about.** {{FIG1 = fig1_categories.png, small}}
{{ONE SENTENCE. Source: category_distribution.csv. Compare older vs newer within each family on the
"pun / wordplay" and "identity-targeted" shares and the refusal share. Example shape: "Asked for 'a joke', newer
models tell more puns (Llama 2 → 3.1: A% → B%) and fewer identity-targeted jokes (C% → D%)." This previews H1.}}

**2. Refusals depend on which group is named.** {{FIG2 = fig2_refusals.png, small}}
{{ONE SENTENCE. Source: refusal_by_side.csv. State the largest within-pair gap and which model shows it, plus whether
the gap grows or shrinks across generations. Example shape: "GPT-4o mini declined X% of jokes about women and Y%
about men; Llama 2 declined both at Z%." Refusal is measured by a keyword rule checked by hand on 30 outputs.}}

**3. Agreement with human ratings.** {{FIG3 = fig3_spearman.png, small; OR, if HaHackathon was not available:
"Pending: needs the HaHackathon ratings (data request in progress)."}}
{{ONE SENTENCE. Source: spearman.csv. Spearman rho between model funniness and human humour ratings, norm-stable vs
norm-shifted jokes (n = 60 each). Example shape: "rho ranges from a to b on norm-stable jokes and c to d on
norm-shifted ones." H2 predicts lower agreement on the shifted set. With n = 60, say the interval is wide.}}

**4. One prompt per item, or many items per prompt?** {{ONE OR TWO SENTENCES. Sources: batch_delivery.csv,
batch_vs_single_categories.csv, batch_vs_single_refusals.csv, batch_vs_single_spearman.csv. Report how much of
what was asked for came back (e.g. "local models returned X-Y of 100 jokes"), the repetition (unique_share), and
whether the category mix or the refusal pattern differs from the one-per-call condition. The design point is
whether batching, which is far cheaper, can stand in for one call per item in the full study.}}

**Feasibility.**

| Model | Calls | Unparsed | Cost (US$) | Model time (h) |
|---|---|---|---|---|
| {{one row per model from feasibility.csv: calls, unparsed_rate, cost_usd, hours}} | | | | |

{{TOTALS: "Total API cost US$X; local GPU time Y h on one RTX 4060 (8 GB)."}}
{{QUALITY: "Category labeler vs my hand labels: kappa = K (50 jokes); refusal rule accuracy A (30 outputs)." Sources:
summary.md notes, after the hand-check sheets are filled.}}

**What this means for the full study.**
- The pipeline runs end to end on both APIs and local models. {{One clause on the measured parse failure rate.}}
- At the measured cost per call, the full Exp. 1-3 design would cost about US${{SCALE_COST}} in API fees per closed model.
  That is about 5,000 calls per model ({{show the arithmetic: cost per call × 5,000}}).
- Open models above ~10B (and the 70B models) don't fit in 8 GB and need lab GPUs.
- {{One sentence on the most promising signal to test properly at scale.}}

**Limits.** This is a pilot:
- small n per cell (at most 100);
- one seed;
- 4-bit local models;
- keyword-based joke categories and refusal rule, with a provisional LLM labeler checked against {{N_HAND}} hand labels;
- no new human data. The human reference is HaHackathon (Meaney et al., 2021).

---

## Fill-in notes (delete this section before sending)

- Keep it to one page. If it runs long, cut finding 4 to one sentence first, then shrink the feasibility table to the
  totals line.
- Quote no joke texts. Part of HaHackathon is tweets (see hahackathon_data.md), and the generated jokes can be offensive.
- All numbers come from the laptop's `exp0\share\` copy of `results\pilot1\` (summary.md and the CSVs named above).
  Round shares to whole percent and rho to 2 decimals. Give n next to every rate.
- If a GPT model failed (for example gpt-6-luna not reachable), drop it from GPT_LINE and N_MODELS and say so in Setup.
- CALLS_PER_MODEL is 328 with HaHackathon or 208 without. N_MODELS is 7 if all ran.
- Figures: use the PNGs as they are (palette already set). In the final Claude Doc, put the three figures side by side in one row.
- Setup line (4 Oct run): all joke generation sampled at temperature 1.0 and top_p 1.0 (OpenAI API default, sent
  explicitly to Ollama). Top_k and the repeat penalty were left at Ollama defaults; only Qwen3 differs (top_k 20, no
  repeat penalty). Local models ran on Ollama's Vulkan backend: the laptop's NVIDIA driver is too old for CUDA, and the
  model was reloaded every 20 calls to work around a RAM leak. So treat local timings in the feasibility table as
  pessimistic, and say so under Limits.
