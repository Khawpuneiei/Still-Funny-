# HaHackathon data: where to get it and where it goes

Checked 4 Oct 2026, 00:20 UTC.

## What you need

The training file `train.csv` from SemEval-2021 Task 7 (HaHackathon). It has 8,000 texts and the columns
`id, text, is_humor, humor_rating, humor_controversy, offense_rating`.
Each text was rated by 20 people in four age groups (18-25, 26-40, 41-55, 56-70).

## Download, no sign-up

**https://smash.inf.ed.ac.uk/hahackathon_data/hahackathon_data.zip** (558 KB, updated 23 Sep 2026)

- This is the "Download the data" link on the organisers' own group page at the University of Edinburgh,
  https://smash.inf.ed.ac.uk/resources/. The page asks for no registration and has no form.
- I couldn't open the zip from the cloud, because the host is blocked from here. So I haven't seen the file names inside.
  The competition forum calls the training file `train.csv`. If the names differ, use the CSV with 8,000 rows and the
  columns above.
- The original source is the CodaLab competition page, https://competitions.codalab.org/competitions/27446. It needs
  a CodaLab login and joining the competition, which accepts its terms. The direct link above makes that unnecessary.
- No Hugging Face mirror turned up, and one isn't needed.

## Terms of use (your decision)

Neither page states a license. The task paper says:

> "The texts and annotations will continue to be available on the Codalab website, and the tweet ids, and
> usernames will be retained for non-commercial research use, in line with the Twitter Academic Developer Policy."

My reading, which you should confirm:
- Treat it as non-commercial research use only.
- Part of the texts are tweets, so don't republish them in bulk. Quote at most a few examples in a paper, and none in
  the one-pager. The rest come from the Kaggle Short Jokes dataset.
- Exp. 0 fits this: it rates 120 of the texts with LLMs, keeps all raw outputs on your laptop, and reports only aggregates.

## Where it goes on the laptop

Rename `train.csv` and save it as:

```
Desktop\Work\Still_Funny\exp0\exp0\data\raw\hahackathon_train.csv
```

run_all.ps1 checks that exact path. With the file there, it runs `python src/build_pilot_set.py
data/raw/hahackathon_train.csv`, which writes `data/pilot_jokes.csv` (120 jokes) and `data/pilot_pairs.csv`. That
unlocks the joke-rating part (exp2_likert) and Figure 3.

If you reply "ok" in the thread, the laptop session can download and place it for you.

## Citation

Meaney, J. A., Wilson, S. R., Chiruzzo, L., Lopez, A., & Magdy, W. (2021). SemEval-2021 Task 7: HaHackathon,
Detecting and Rating Humor and Offense. In *Proceedings of the 15th International Workshop on Semantic Evaluation
(SemEval-2021)*, pp. 105-119. https://aclanthology.org/2021.semeval-1.9
