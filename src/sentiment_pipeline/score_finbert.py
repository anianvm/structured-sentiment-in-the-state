"""Score the SAME headlines as the LLM with FinBERT (classical baseline).

Ported unchanged from master_thesis/src/sentiment/finbert_score.py
(methodology §FinBERT sentiment extraction); only paths, imports and module
names were adapted (torch/transformers are imported lazily inside main).

FinBERT (ProsusAI/finbert) is a discriminative classifier, so it carries no
look-ahead bias and needs no anonymisation for that reason. We still feed it the
identical cleaned + anonymised text the LLM saw, so the only variable in the
LLM-vs-FinBERT comparison is the scoring model itself. A --no-anon run scores the
cleaned-but-named text to measure how much anonymisation costs FinBERT.

Output (standalone, mirrors structured_scores.csv):
  date, ticker, headline, p_pos, p_neg, p_neu, finbert_score, finbert_label
  finbert_score = p_pos - p_neg  in [-1, 1]   (continuous -> RL grid + correlation)
  finbert_label = argmax class                (categorical -> confusion matrix)

Run:  python -m src.sentiment_pipeline.score_finbert            # anonymised
      python -m src.sentiment_pipeline.score_finbert --no-anon  # named text
"""

import argparse

import numpy as np
import pandas as pd

from src import config
from src.sentiment_pipeline.score_llm import anonymise, clean_headline

MODEL = "ProsusAI/finbert"
SRC = config.SENTIMENT_DIR / "selected_headlines.csv"  # from select_headlines.py
# Not in config: robustness-only output of the --no-anon run.
FINBERT_NAMED_SCORES_FILE = config.SENTIMENT_DIR / "finbert_scores_named.csv"


def main() -> None:
    import torch  # lazy: only needed to re-score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--no-anon", action="store_true", dest="no_anon",
                    help="score cleaned-but-named text (robustness variant)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    out = (FINBERT_NAMED_SCORES_FILE if args.no_anon
           else config.FINBERT_SCORES_FILE)

    df = pd.read_csv(SRC)
    # identical input pipeline to the LLM (anonymise unless --no-anon)
    if args.no_anon:
        texts = [clean_headline(h) for h in df["headline"]]
    else:
        texts = [anonymise(clean_headline(h), t)
                 for h, t in zip(df["headline"], df["ticker"])]

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device).eval()

    # read class order from the model rather than hard-coding it
    id2label = {i: l.lower() for i, l in model.config.id2label.items()}
    pos_i = next(i for i, l in id2label.items() if l == "positive")
    neg_i = next(i for i, l in id2label.items() if l == "negative")
    neu_i = next(i for i, l in id2label.items() if l == "neutral")
    print(f"model={MODEL}  device={device}  anon={not args.no_anon}  "
          f"rows={len(texts):,}  id2label={id2label}")

    probs = np.zeros((len(texts), 3), dtype=np.float32)  # cols: pos, neg, neu
    with torch.no_grad():
        for s in range(0, len(texts), args.batch):
            batch = texts[s:s + args.batch]
            enc = tok(batch, padding=True, truncation=True, max_length=128,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits
            p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs[s:s + len(batch), 0] = p[:, pos_i]
            probs[s:s + len(batch), 1] = p[:, neg_i]
            probs[s:s + len(batch), 2] = p[:, neu_i]
            if (s // args.batch) % 200 == 0:
                print(f"  {s:>7d}/{len(texts)}")

    res = pd.DataFrame({
        "date": df["date"], "ticker": df["ticker"], "headline": texts,
        "p_pos": probs[:, 0], "p_neg": probs[:, 1], "p_neu": probs[:, 2],
    })
    res["finbert_score"] = (res["p_pos"] - res["p_neg"]).round(4)
    argmax = probs.argmax(axis=1)  # 0=pos,1=neg,2=neu (our column order)
    res["finbert_label"] = np.array(["positive", "negative", "neutral"])[argmax]
    for c in ("p_pos", "p_neg", "p_neu"):
        res[c] = res[c].round(4)
    res.to_csv(out, index=False)

    n = len(res)
    print(f"\nsaved {n:,} rows -> {out}")
    print(f"label mix : pos {100*(res.finbert_label=='positive').mean():.1f}%  "
          f"neu {100*(res.finbert_label=='neutral').mean():.1f}%  "
          f"neg {100*(res.finbert_label=='negative').mean():.1f}%")
    print(f"score     : mean {res.finbert_score.mean():+.3f}  "
          f"sd {res.finbert_score.std():.3f}")


if __name__ == "__main__":
    main()
