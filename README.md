# leaky-vs-honest-splits-toy
How homology-aware train/test splitting affects apparent protein ML benchmark performance

A small, reproducible demonstration of why protein property-prediction models
often look more accurate than they are. So, the way you split your data
determines which question you're really asking.

## The problem

Protein sequences are not independent samples. Evolution makes many proteins
near-copies of each other (homologs). If a random train/test split lets near-copies
land on both sides, a model can score well simply by recognizing relatives of its
training examples — not by learning anything that generalizes to a genuinely new
protein. The reported accuracy is then an artifact of leakage, not of capability.

This matters for anyone validating a model against experimental ground truth: a
benchmark that looks 90% accurate under a random split may perform far worse on the
protein actually on your bench.

## What this repo shows

A controlled toy experiment isolates the effect. I generate synthetic "proteins"
organized into families of near-copies, build a target with a strong family-specific
component (memorizable) and a weak generalizable component, then train an identical
Ridge regression model under two split strategies:

- **Leaky (random):** individuals shuffled at random; near-copies leak across the split.
- **Honest (by family):** whole families held out; the model is tested only on
  families it has never seen.

**Result:** identical model, identical data, identical features — only the split differs.

| Split            | Spearman |
|------------------|----------|
| Leaky (random)   | 0.94     |
| Honest (by family)| 0.38    |

The ~0.56 gap is pure leakage inflation: the model was largely memorizing family
identity, which the honest split makes invisible.

## Why a toy first

Controlling the homology structure lets the leakage effect be seen directly and
explained mechanistically. The next step (in progress) reproduces the same signature
on real data — the FLIP Meltome thermostability benchmark — comparing FLIP's
MMseqs2-clustered split against a naive random split.

## Background

The split-strategy problem is formalized for proteins in Dallago et al., *FLIP:
Benchmark tasks in fitness landscape inference for proteins* (NeurIPS 2021), which
reports that random ("sampled") splits substantially overestimate model performance
relative to biologically motivated splits.

## Run it

Open [`leaky_vs_honest_toymodel.ipynb`](leaky_vs_honest_toymodel.ipynb) in Google Colab and run top to bottom. No setup required.

New to Python or coming from the wet lab? See [`in_plain_english.md`](in_plain_english.md) for a line-by-line walkthrough.
