# In Plain English

I'm a wet-lab biochemist who built this through AI-assisted coding. This document is for anyone with bench experience who wants to expand into the in-silico world but finds GitHub intimidating. It walks through the code line by line in plain language.

This toy model sets up and answers one question: **if you test a model on proteins that are near-copies of its training proteins, does it look better than it really is?** The answer is yes, and this code measures by how much.

The usual way to train a model from a dataset is to split the data randomly into training and testing sets, train on one, and use the other to check if the model has learned anything useful. 
But for proteins, random splits are misleading — proteins violate the I.I.D. (Independent and Identically Distributed) assumption because they're often evolutionarily related (homologous). 
Imagine you have a hundred protein sequences and their melting temperatures. If you split randomly, members of the same family can end up on *both* sides, and the model can lean on family identity as a shortcut. This inflates the model's apparent accuracy without actually learning what generalizes to a new family. That's a **leaky split**. 
An **honest split** keeps whole families on one side or the other, so the model is tested only on families it has never seen.

I'll walk through the code line by line in plain language.

---

## Setup

```python
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import spearmanr
```

Three tools. `numpy` handles arrays of numbers (think: it does math on whole columns
at once, like a spreadsheet). `Ridge` is a simple prediction model — a well-behaved
version of linear regression. `spearmanr` is a score that asks "did the model rank
things in the right order?" — 1.0 is perfect ranking, 0 is no better than chance.

```python
rng = np.random.default_rng(0)
n_families = 150
copies = 4
dim = 200
```

`rng` is a random-number generator with a fixed starting point (the `0`, called a
seed). Fixing the seed means the "random" numbers come out the same every run, so
my results are reproducible. The next three lines just name some quantities: 150
protein families, 4 near-copies per family (stand-in for homologs that share most of their sequence), and 200 numbers describing each protein.

That "200 numbers per protein" is standing in for a real protein embedding — the
fixed-length numerical fingerprint a model like ESM produces from an amino acid
sequence. Here they're made up; in the real version they'd be computed from sequences.

---

## Building the fake proteins

```python
family_trait = rng.normal(size=(n_families, dim))
X = np.repeat(family_trait, copies, axis=0)
X = X + rng.normal(scale=0.05, size=X.shape)
family_id = np.repeat(np.arange(n_families), copies)
```

Line 1 makes one 200-number signature for each of the 150 families — the family's
"ancestral barcode."

Line 2 stamps each family's barcode out 4 times, giving 600 proteins. At this point
the 4 copies of a family are *identical*.

Line 3 adds a tiny random wobble (scale 0.05, about twenty times smaller than the
signatures themselves) so the 4 copies stop being identical and become *near*-copies (like homologs that differ by a handful of mutations).

Line 4 keeps a label for each of the 600 proteins saying which family it belongs to:
`[0,0,0,0, 1,1,1,1, 2,2,2,2, ...]`. We'll need this to split by family later.

The capital `X` is a near-universal convention: `X` is the inputs the model sees,
lowercase `y` (coming up) is what we're trying to predict.

---

## The property we're predicting — built from two ingredients

```python
family_offset = rng.normal(scale=3.0, size=n_families)
weak_signal   = rng.normal(scale=0.3, size=dim)
y = family_offset[family_id] + X @ weak_signal + rng.normal(scale=1.0, size=len(X))
```

This is the heart of the whole thing. The property `y` (think: melting
temperature) is built from two deliberately different ingredients plus noise:

`family_offset` is one number per *family* (150 of them), and it's large (scale 3).
It's the part a model can only use if it has *seen a member of that family*. This is
the memorizable cheat.

`weak_signal` is a set of weights, one per embedding dimension (200 of them), and
it's small (scale 0.3). It's a rule that applies to *every* protein regardless of
family — the part a model could legitimately learn and apply to a brand-new protein.

The `y =` line combines them:
- `family_offset[family_id]` looks up each protein's family quirk, so all 4 siblings
  of a family share the same big number.
- `X @ weak_signal` (the `@` means matrix multiply) applies the weak rule to each
  protein's embedding, giving one number per protein.
- the last term adds measurement noise, because real lab measurements are never clean.

The design is rigged so family identity is a big, easy-to-memorize signal and the
transferable rule is a faint one. That's what lets leakage show up.

---

## The two splits

```python
n = len(X)
idx = rng.permutation(n)
test_leaky  = idx[:120]
train_leaky = idx[120:]
```

The **leaky** split shuffles all 600 proteins randomly and takes 120 for testing.
Because the shuffle ignores families, a protein's near-copies scatter freely — some
land in training, some in test. The model gets to study a protein's sibling, then is
tested on it. That's the leak.

```python
test_families = rng.permutation(n_families)[:30]
test_honest  = np.where(np.isin(family_id, test_families))[0]
train_honest = np.where(~np.isin(family_id, test_families))[0]
```

The **honest** split picks 30 whole families and puts *all* their members in the test
set. `np.isin(family_id, test_families)` marks each protein true/false for "is your
family one of the held-out 30?"; `np.where(...)[0]` turns those trues into row numbers;
the `~` flips true/false to get the training side. Because whole families go one way
or the other, the model is tested only on families it has *never* seen.

Both test sets are 120 proteins, so the comparison is fair — only the *strategy* differs.

---

## Train, predict, score

```python
def score(train, test, label):
    m = Ridge().fit(X[train], y[train])
    rho = spearmanr(m.predict(X[test]), y[test]).correlation
    print(f"{label:8s} Spearman = {rho:.3f}")
    return rho

leaky  = score(train_leaky,  test_leaky,  "LEAKY")
honest = score(train_honest, test_honest, "HONEST")
print(f"\nInflation: {leaky - honest:+.3f}")
```

`score` is a small reusable recipe: train a fresh Ridge model on the training proteins
(`X[train]`, `y[train]`), predict the test proteins, and report the Spearman score.
Writing it once and calling it twice guarantees the leaky and honest runs are identical
except for which proteins go where — so any difference in score is caused by the split
and nothing else.

The result: **LEAKY ≈ 0.94, HONEST ≈ 0.38.** Same model, same data, same features. The
0.56 gap is pure leakage. The model was mostly memorizing family identity, which the
honest split makes invisible.

---

## Why this matters at the bench

A model that reports 90%-ish accuracy on a random split might do far worse on the actual
protein in front of you, because that number was propped up by homology it had already
seen. Before trusting any prediction tool, I'd check this.
