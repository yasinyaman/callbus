"""Paired tests for two runs over the same queries (``toolrank compare --paired``).

A runs file (``toolrank eval --runs-out``) carries one row per query, so two scorers are compared
query by query — the pairing removes the between-query variance a two-sample test would drown in.
P@1 and hit@5 are per-query 0/1, so they get the sign test; NDCG@10 is graded, so it gets a paired
permutation (sign-flip) test on the per-query differences: exact over all 2**n assignments up to
12 queries, a seeded Monte Carlo above. ToolRet's category averages weigh queries unequally, so they
get the same test on a weighted sum (``weighted_permutation_test``, ``category_weights``).
"""

from __future__ import annotations

import itertools
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

EXACT_UP_TO = 12  # 2**12 = 4096 assignments: cheaper to be exact than to sample


def sign_test(a: Sequence[float], b: Sequence[float]) -> tuple[int, int, float]:
    """Wins of ``a`` over ``b``, losses, and the two-sided exact sign-test p-value.

    Ties carry no information and drop out; under the null the wins are binomial(1/2) over what
    is left.
    """
    if len(a) != len(b):
        raise ValueError(f"paired rows differ: {len(a)} vs {len(b)}")
    wins = sum(1 for x, y in zip(a, b, strict=True) if x > y)
    losses = sum(1 for x, y in zip(a, b, strict=True) if x < y)
    n = wins + losses
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2**n
    return wins, losses, min(1.0, 2 * tail)


def permutation_test(a: Sequence[float], b: Sequence[float], *, perms: int = 10000, seed: int = 0) -> float:
    """Two-sided paired permutation p-value on the mean difference ``mean(a) - mean(b)``.

    Each query's difference keeps its size; the null says its sign is chance. The Monte Carlo
    counts the observed assignment too (the +1 correction), so p never reports 0.
    """
    if len(a) != len(b):
        raise ValueError(f"paired rows differ: {len(a)} vs {len(b)}")
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)
    if n == 0:
        return 1.0
    obs = abs(sum(d))
    eps = 1e-9 * max(1.0, obs)  # sign flips reorder the additions; floats need slack
    if n <= EXACT_UP_TO:
        reach = sum(
            1
            for bits in range(1 << n)
            if abs(sum(x if bits >> i & 1 else -x for i, x in enumerate(d))) >= obs - eps
        )
        return reach / (1 << n)
    rng = random.Random(seed)
    reach = 1  # the observed assignment
    for _ in range(perms):
        reach += abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs - eps
    return reach / (perms + 1)


def weighted_permutation_test(
    a: Sequence[float], b: Sequence[float], weights: Sequence[float], *, perms: int = 10000, seed: int = 0
) -> float:
    """Two-sided paired permutation p-value on ``sum(w * (a - b))``, a weighted mean difference.

    The sign-flip null of ``permutation_test`` with each query's difference scaled by its weight;
    queries of weight 0 drop out. Exact up to ``EXACT_UP_TO`` weighted queries, else a seeded Monte
    Carlo (numpy, in chunks) that counts the observed assignment too.
    """
    if not len(a) == len(b) == len(weights):
        raise ValueError(f"paired rows differ: {len(a)}, {len(b)}, {len(weights)} weights")
    w = np.asarray(weights, dtype=float)
    x = (w * (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))[w > 0]
    n = len(x)
    if n == 0:
        return 1.0
    obs = abs(float(x.sum()))
    eps = 1e-9 * max(float(np.abs(x).sum()), 1e-12)  # sign flips reorder the additions
    if n <= EXACT_UP_TO:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=n)))
        return float((np.abs(signs @ x) >= obs - eps).mean())
    rng = np.random.default_rng(seed)
    reach, done = 1, 0  # the observed assignment
    while done < perms:
        m = min(1000, perms - done)
        signs = rng.integers(0, 2, size=(m, n)) * 2.0 - 1.0
        reach += int((np.abs(signs @ x) >= obs - eps).sum())
        done += m
    return reach / (perms + 1)


def category_weights(groups: Mapping[str, tuple[str, str]], ids: Sequence[str]) -> dict[str, list[float]]:
    """Per-query weights that turn a weighted sum into ToolRet's category averages over ``ids``.

    ``groups`` maps a query id to its (task, category). A category's average is the plain mean over
    its tasks of each task's mean, so a query weighs 1 / (its task's size x its category's task
    count) there; ``"cat-macro"``, the plain mean over categories, divides that by the number of
    categories (``eval.metrics.category_macro``). Each list sums to 1 and follows ``ids``.
    """
    pairs = [groups.get(i, ("all", "")) for i in ids]
    size = Counter(task for task, _ in pairs)
    tasks = Counter(cat for _, cat in set(pairs))
    out: dict[str, list[float]] = {}
    for c in sorted(tasks):
        out[c] = [1.0 / (size[t] * tasks[c]) if cat == c else 0.0 for t, cat in pairs]
    out["cat-macro"] = [1.0 / (size[t] * tasks[cat] * len(tasks)) for t, cat in pairs]
    return out
