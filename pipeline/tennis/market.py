"""
Пазарът без маржа. В тениса има два изхода (няма равен), което прави нещата ясни:
коефициентите 1.80 / 2.05 дават 1/1.80 + 1/2.05 = 1.043 - тези 4.3% са печалбата на букмейкъра,
а не вероятност. Честната вероятност се получава, като маржът се махне.

Два начина:
  proportional  просто делене на сбора - маржът се разпределя поравно (по-слабо точно)
  power         степен k >= 1, за която a^k + b^k = 1. Маржът пада върху аутсайдера повече,
                тоест хваща favourite-longshot bias: аутсайдерите се залагат прекалено
                и букмейкърът ги цени по-ниско от честното. По подразбиране е този.
"""

import numpy as np


def implied(odds):
    return 1.0 / np.asarray(odds, dtype=float)


def margin(o1, o2):
    return implied(o1) + implied(o2) - 1.0


def devig(o1, o2, method="power"):
    """Честната вероятност играч 1 да победи. Приема числа или масиви."""
    a, b = implied(o1), implied(o2)
    if method == "proportional":
        out = a / (a + b)
    elif method == "power":
        lo = np.ones_like(a + b)
        hi = np.full_like(lo, 30.0)
        for _ in range(60):                       # бисекция: a^k + b^k намалява с k
            mid = (lo + hi) / 2.0
            above = a ** mid + b ** mid > 1.0
            lo = np.where(above, mid, lo)
            hi = np.where(above, hi, mid)
        k = (lo + hi) / 2.0
        ak, bk = a ** k, b ** k
        out = ak / (ak + bk)
    else:
        raise ValueError(f"непознат метод: {method}")
    return out if np.ndim(out) else float(out)


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))
