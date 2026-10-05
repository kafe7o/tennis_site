"""
„xG“ на тениса: Elo, което брои и преднината (дял спечелени геймове), не само победата.
Във футбола майсторът иска „половин голове, половин xG“ за петте големи лиги - подобрение на модела
без пазара, а той е модел, от който идва процентът на съвета.

ПРОТОКОЛ (записан преди пускането):
  Данни: tennis-data.co.uk 2012-2026, всеки тур поотделно, модел БЕЗ пазара (stack_nomkt).
  Мярка: лог-загуба на модела на мачовете с цени. Кандидати: margin_weight 0, .25, .5, .75, 1.
  Избор: САМО по мачовете от 2015 до 2021 включително (по-ниска лог-загуба). Оценка на избора: ЕДИН път на
  2022-2026, срещу margin_weight = 0 на същите мачове, със сдвоен t-тест.
  Ако изборът не е по-добър в чистата проверка, остава 0 (без промяна).
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import backtest, dataset, elo, loaders, stack      # noqa: E402

GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
TOURS = {"man": "ATP", "woman": "WTA"}
SELECT = (2015, 2021)
FIRST_TEST = 2022


def matches(tour):
    folder = ROOT.parent / "data" / tour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [loaders.tennis_data(p, TOURS[tour]) for p in sorted(folder.glob("*"))
                  if p.suffix.lower() in (".xls", ".xlsx", ".csv")]
    return pd.concat(frames, ignore_index=True)


def loss(df, col):
    d = df[df[col].notna() & df["mkt_p1"].notna()]
    y = d["y"].to_numpy(float)
    p = np.clip(d[col].to_numpy(float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def main():
    for tour in TOURS:
        m = matches(tour)
        print(f"\n######## {tour} ########")
        runs = {}
        for w in GRID:
            df = dataset.build(m, elo.EloParams(margin_weight=w))
            df["own"] = stack.walk_forward(df, with_market=False, first_year=SELECT[0])
            runs[w] = df
            sel = df[df["year"].between(*SELECT)]
            print(f"  margin_weight {w:.2f}: лог-загуба избор {loss(sel, 'own').mean():.5f}"
                  f"  | 2022-2026 {loss(df[df['year'] >= FIRST_TEST], 'own').mean():.5f}")
        best = min(GRID, key=lambda w: loss(runs[w][runs[w]['year'].between(*SELECT)], "own").mean())
        a = runs[best][runs[best]["year"] >= FIRST_TEST]
        b = runs[0.0][runs[0.0]["year"] >= FIRST_TEST]
        ok = a["own"].notna() & b["own"].notna() & a["mkt_p1"].notna()
        diff = loss(a[ok], "own") - loss(b[ok], "own")
        t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        print(f"  ИЗБРАНО по избора: margin_weight={best}. Чиста проверка 2022-2026 срещу 0: разлика {diff.mean():+.5f} "
              f"(t {t:+.2f}; отрицателно = по-добре), мачове {len(diff)}")
        mk = backtest.compare(a[ok], ["own"])
        print(f"  спрямо пазара (чиста проверка): лог-загуба {mk.loc['own', 'logloss']:.4f} срещу пазара {mk.loc['mkt_p1', 'logloss']:.4f}")


if __name__ == "__main__":
    main()
