"""
Правилата на майстора, „изкривени“ към тениса - четири варианта, избрани ПРЕДИ пускането.

  V0  както са във футбола: съвет по модела без пазара, коефициент от 1.40, сигурна <= 1.80, рискова над 1.80
  V1  V0 + „тото“: без съвет с коефициент 1.30-1.55 (потвърдено и за двата тура: фаворитите в лентата
      печелят по-рядко от обещаното и в 2012-2019, и в 2019-2026 - research/master_tennis.py)
  V2  V0 + „съгласие“: съвет само когато моделът и пазарът са за същия играч (идеята: рисковата прогноза е
      почти винаги несъгласие с пазара, а там моделът е прав само в ~40%)
  V3  V1 + V2

Данни: tennis-data.co.uk 2012-2026, всеки тур поотделно; модел БЕЗ пазара (stack_nomkt, Elo с margin_weight 0.75).
ОЦЕНКА: само чистият период 2022-2026 (моделът се учи всяка година от минали години). Цени: средни (Avg) и най-добри
между всички букмейкъри (Max). Показва се и доход при цената, която наистина може да се вземе: Avg.
Нищо тук не променя robots rules: това е справка, не настройка.
"""

import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import backtest, dataset, loaders, rules, stack      # noqa: E402

TOURS = {"man": "ATP", "woman": "WTA"}
FIRST_TEST = 2022
VARIANTS = {
    "V0 както във футбола": dict(),
    "V1 + без тото 1.30-1.55": dict(exclude_odds=(1.30, 1.55)),
    "V2 + само при съгласие с пазара": dict(agree_col="mkt_p1"),
    "V3 тото + съгласие": dict(exclude_odds=(1.30, 1.55), agree_col="mkt_p1"),
}


def load(tour):
    folder = ROOT.parent / "data" / tour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [loaders.tennis_data(p, TOURS[tour]) for p in sorted(folder.glob("*"))
                  if p.suffix.lower() in (".xls", ".xlsx", ".csv")]
    df = dataset.build(pd.concat(frames, ignore_index=True))
    df["own"] = stack.walk_forward(df, with_market=False, first_year=2015)
    return df[df["year"] >= FIRST_TEST]


def fmt(s):
    return f"n={s['n']:5d} позн.{s['hit']:.1%} доход {s['roi']:+.3f} (t {s['t']:+.2f}) ср.коеф {s['avg_odds']:.2f}"


def main():
    for tour in TOURS:
        df = load(tour)
        print(f"\n######## {tour}: чиста проверка {FIRST_TEST}-2026, {len(df)} мача ########")
        for book in ("Avg", "Max"):
            print(f"  --- цени {book} ---")
            for name, kw in VARIANTS.items():
                t = rules.tips(df, "own", book, **kw)
                ev = rules.evaluate_tips(t)
                col = rules.evaluate_columns(t)
                print(f"  {name:34s} всички: {fmt(ev['всички'])}")
                print(f"  {'':34s} сигурна: {fmt(ev['сигурна 1.40-1.80']) if ev['сигурна 1.40-1.80']['n'] else '-'}")
                if ev["рискова над 1.80"]["n"]:
                    print(f"  {'':34s} рискова: {fmt(ev['рискова над 1.80'])}")
                print(f"  {'':34s} колонки от 3: {col['columns']} бр., минали {col['passed']:.0%} при казано "
                      f"{col['claimed']:.0%}, връщат {col['return']:.2f} от 1" if col["columns"] else
                      f"  {'':34s} колонки от 3: твърде малко")


if __name__ == "__main__":
    main()
