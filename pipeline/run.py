"""
Един вход за всичко.

    python run.py selftest                       проверка на системата върху СИМУЛИРАН свят (без данни)
    python run.py backtest --data DIR [--tour man|woman]
                                                 проверката назад върху реални мачове и коефициенти;
                                                 DIR/man (ATP) и DIR/woman (WTA) със файловете от tennis-data.co.uk,
                                                 всеки тур се пуска отделно

Първо се пуска selftest: ако системата не различава "няма предимство" от "има предимство" в свят, в
който истината се знае, резултатът от backtest не значи нищо.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from tennis import backtest, dataset, loaders, rules, simulate, stack

BOOKS_SHOWN = ("Avg", "Max", "PS")
TOURS = {"man": "ATP", "woman": "WTA"}        # папката в data/ -> тур


def fmt(d):
    return ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in d.items())


def report(df, first_year, select_until, book):
    test = df[df["year"] >= first_year].copy()
    test["stack_nomkt"] = stack.walk_forward(df, with_market=False, first_year=first_year).reindex(test.index)
    test["stack_mkt"] = stack.walk_forward(df, with_market=True, first_year=first_year).reindex(test.index)
    print(f"\nМачове за оценка: {len(test)} (от {first_year}), с коефициенти: {int(test['mkt_p1'].notna().sum())}")
    print("\n1. Точност на вероятността (лог-загуба по-малка = по-добре; t < 0 = по-добре от пазара)")
    print(backtest.compare(test, ["elo_p1", "stack_nomkt", "stack_mkt"]).round(4).to_string())
    print("\n2. Слепи правила (залог 1 единица на всеки мач)")
    for b in BOOKS_SHOWN:
        if test[f"o1_{b}"].notna().any():
            for name, s in backtest.baselines(test, b).items():
                print(f"   {b:4s} {name:10s} {fmt(s)}")
    print(f"\n3. Правилото се избира до {select_until}, оценява се ЕДИН път след това (цени: {book})")
    for col in ("stack_mkt", "stack_nomkt"):
        r = backtest.protocol(test, col, book, select_until)
        if r["chosen"] is None:
            print(f"   {col}: твърде малко залози за избор")
            continue
        print(f"   {col}: правило {r['chosen']}\n      избор: {fmt(r['select'])}\n      тест : {fmt(r['test'])}")
        print(f"      (опитани комбинации: {len(r['grid'])} - най-добрата в избора често е късмет)")
    print(f"\n4. Логиката на майстора (цени: {book}) - съветът е по СОБСТВЕНИЯ модел (stack_nomkt), без пазара")
    tips = rules.tips(test, "stack_nomkt", book)
    for name, s in rules.evaluate_tips(tips).items():
        print(f"   {name:18s} {fmt(s)}")
    print(f"   колонки от {rules.COLUMN_SIZE} (шанс >= {rules.COLUMN_MIN_P:.0%}, един мач на турнир): "
          f"{fmt(rules.evaluate_columns(tips))}")
    print("   (прагове 1.40 / 1.80 / 65% / 3 мача са от футбола - тук се мерят наново)")
    return test


def cmd_selftest(_):
    for sees, meaning in (("all", "пазарът знае всичко - предимство НЯМА"),
                          ("no_surface", "пазарът не знае настилката - предимство ИМА")):
        print(f"\n######## СИМУЛАЦИЯ: {meaning} ########")
        df = dataset.build(simulate.world(seed=1, market_sees=sees))
        report(df, 2017, "2020-12-31", "Avg")
    print("\nОчаквано: в първия свят t за залозите не е над ~2 и доходът е около -маржа; във втория - ясно положителен.")


def cmd_backtest(args):
    folder = Path(args.data) / args.tour
    frames = [loaders.tennis_data(path, TOURS[args.tour]) for path in sorted(folder.glob("*"))
              if path.suffix.lower() in (".xls", ".xlsx", ".csv")]
    if not frames:
        raise SystemExit(f"Няма файлове в {folder} (tennis-data.co.uk: .xls/.xlsx/.csv)")
    # мъжете и жените са отделни светове: никога не играят един срещу друг, затова модел за всеки
    matches = pd.concat(frames, ignore_index=True)
    print(f"Заредени {len(matches)} мача, {matches['date'].min().date()} - {matches['date'].max().date()}")
    df = dataset.build(matches)
    first_year = int(args.first_year or df["year"].min() + 3)
    select_until = args.select_until or f"{int(df['year'].max()) - 2}-12-31"
    report(df, first_year, select_until, args.book)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest").set_defaults(fn=cmd_selftest)
    bt = sub.add_parser("backtest")
    bt.add_argument("--data", required=True)
    bt.add_argument("--tour", default="man", choices=tuple(TOURS), help="man (ATP) или woman (WTA) - отделно")
    bt.add_argument("--first-year", default=None)
    bt.add_argument("--select-until", default=None)
    bt.add_argument("--book", default="Avg", choices=("Avg", "Max", "PS", "B365"))
    bt.set_defaults(fn=cmd_backtest)
    args = parser.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
