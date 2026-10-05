"""
Друг спорт: бейзбол (MLB). Има ли предимство над пазара? Протоколът е записан ПРЕДИ пускането.

Данни: Hugging Face "Oronto/baseball-stats-cleaned_oddsportal_mlb" - 40 359 мача 2006-2024, коефициенти за домакин и гост
(изстъргани от OddsPortal, американски формат; кой е моментът на цената - не е посочен; лицензът не е посочен). НЕ се
качва в публичното репозитори. Почистване: конверсия към десетични; редове с марж < 0 или > 10% (6) и истински
дубликати (същият час, същите отбори; 2) се махат. „Дубликатите“ по дата са двойни мачове - остават.

Модел: Elo за отбори (tennis/team_elo.py) - домакинско предимство, множител по разликата, реверсия на сезона. Параметрите
(k 2/4/6, hfa 15/24/35, mov да/не; 18 комбинации) се избират САМО по лог-загубата на 2009-2017; после се оценява ЕДИН път
на 2018-2024. Логистичен слой над него - с пазара и без пазара, всяка година само от минали години (stack.walk_forward).

Оценка: стандартната на backtest.py - лог-загуба срещу пазара със сдвоен t-тест; слепи правила; правило, избрано до
2017 и оценено на 2018-2024. „Предимство“ = лог-загуба по-добра от пазарната с t <= -3 (по-строго от 2, защото се тестват
няколко спорта и някой излиза на плюс от късмет) И положителен доход на чистата проверка при средни цени.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import backtest, market, stack, team_elo      # noqa: E402

SELECT_YEARS = (2009, 2017)
FIRST_TEST = 2018
GRID = [team_elo.TeamEloParams(k=k, hfa=h, mov=m) for k, h, m in itertools.product((2.0, 4.0, 6.0), (15.0, 24.0, 35.0), (True, False))]


def american_to_decimal(a):
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def load(path):
    raw = pd.read_parquet(path)
    df = pd.DataFrame({"date": pd.to_datetime(raw["game_date"]), "start": pd.to_datetime(raw["game_datetime"]),
                       "home": raw["home_team_abbr"], "away": raw["away_team_abbr"],
                       "home_score": raw["home_score"], "away_score": raw["away_score"],
                       "ho": american_to_decimal(raw["home_odds"]), "ao": american_to_decimal(raw["away_odds"])})
    margin = 1 / df["ho"] + 1 / df["ao"] - 1
    before = len(df)
    df = df[(margin >= 0) & (margin <= 0.10)]
    df = df[~df.duplicated(["start", "home", "away"], keep="first")]
    df = df[df["home_score"] != df["away_score"]].reset_index(drop=True)
    print(f"Мачове: {before} -> {len(df)} след почистване ({before - len(df)} махнати)")
    return df


def frame(games, params):
    feats = team_elo.walk_forward(games, params)
    out = pd.DataFrame({"date": games["date"], "year": games["date"].dt.year, "y": feats["y"], "elo_p1": feats["elo_p1"],
                        "elo_diff": feats["elo_diff"], "surf_diff": 0.0, "n1": feats["n1"], "n2": feats["n2"], "best_of": 3,
                        "rank1": np.nan, "rank2": np.nan, "pts1": np.nan, "pts2": np.nan,
                        "o1_Avg": games["ho"], "o2_Avg": games["ao"]})
    out["mkt_p1"] = market.devig(out["o1_Avg"].to_numpy(), out["o2_Avg"].to_numpy(), "power")
    return out


def loss(df, col):
    d = df[df[col].notna()]
    y, p = d["y"].to_numpy(float), np.clip(d[col].to_numpy(float), 1e-9, 1 - 1e-9)
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="mlb.parquet от Hugging Face (не се качва в репозиторито)")
    args = ap.parse_args()
    games = load(args.file)
    print(f"Период: {games['date'].min().date()} - {games['date'].max().date()}; средно домакинът печели {np.mean(games['home_score'] > games['away_score']):.3f}")

    # 1. избор на параметри САМО по 2009-2017 (суровата Elo прогноза, без пазара)
    scored = []
    for p in GRID:
        f = frame(games, p)
        scored.append((loss(f[f["year"].between(*SELECT_YEARS)], "elo_p1"), p))
    scored.sort(key=lambda t: t[0])
    best = scored[0][1]
    print("\nНай-добри 3 комбинации по лог-загуба на 2009-2017:")
    for l, p in scored[:3]:
        print(f"   {l:.5f}  k={p.k} hfa={p.hfa} mov={p.mov}")
    print(f"Избрано: k={best.k} hfa={best.hfa} mov={best.mov}")

    # 2. оценка ЕДИН път на чистия период
    df = frame(games, best)
    df["stack_nomkt"] = stack.walk_forward(df, with_market=False, first_year=SELECT_YEARS[0])
    df["stack_mkt"] = stack.walk_forward(df, with_market=True, first_year=SELECT_YEARS[0])
    test = df[df["year"] >= FIRST_TEST]
    print(f"\nЧиста проверка {FIRST_TEST}-2024: {len(test)} мача")
    print(backtest.compare(test, ["elo_p1", "stack_nomkt", "stack_mkt"]).round(4).to_string())
    print("\nСлепи правила (средни цени):")
    for name, s in backtest.baselines(test, "Avg").items():
        print(f"   {name:10s} n={s['n']}, позн.{s['hit']:.1%}, доход {s['roi']:+.3f} (t {s['t']:+.2f}), ср.коеф {s['avg_odds']:.2f}")
    for name, col in (("с пазара", "stack_mkt"), ("без пазара", "stack_nomkt")):
        r = backtest.protocol(df[df["year"] >= SELECT_YEARS[0] + 0], col, "Avg", select_until=f"{SELECT_YEARS[1]}-12-31", min_bets=300)
        if r["chosen"]:
            print(f"\nПравило ({name}): ev>={r['chosen']['ev_min']}, коеф {r['chosen']['odds']}")
            print("   избор:", {k: round(float(v), 3) for k, v in r["select"].items()})
            print("   тест :", {k: round(float(v), 3) for k, v in r["test"].items()})
    t = backtest.compare(test, ["stack_mkt"]).loc["stack_mkt", "t"]
    print(f"\nПрагът за „предимство“: t <= -3 и положителен доход. stack_mkt t = {t:+.2f}")


if __name__ == "__main__":
    main()
