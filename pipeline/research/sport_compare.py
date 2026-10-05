"""
Кой спорт е най-предвидим? Сравнение по РЕЗУЛТАТИ, без цени: колко мача може да се познае надеждно със собствен модел.

Протокол (записан преди пускането): за всеки спорт един и същ метод - Elo (домакинско предимство за отборните спортове,
множител по разликата, реверсия на сезона), параметрите се избират САМО по лог-загубата на ранните години, после
логистична калибровка (всяка година само от минали) и оценка на последните години. Мярка: за ниво на сигурност
(>=60/70/80/90%) - колко от мачовете покрива и колко от тях познава (честно ли е казаното). Тенисът е от event_lab.py
(модел без пазара). Събитието е „печели по-вероятният“; по-късно може да се добавят над/под и разлики.

Данни: MLB - Hugging Face Oronto/baseball-stats-cleaned_oddsportal_mlb; НБА - hamzas/nba-games (games.csv, до 2025-02).
Не се качват в публичното репозитори.
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import stack, team_elo      # noqa: E402

TIERS = (0.6, 0.7, 0.8, 0.9)


def load_mlb(path):
    raw = pd.read_parquet(path)
    df = pd.DataFrame({"date": pd.to_datetime(raw["game_date"]), "start": pd.to_datetime(raw["game_datetime"]),
                       "home": raw["home_team_abbr"], "away": raw["away_team_abbr"],
                       "home_score": raw["home_score"], "away_score": raw["away_score"]})
    df = df[~df.duplicated(["start", "home", "away"])]
    return df[df["home_score"] != df["away_score"]].drop(columns="start").reset_index(drop=True)


def load_nba(path):
    g = pd.read_csv(path)
    g = g[g["PTS_home"].notna() & g["PTS_away"].notna() & ~g["GAME_ID"].duplicated()]
    df = pd.DataFrame({"date": pd.to_datetime(g["GAME_DATE_EST"]), "home": g["HOME_TEAM_ID"].astype(str),
                       "away": g["VISITOR_TEAM_ID"].astype(str), "home_score": g["PTS_home"], "away_score": g["PTS_away"]})
    return df[df["home_score"] != df["away_score"]].reset_index(drop=True)


SPORTS = {
    "бейзбол (MLB)": dict(loader=load_mlb, arg="mlb", select=(2009, 2017), test=2018,
                          grid=[team_elo.TeamEloParams(k=k, hfa=h, mov=m) for k, h, m in itertools.product((2.0, 4.0, 6.0), (15.0, 24.0, 35.0), (True, False))]),
    "баскетбол (НБА)": dict(loader=load_nba, arg="nba", select=(2006, 2016), test=2018,
                            grid=[team_elo.TeamEloParams(k=k, hfa=h, mov=m) for k, h, m in itertools.product((5.0, 10.0, 20.0), (60.0, 100.0), (True, False))]),
}


def loss(p, y):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((-(y * np.log(p) + (1 - y) * np.log(1 - p))).mean())


def tiers_table(p, y):
    side = p >= 0.5
    conf = np.where(side, p, 1 - p)
    hit = np.where(side, y == 1, y == 0)
    cells = []
    for t in TIERS:
        m = conf >= t
        cells.append(f"{m.mean():5.0%} {hit[m].mean():6.1%}" if m.sum() >= 100 else "      -     ")
    return hit.mean(), cells


def run_team_sport(name, cfg, path):
    games = cfg["loader"](path)
    lo, hi = cfg["select"]
    best, best_loss = None, 9
    for params in cfg["grid"]:
        f = team_elo.walk_forward(games, params)
        yr = games["date"].dt.year
        sel = f[yr.between(lo, hi)]
        l = loss(sel["elo_p1"].to_numpy(), sel["y"].to_numpy(float))
        if l < best_loss:
            best, best_loss = params, l
    f = team_elo.walk_forward(games, best)
    df = pd.DataFrame({"year": games["date"].dt.year, "y": f["y"], "elo_diff": f["elo_diff"], "surf_diff": 0.0,
                       "best_of": 3, "rank1": np.nan, "rank2": np.nan, "pts1": np.nan, "pts2": np.nan, "mkt_p1": np.nan})
    df["p"] = stack.walk_forward(df, with_market=False, first_year=lo)
    test = df[(df["year"] >= cfg["test"]) & df["p"].notna()]
    acc, cells = tiers_table(test["p"].to_numpy(), test["y"].to_numpy())
    print(f"{name:18s} {len(test):6d} {acc:6.1%} | " + " | ".join(cells) + f"   (Elo k={best.k} hfa={best.hfa} mov={best.mov})")


def run_tennis():
    import event_lab
    for tour, label in (("man", "тенис мъже"), ("woman", "тенис жени")):
        df = event_lab.load(tour).dropna(subset=["own"])
        lab, prob = event_lab.predict_events(df, "own")
        test = (df["year"] >= event_lab.FIRST_TEST) & prob["fav_won"].notna()
        # печели фаворитът (по модела): вероятността на събитието е за „фаворит печели“ -> p за p1
        fav1 = (df["own"] >= 0.5).to_numpy()
        pf = prob.loc[test, "fav_won"].to_numpy()
        p1 = np.where(fav1[test.to_numpy()], pf, 1 - pf)
        acc, cells = tiers_table(p1, df.loc[test, "y"].to_numpy())
        print(f"{label:18s} {int(test.sum()):6d} {acc:6.1%} | " + " | ".join(cells) + "   (Elo с геймове, логистика)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlb")
    ap.add_argument("--nba")
    args = ap.parse_args()
    print(f"{'спорт':18s} {'мача':>6s} {'вярно':>6s} | " + " | ".join(f"≥{int(t * 100)}%: покрива познава" for t in TIERS))
    run_tennis()
    for name, cfg in SPORTS.items():
        path = getattr(args, cfg["arg"])
        if path:
            run_team_sport(name, cfg, path)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
