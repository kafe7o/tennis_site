"""
Elo за отборни спортове с два изхода (бейзбол, баскетбол, хокей с продължения, НФЛ): домакинско предимство,
множител за разликата в резултата и връщане към средното в началото на сезона.

    очаквано(домакин) = 1 / (1 + 10^(-(оценка[домакин] + HFA - оценка[гост]) / 400))
    стъпка            = K * множител(разлика) * (резултат - очаквано)

Множителят по разликата (MOV): ln(|разлика| + 1) - голяма победа променя повече от минимална. Без него (mov=False)
всяка победа тежи еднакво. Без поглед напред: за всеки ден първо се прогнозират ВСИЧКИ мачове, после се обновяват
оценките (като elo.walk_forward) - двойният мач в един ден не вижда първия.

НЕ знае стартовия питчър, контузиите и съставите - пазарът ги знае. Затова се очаква да е по-слаб от пазара;
проверката назад (research/other_sport_mlb.py) казва дали е така.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

BASE = 1500.0
SCALE = 400.0 / math.log(10)


def _prob(diff):
    return 1.0 / (1.0 + math.exp(-diff / SCALE))


@dataclass(frozen=True)
class TeamEloParams:
    k: float = 4.0
    hfa: float = 24.0           # домакинско предимство в Elo пунктове
    mov: bool = True
    revert: float = 0.25        # дял, с който оценката се връща към 1500 в началото на всеки сезон


def walk_forward(games, params=None):
    """
    games: DataFrame с date, home, away, home_score, away_score (изиграни мачове).
    Връща със същия индекс: p1 (домакин), p2 (гост), y (1 ако е спечелил домакинът), elo_p1, elo_diff (с домакинското
    предимство, домакин минус гост), n1, n2 (мачове, виждани от модела).
    """
    need = {"date", "home", "away", "home_score", "away_score"}
    if need - set(games.columns):
        raise ValueError(f"липсват колони: {sorted(need - set(games.columns))}")
    if games["date"].isna().any():
        raise ValueError("мачове без дата")
    p = params or TeamEloParams()
    rating, seen, season_of = {}, {}, {}
    rows = {}
    for _, day in games.sort_values("date", kind="stable").groupby("date", sort=True):
        year = day["date"].iloc[0].year
        for team in set(day["home"]) | set(day["away"]):
            if team in rating and season_of[team] != year:                  # нов сезон: връщане към средното
                rating[team] = BASE + (1.0 - p.revert) * (rating[team] - BASE)
            season_of[team] = year
        for g in day.itertuples():
            diff = rating.get(g.home, BASE) + p.hfa - rating.get(g.away, BASE)
            rows[g.Index] = (g.home, g.away, int(g.home_score > g.away_score), _prob(diff), diff,
                             seen.get(g.home, 0), seen.get(g.away, 0))
        for g in day.itertuples():
            diff = rating.get(g.home, BASE) + p.hfa - rating.get(g.away, BASE)
            home_won = g.home_score > g.away_score
            margin = abs(g.home_score - g.away_score)
            mult = math.log(margin + 1.0) if p.mov else 1.0
            delta = p.k * mult * ((1.0 if home_won else 0.0) - _prob(diff))
            rating[g.home] = rating.get(g.home, BASE) + delta
            rating[g.away] = rating.get(g.away, BASE) - delta
            seen[g.home] = seen.get(g.home, 0) + 1
            seen[g.away] = seen.get(g.away, 0) + 1
    out = pd.DataFrame.from_dict(rows, orient="index", columns=["p1", "p2", "y", "elo_p1", "elo_diff", "n1", "n2"])
    return out.loc[games.index]
