"""
Elo за тенис: обща оценка + оценка по настилка, с намаляваща стъпка.

    очаквано = 1 / (1 + 10^(-(оценка[А] - оценка[Б]) / 400))
    стъпка K  = k_delta / (мачове + k_offset)^k_shape      (по-малко мачове - по-голяма стъпка)

ЗАЩО два Elo-та, а не един: играчът на клей и играчът на трева са различни играчи. Общата оценка
има много данни, но е размита; оценката по настилка е точна, но с малко мачове. Прогнозата ползва
смес (surface_weight). Оценката по настилка започва от общата в момента на първия мач на тази
настилка - иначе всеки нов играч на нова настилка тръгва от 1500 и разваля прогнозите.

В мач от 5 сета по-добрият печели по-често (по-малко късмет) - bo5_boost увеличава разликата.

НЯМА поглед напред: walk_forward() прогнозира всички мачове от един ден, ЧАК СЛЕД ТОВА ги вкарва.
Тест: tests/test_elo.py проверява, че промяна в бъдещите мачове не променя миналите прогнози.
"""

import math
from dataclasses import dataclass

import pandas as pd

BASE = 1500.0
SCALE = 400.0 / math.log(10)          # разлика в оценката -> логит


def _prob(diff):
    return 1.0 / (1.0 + math.exp(-diff / SCALE))


@dataclass(frozen=True)
class EloParams:
    k_delta: float = 250.0
    k_offset: float = 5.0
    k_shape: float = 0.4
    surface_weight: float = 0.5
    bo5_boost: float = 1.0


class Elo:
    def __init__(self, params=None):
        self.params = params or EloParams()
        self.overall = {}
        self.n = {}                    # мачове на играча (всички настилки)
        self.surf = {}
        self.ns = {}

    # --- оценки ---
    def rating(self, player):
        return self.overall.get(player, BASE)

    def surface_rating(self, player, surface):
        return self.surf.get((player, surface), self.rating(player))

    def blended(self, player, surface):
        w = self.params.surface_weight
        return (1.0 - w) * self.rating(player) + w * self.surface_rating(player, surface)

    def seen(self, player):
        return self.n.get(player, 0)

    # --- прогноза ---
    def predict(self, a, b, surface, best_of=3):
        diff = self.blended(a, surface) - self.blended(b, surface)
        if best_of == 5:
            diff *= self.params.bo5_boost
        return _prob(diff)

    # --- обновяване ---
    def _k(self, matches):
        p = self.params
        return p.k_delta / (matches + p.k_offset) ** p.k_shape

    def update(self, winner, loser, surface):
        exp = _prob(self.rating(winner) - self.rating(loser))
        kw, kl = self._k(self.seen(winner)), self._k(self.seen(loser))
        # оценката по настилка тръгва от общата ПРЕДИ този мач
        sw, sl = self.surface_rating(winner, surface), self.surface_rating(loser, surface)
        exp_s = _prob(sw - sl)
        ksw, ksl = self._k(self.ns.get((winner, surface), 0)), self._k(self.ns.get((loser, surface), 0))

        self.overall[winner] = self.rating(winner) + kw * (1.0 - exp)
        self.overall[loser] = self.rating(loser) - kl * (1.0 - exp)
        self.surf[(winner, surface)] = sw + ksw * (1.0 - exp_s)
        self.surf[(loser, surface)] = sl - ksl * (1.0 - exp_s)
        self.n[winner] = self.seen(winner) + 1
        self.n[loser] = self.seen(loser) + 1
        self.ns[(winner, surface)] = self.ns.get((winner, surface), 0) + 1
        self.ns[(loser, surface)] = self.ns.get((loser, surface), 0) + 1


def walk_forward(matches, params=None):
    """
    matches: DataFrame с колони date, winner, loser, surface, best_of (всеки ред е изигран мач).

    За всеки ден СНАЧАЛА се прогнозират всички мачове, после се обновяват оценките.
    Играч 1 е първият по азбучен ред - не победителят, иначе етикетът би издавал отговора.

    Връща DataFrame със същия индекс: p1, p2, y (1 ако е победил p1), elo_p1, elo_diff,
    surf_diff (преди мача, p1 минус p2), n1, n2 (колко мача е виждал моделът за всеки).
    """
    need = {"date", "winner", "loser", "surface", "best_of"}
    if need - set(matches.columns):
        raise ValueError(f"липсват колони: {sorted(need - set(matches.columns))}")
    ordered = matches.sort_values("date", kind="stable")
    model = Elo(params)
    rows = {}
    for _, day in ordered.groupby("date", sort=True):
        for m in day.itertuples():
            a, b = sorted((m.winner, m.loser))
            rows[m.Index] = (a, b, int(m.winner == a),
                             model.predict(a, b, m.surface, m.best_of),
                             model.rating(a) - model.rating(b),
                             model.surface_rating(a, m.surface) - model.surface_rating(b, m.surface),
                             model.seen(a), model.seen(b))
        for m in day.itertuples():
            model.update(m.winner, m.loser, m.surface)
    out = pd.DataFrame.from_dict(rows, orient="index",
                                 columns=["p1", "p2", "y", "elo_p1", "elo_diff", "surf_diff", "n1", "n2"])
    return out.loc[matches.index]
