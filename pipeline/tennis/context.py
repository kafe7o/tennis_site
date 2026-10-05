"""
Контекстът на мача - тениски превод на идеите на майстора от футбола:

  „дерби“ (bets/derbies.py)       -> h2h: колко пъти двамата вече са играли (вечните съперници)
  „след паузата“ (national break)  -> rest1/rest2: дни от последния мач на играча (без междусезонието)
  „формата не значи нищо“         -> ret1/ret2: последният му мач е загубен с отказване (контузия?);
                                     long1/long2: последният му мач е бил дълъг (умора)

Всичко е само от МИНАЛИ дати: за всеки ден първо се смята контекстът на всичките му мачове, после
състоянието се обновява (като elo.walk_forward) - мач от същия ден не влияе. Историята започва от
първия зареден файл (2012), затова h2h е броят срещи от 2012 насам, а първият мач на играча няма rest.

Изискват се колони: date, p1, p2, y, best_of, sets1, sets2, retired (по желание).
"""

import numpy as np
import pandas as pd

OFFSEASON_MONTHS = (11, 12, 1, 2)       # пауза от края на сезона до началото му: не е „пауза“, а междусезоние


def add(df):
    """Връща копие на df с колони rest1, rest2, ret1, ret2, long1, long2, h2h, offseason."""
    need = {"date", "p1", "p2", "y", "best_of", "sets1", "sets2"}
    if need - set(df.columns):
        raise ValueError(f"липсват колони: {sorted(need - set(df.columns))}")
    out = df.copy()
    retired = out["retired"].fillna(False).astype(bool).to_numpy() if "retired" in out else np.zeros(len(out), bool)
    cols = {k: np.full(len(out), np.nan) for k in ("rest1", "rest2")}
    flags = {k: np.zeros(len(out), bool) for k in ("ret1", "ret2", "long1", "long2", "offseason")}
    h2h = np.zeros(len(out), dtype=int)

    last_date, last_ret, last_long, meetings = {}, {}, {}, {}
    pos = {idx: i for i, idx in enumerate(out.index)}
    for _, day in out.sort_values("date", kind="stable").groupby("date", sort=True):
        for idx, m in zip(day.index, day.itertuples()):
            i = pos[idx]
            for tag, player in (("1", m.p1), ("2", m.p2)):
                if player in last_date:
                    cols[f"rest{tag}"][i] = (m.date - last_date[player]).days
                    flags[f"ret{tag}"][i] = last_ret[player]
                    flags[f"long{tag}"][i] = last_long[player]
            if m.p1 in last_date and m.p2 in last_date:
                # междусезонието: единият е играл за последно през ноември/декември, а сега е януари/февруари
                flags["offseason"][i] = (m.date.month in (1, 2)) and any(
                    last_date[p].month in (11, 12) for p in (m.p1, m.p2))
            h2h[i] = meetings.get((m.p1, m.p2), 0)
        for idx, m in zip(day.index, day.itertuples()):
            i = pos[idx]
            total_sets = (m.sets1 if m.sets1 == m.sets1 else 0) + (m.sets2 if m.sets2 == m.sets2 else 0)
            long_match = total_sets >= (4 if m.best_of == 5 else 3)
            loser = m.p2 if m.y == 1 else m.p1
            for player in (m.p1, m.p2):
                last_date[player] = m.date
                last_ret[player] = bool(retired[i] and player == loser)
                last_long[player] = bool(long_match)
            meetings[(m.p1, m.p2)] = meetings.get((m.p1, m.p2), 0) + 1
    for k, v in cols.items():
        out[k] = v
    for k, v in flags.items():
        out[k] = v
    out["h2h"] = h2h
    return out
