"""
Проверката назад - единственото място, което решава дали има смисъл да се залага.

Три въпроса, по ред:
  1. compare()   Точна ли е вероятността? Лог-загуба и Brier срещу пазара без маржа, на едни и същи
                 мачове, със сдвоен t-тест. Отрицателно t = по-добре от пазара, положително = по-зле.
                 Без по-добра вероятност от пазара залагането няма предимство - има само маржа.
  2. place()     Какво би станало, ако залагаме? Залог само ако P*коеф - 1 >= праг, по ВИДИМИТЕ цени.
  3. protocol()  Честна проверка: правилото (прагът и обхватът на коефициентите) се избира САМО по
                 първата част от времето и се оценява ЕДИН път на втората. Показва се и цялата
                 таблица - колкото повече комбинации, толкова по-вероятно е най-добрата да е късмет.

Доходът е на 1 единица залог. t = доход / стандартна грешка; под ~2 няма доказателство, че
доходът не е шум. Максималните цени (Max) са най-добрите от много къщи - на практика не винаги
са налични, затова се гледат и средните (Avg).
"""

import numpy as np
import pandas as pd


def _loss(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def compare(df, cols, ref="mkt_p1"):
    """Таблица: за всяка колона n, лог-загуба, Brier, разлика спрямо ref и t (само общите мачове)."""
    cols = [c for c in dict.fromkeys([ref, *cols])]
    d = df.loc[df[cols].notna().all(axis=1)]
    y = d["y"].to_numpy(dtype=float)
    ref_loss = _loss(d[ref], y)
    rows = []
    for c in cols:
        loss = _loss(d[c], y)
        diff = loss - ref_loss
        t = np.nan
        if c != ref and diff.std(ddof=1) > 0:
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
        rows.append({"model": c, "n": len(d), "logloss": loss.mean(),
                     "brier": ((d[c].to_numpy() - y) ** 2).mean(),
                     "vs_ref": diff.mean(), "t": t})
    return pd.DataFrame(rows).set_index("model")


def place(df, p_col, book, ev_min=0.0, odds_range=(1.0, 99.0)):
    """Залозите по правилото: изходът с по-голямо очаквано, ако е поне ev_min и в обхвата на цените."""
    o1c, o2c = f"o1_{book}", f"o2_{book}"
    d = df.loc[df[p_col].notna() & df[o1c].notna() & df[o2c].notna()]
    p1, o1, o2 = d[p_col].to_numpy(), d[o1c].to_numpy(), d[o2c].to_numpy()
    ev1, ev2 = p1 * o1 - 1.0, (1.0 - p1) * o2 - 1.0
    first = ev1 >= ev2
    ev = np.where(first, ev1, ev2)
    odds = np.where(first, o1, o2)
    won = np.where(first, d["y"].to_numpy() == 1, d["y"].to_numpy() == 0)
    keep = (ev >= ev_min) & (odds >= odds_range[0]) & (odds <= odds_range[1])
    out = pd.DataFrame({"date": d["date"].to_numpy(), "p": np.where(first, p1, 1.0 - p1),
                        "odds": odds, "ev": ev, "won": won,
                        "profit": np.where(won, odds - 1.0, -1.0)}, index=d.index)
    return out[keep]


def summary(bets):
    n = len(bets)
    if n == 0:
        return {"n": 0, "hit": np.nan, "roi": np.nan, "t": np.nan, "avg_odds": np.nan}
    roi = bets["profit"].mean()
    se = bets["profit"].std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"n": n, "hit": bets["won"].mean(), "roi": roi,
            "t": roi / se if se and se > 0 else np.nan, "avg_odds": bets["odds"].mean()}


def baselines(df, book, p_col="mkt_p1"):
    """Слепи правила за сравнение: винаги фаворит и винаги аутсайдер на пазара. Разликата между тях
    е favourite-longshot bias; и двете губят около маржа, ако пазарът е честен."""
    d = df.loc[df[p_col].notna() & df[f"o1_{book}"].notna() & df[f"o2_{book}"].notna()]
    fav1 = d[p_col] >= 0.5
    o_fav = np.where(fav1, d[f"o1_{book}"], d[f"o2_{book}"])
    fav_won = np.where(fav1, d["y"] == 1, d["y"] == 0)
    res = {}
    for name, won, odds in (("фаворит", fav_won, o_fav),
                            ("аутсайдер", ~fav_won, np.where(fav1, d[f"o2_{book}"], d[f"o1_{book}"]))):
        res[name] = summary(pd.DataFrame({"profit": np.where(won, odds - 1.0, -1.0), "won": won, "odds": odds}))
    return res


GRID_EV = (0.0, 0.02, 0.04, 0.06, 0.10)
GRID_ODDS = ((1.0, 99.0), (1.2, 3.0), (1.4, 2.5), (1.7, 4.0))


def protocol(df, p_col, book, select_until, grid_ev=GRID_EV, grid_odds=GRID_ODDS, min_bets=300):
    """
    Избира правилото по мачовете до select_until (включително) и го оценява веднъж след него.
    Връща: grid (всички комбинации и в двете части), chosen, select, test.
    """
    sel_mask = df["date"] <= pd.Timestamp(select_until)
    sel_df, test_df = df[sel_mask], df[~sel_mask]
    rows = []
    for ev_min in grid_ev:
        for rng in grid_odds:
            s = summary(place(sel_df, p_col, book, ev_min, rng))
            t = summary(place(test_df, p_col, book, ev_min, rng))
            rows.append({"ev_min": ev_min, "odds_from": rng[0], "odds_to": rng[1],
                         **{f"sel_{k}": v for k, v in s.items()}, **{f"test_{k}": v for k, v in t.items()}})
    grid = pd.DataFrame(rows)
    eligible = grid[grid["sel_n"] >= min_bets]
    if eligible.empty:
        return {"grid": grid, "chosen": None, "select": None, "test": None}
    best = eligible.loc[eligible["sel_roi"].idxmax()]
    pick = lambda prefix: {k: best[f"{prefix}_{k}"] for k in ("n", "hit", "roi", "t", "avg_odds")}
    return {"grid": grid, "chosen": {"ev_min": float(best["ev_min"]),
                                     "odds": (float(best["odds_from"]), float(best["odds_to"]))},
            "select": pick("sel"), "test": pick("test")}
