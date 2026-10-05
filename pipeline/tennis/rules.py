"""
Логиката на майстора (професионалиста) от football-site, пренесена към тениса.

Какво е казал той и как е вградено във футболния сайт (bets/robot.py, bets/columns.py, data/pro_tips.json):
  1. Главният съвет е най-вероятният изход ПО СОБСТВЕНИЯ МОДЕЛ на робота, а не по пазара:
     „да не се влияе от коефициентите, когато казва процента“.
  2. Коефициентът служи само за праг и за етикет: под 1.40 няма съвет; 1.40-1.80 е „сигурна“,
     над 1.80 е „рискова“ („рисковата не може да е 1.48“).
  3. Без съвет в непредсказуеми мачове (при футбола - дербитата; при тениса - виж unpredictable()).
  4. Колонка на деня: няколко мача, от които ТРЯБВА да излязат всичките - само прогнози с шанс >= 65%,
     по един мач на първенство (тук - на турнир), всеки мач най-много в една колонка. Показва се
     шансът ѝ, поправен с ИЗМЕРЕНОТО надценяване, коефициентът ѝ и колко от 1 се връщат средно.
  5. Нищо не се променя без число: правилото се мери назад и на живо и се показва честно,
     включително когато губи (във футбола: -6.2% доход; колонка от 3 връща ~0.8).

ВАЖНО: числата по-долу (1.40, 1.80, 65%, 3 мача, 0.92) са измерени ВЪРХУ ФУТБОЛ. Те са отправна
точка, не истина за тениса - evaluate_tips() и evaluate_columns() ги мерят върху тенис данни.
В тениса има два изхода, затова най-вероятният изход винаги е с шанс >= 50% и MIN_PROB реално
не филтрира нищо; тук го вдигаме с параметър, ако данните покажат смисъл.
"""

import numpy as np
import pandas as pd

from . import backtest

MIN_ODDS = 1.40
SAFE_MAX = 1.80
MIN_PROB = 0.50
COLUMN_MIN_P = 0.65
COLUMN_SIZE = 3
COLUMN_MAX = 4
CALIBRATION = 0.92          # футболът: роботът казва 36% за колонка от 3, излиза 29%


def kind(odds):
    """сигурна 1.40-1.80, рискова над 1.80, иначе None (под 1.40 няма съвет)."""
    if odds < MIN_ODDS:
        return None
    return "safe" if odds <= SAFE_MAX else "risky"


def unpredictable(n1, n2, min_seen=10):
    """Тенис аналогът на футболните дербита: мач, в който поне един играч е виждан от модела твърде малко
    пъти (нов в тура, връщане след дълга пауза) - оценката му е догадка, а не мнение."""
    return min(n1, n2) < min_seen


def tips(df, p_col, book, min_prob=MIN_PROB, min_odds=MIN_ODDS, min_seen=10, exclude_odds=None, agree_col=None):
    """
    Главният съвет за всеки мач по модела p_col (вероятност p1 да победи; НЕ пазарът!).
    exclude_odds=(lo, hi): без съвет, когато коефициентът му е в тази лента („тото“ - нисък коефициент, който
    не излиза). agree_col: колона с вероятност (пазарът); съвет само когато и тя е за същия играч.
    Връща: date, tournament, side (1 или 2), p, odds, kind, won, profit.
    """
    o1c, o2c = f"o1_{book}", f"o2_{book}"
    d = df.loc[df[p_col].notna() & df[o1c].notna() & df[o2c].notna()]
    d = d.loc[~((d[["n1", "n2"]].min(axis=1)) < min_seen)]
    if agree_col is not None:
        d = d.loc[d[agree_col].notna() & ((d[p_col] >= 0.5) == (d[agree_col] >= 0.5))]
    first = d[p_col].to_numpy() >= 0.5
    p = np.where(first, d[p_col].to_numpy(), 1.0 - d[p_col].to_numpy())
    odds = np.where(first, d[o1c].to_numpy(), d[o2c].to_numpy())
    won = np.where(first, d["y"].to_numpy() == 1, d["y"].to_numpy() == 0)
    out = pd.DataFrame({"date": d["date"].to_numpy(), "side": np.where(first, 1, 2), "p": p, "odds": odds,
                        "won": won, "profit": np.where(won, odds - 1.0, -1.0)}, index=d.index)
    out["tournament"] = d["tournament"].to_numpy() if "tournament" in d else "?"
    out["kind"] = [kind(o) for o in odds]
    keep = (out["p"] >= min_prob) & (out["odds"] >= min_odds)
    if exclude_odds is not None:
        keep &= ~out["odds"].between(*exclude_odds)
    return out[keep]


def evaluate_tips(t):
    """По вид (сигурна / рискова / всички): колко са, колко познава, колко е казвал, средна цена, доход и t."""
    rows = {}
    for name, part in (("всички", t), ("сигурна 1.40-1.80", t[t["kind"] == "safe"]),
                       ("рискова над 1.80", t[t["kind"] == "risky"])):
        s = backtest.summary(part)
        s["казва"] = float(part["p"].mean()) if len(part) else np.nan
        rows[name] = s
    return rows


def build_columns(cands, size=COLUMN_SIZE, max_columns=COLUMN_MAX):
    """
    Колонките за един ден (порт на bets/columns.build): cands са записи с id, tournament, p, odds, won, date,
    подредени от най-вероятния. По един мач на турнир; всеки мач най-много в една колонка; по-къса колонка
    от size не се прави.
    """
    used, columns = set(), []
    for _ in range(max_columns):
        col, seen = [], set()
        for c in cands:
            if c["id"] in used or len(col) == size or c["tournament"] in seen:
                continue
            col.append(c)
            seen.add(c["tournament"])
        if len(col) < size:
            break
        used.update(c["id"] for c in col)
        columns.append(col)
    return columns


def evaluate_columns(t, min_p=COLUMN_MIN_P, size=COLUMN_SIZE, max_columns=COLUMN_MAX):
    """
    Колонките назад във времето, ден по ден (денят е календарната дата). Връща: колонки, колко са минали,
    какво е казвал роботът (произведението от шансовете), надценяване (казвал / излязло) и колко от 1
    се връщат средно при залог 1 на всяка колонка.
    """
    t = t[t["p"] >= min_p].copy()
    t["id"] = t.index
    claimed, passed, ret = [], [], []
    for _, day in t.sort_values("p", ascending=False).groupby("date"):
        for col in build_columns(day.to_dict("records"), size, max_columns):
            odds = float(np.prod([c["odds"] for c in col]))
            ok = all(c["won"] for c in col)
            claimed.append(float(np.prod([c["p"] for c in col])))
            passed.append(ok)
            ret.append(odds if ok else 0.0)
    n = len(passed)
    if n == 0:
        return {"columns": 0, "passed": np.nan, "claimed": np.nan, "overrate": np.nan, "return": np.nan}
    rate, said = float(np.mean(passed)), float(np.mean(claimed))
    return {"columns": n, "passed": rate, "claimed": said, "overrate": said / rate if rate else np.inf,
            "return": float(np.mean(ret))}
