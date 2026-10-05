
"""
Зареждане на историята в стандартния вид (виж dataset.py).

ВНИМАНИЕ: колоните са по документацията и по памет - БЕЗ да са сверени с реални файлове
(2026-10-05 мрежата на средата не пуска tennis-data.co.uk). Затова зареждането е строго:
липсваща задължителна колона, непозната настилка или непознат кръг СПИРАТ с грешка, а не се
пълнят тихо. При първия реален файл това е първото нещо за сверяване.

  tennis_data()  tennis-data.co.uk: xls/xlsx/csv по година и по тур. Има коефициенти (B365, PS,
                 Max, Avg), точна дата на мача, ранг и точки. Годен за проверката срещу пазара.
  sackmann()     JeffSackmann/tennis_atp и tennis_wta (atp_matches_YYYY.csv): пълни имена,
                 ранг, статистика по точки. НЯМА коефициенти; датата е на началото на турнира,
                 затова към нея се добавят дни по кръга, за да остане редът вътре в турнира.
"""

import numpy as np
import pandas as pd

SURFACE_MAP = {"hard": "Hard", "clay": "Clay", "grass": "Grass", "carpet": "Hard"}
TD_BOOKS = ("B365", "PS", "Max", "Avg")
TD_REQUIRED = ("Date", "Surface", "Round", "Winner", "Loser")
SACK_REQUIRED = ("tourney_name", "surface", "tourney_date", "winner_name", "loser_name", "round", "score")
ROUND_OFFSET = {"ER": 0, "R128": 0, "R64": 1, "R32": 2, "RR": 2, "R16": 3, "QF": 4, "SF": 5, "BR": 6, "F": 6}


def _col(df, name):
    return df[name] if name in df.columns else pd.Series(np.nan, index=df.index)


def _num(series):
    return pd.to_numeric(series, errors="coerce")


def _surface(series, where):
    mapped = series.astype(str).str.strip().str.lower().map(SURFACE_MAP)
    bad = sorted(set(series[mapped.isna()].astype(str)))
    if bad:
        raise ValueError(f"{where}: непозната настилка {bad}")
    return mapped


def tennis_data(path, tour):
    """tour: 'ATP' или 'WTA'. path: .xls / .xlsx / .csv"""
    path = str(path)
    df = pd.read_csv(path) if path.lower().endswith(".csv") else pd.read_excel(path)
    missing = [c for c in TD_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: липсват колони {missing}")
    df = df.dropna(subset=["Winner", "Loser"]).reset_index(drop=True)
    comment = _col(df, "Comment").astype(str).str.lower()
    date = df["Date"] if pd.api.types.is_datetime64_any_dtype(df["Date"]) \
        else pd.to_datetime(df["Date"], dayfirst=True, errors="raise")
    best_of = _num(_col(df, "Best of")).fillna(3).astype(int)
    out = pd.DataFrame({
        "date": date, "tour": tour, "tournament": _col(df, "Tournament"), "round": df["Round"],
        "surface": _surface(df["Surface"], path), "best_of": best_of,
        "winner": df["Winner"].astype(str).str.strip(), "loser": df["Loser"].astype(str).str.strip(),
        "wrank": _num(_col(df, "WRank")), "lrank": _num(_col(df, "LRank")),
        "wpts": _num(_col(df, "WPts")), "lpts": _num(_col(df, "LPts")),
        "walkover": comment.str.contains("walkover") | comment.str.contains("w/o"),
        "retired": comment.str.contains("retired"),
    })
    for book in TD_BOOKS:
        out[f"{book}W"] = _num(_col(df, f"{book}W"))
        out[f"{book}L"] = _num(_col(df, f"{book}L"))
    return out


def sackmann(path, tour):
    """tour: 'ATP' или 'WTA'. path: atp_matches_YYYY.csv"""
    df = pd.read_csv(path)
    missing = [c for c in SACK_REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: липсват колони {missing}")
    unknown = sorted(set(df["round"].astype(str)) - set(ROUND_OFFSET))
    if unknown:
        raise ValueError(f"{path}: непознат кръг {unknown}")
    start = pd.to_datetime(df["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    score = df["score"].astype(str).str.upper()
    return pd.DataFrame({
        "date": start + pd.to_timedelta(df["round"].map(ROUND_OFFSET), unit="D"),
        "tour": tour, "tournament": df["tourney_name"], "round": df["round"],
        "surface": _surface(df["surface"], path),
        "best_of": _num(_col(df, "best_of")).fillna(3).astype(int),
        "winner": df["winner_name"].astype(str).str.strip(), "loser": df["loser_name"].astype(str).str.strip(),
        "wrank": _num(_col(df, "winner_rank")), "lrank": _num(_col(df, "loser_rank")),
        "wpts": _num(_col(df, "winner_rank_points")), "lpts": _num(_col(df, "loser_rank_points")),
        "walkover": score.str.contains("W/O"),
        "retired": score.str.contains("RET"),
    })
