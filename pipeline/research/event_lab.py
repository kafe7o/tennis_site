"""
Колко надеждно може да се ПОЗНАВАТ събития в тениса - без значение каква е цената им.

Целта (майсторът): „сигурна прогноза“, която познава средно ~70%, и да се види къде са силните и слабите ѝ страни.
Протокол (записан преди пускането):
  Данни: tennis-data.co.uk 2012-2026, мъже и жени поотделно, без отказали се (сетовете и геймовете не са пълни).
  Събития (по фаворита на пазара): победа на фаворита; победа на фаворита без загубен сет; аутсайдерът взима поне един
  сет; мачът стига до решаващия сет; фаворитът взима поне един сет; общо геймове над/под линия (само мачове до 3 сета,
  линии 19.5-24.5).
  Прогноза за всяко събитие: логистична регресия върху силата на фаворита (логит на пазара без маржа), мач до 5 сета,
  настилка, закрит корт, ниво на турнира; всяка година се учи само от минали години (от 2015). Втори вариант
  „само моделът“ - със stack_nomkt вместо пазара.
  Оценка: само 2022-2026. За всяко събитие и ниво на сигурност (>=60/70/80/90%): колко от мачовете покрива, колко
  познава и колко е казвал. Страната е по-вероятната (събитието или обратното). Калибровка: казано = излязло?

Важно: колкото по-лесно събитие, толкова по-висок процент (поне един сет за фаворита излиза ~90%+). Така че процентът
сам не е заслуга - гледа се и какво е базовото ниво на събитието и колко мача покрива нивото.
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import dataset, loaders, market, stack      # noqa: E402

TOURS = {"man": "ATP", "woman": "WTA"}
FIRST_MODEL_YEAR, FIRST_TEST = 2015, 2022
TIERS = (0.6, 0.7, 0.8, 0.9)
LINES = (19.5, 20.5, 21.5, 22.5, 23.5, 24.5)


def load(tour):
    folder = ROOT.parent / "data" / tour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [loaders.tennis_data(p, TOURS[tour]) for p in sorted(folder.glob("*"))
                  if p.suffix.lower() in (".xls", ".xlsx", ".csv")]
    df = dataset.build(pd.concat(frames, ignore_index=True))
    df["own"] = stack.walk_forward(df, with_market=False, first_year=FIRST_MODEL_YEAR)
    df = df[~df["retired"].fillna(False).astype(bool) & df["sets1"].notna() & df["sets2"].notna() & df["mkt_p1"].notna()].copy()
    return df


def orient(df, p_col):
    """Етикетите спрямо фаворита по p_col (вероятност p1)."""
    p1 = df[p_col]
    fav1 = (p1 >= 0.5).to_numpy()
    fav_p = np.where(fav1, p1, 1 - p1)
    sets_fav = np.where(fav1, df["sets1"], df["sets2"])
    sets_dog = np.where(fav1, df["sets2"], df["sets1"])
    fav_won = np.where(fav1, df["y"] == 1, df["y"] == 0)
    total_sets = sets_fav + sets_dog
    full = np.where(df["best_of"] == 5, 5, 3)
    lab = pd.DataFrame({"fav_p": fav_p, "fav_won": fav_won,
                        "straight": fav_won & (sets_dog == 0),
                        "dog_set": ~(fav_won & (sets_dog == 0)),
                        "distance": total_sets == full,
                        "fav_set": ~(~fav_won & (sets_fav == 0))}, index=df.index)
    games = df["games1"] + df["games2"]
    for line in LINES:
        lab[f"games_over_{line}"] = np.where((df["best_of"] == 3) & games.notna(), games > line, np.nan)
    return lab


def features(df, fav_p):
    fp = np.clip(fav_p, 0.5, 0.999)
    big = df["series"].astype(str).isin(["Masters 1000", "WTA1000", "Premier Mandatory", "Premier 5"]).astype(float)
    return pd.DataFrame({"logit": market.logit(fp), "fp": fp, "bo5": (df["best_of"] == 5).astype(float),
                         "bo5_logit": (df["best_of"] == 5) * market.logit(fp),
                         "clay": (df["surface"] == "Clay").astype(float), "grass": (df["surface"] == "Grass").astype(float),
                         "indoor": (df["court"].astype(str) == "Indoor").astype(float),
                         "slam": (df["series"].astype(str) == "Grand Slam").astype(float), "big": big}, index=df.index)


def predict_events(df, p_col):
    lab = orient(df, p_col)
    X = features(df, lab["fav_p"].to_numpy())
    out = pd.DataFrame(index=df.index)
    for ev in [c for c in lab.columns if c != "fav_p"]:
        y = lab[ev]
        pr = pd.Series(np.nan, index=df.index)
        for year in sorted(df["year"].unique()):
            if year < FIRST_MODEL_YEAR:
                continue
            tr = (df["year"] < year) & y.notna()
            te = (df["year"] == year) & y.notna()
            if tr.sum() < 1500 or not te.any():
                continue
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr].astype(int))
            pr[te] = m.predict_proba(X[te])[:, 1]
        out[ev] = pr
    return lab, out


def report(tour, df, p_col, label):
    lab, prob = predict_events(df, p_col)
    test = df["year"] >= FIRST_TEST
    print(f"\n--- {tour}: {label} ({int(test.sum())} мача, {FIRST_TEST}-2026) ---")
    names = {"fav_won": "печели фаворитът", "straight": "фаворитът без загубен сет", "dog_set": "аутсайдерът взима сет",
             "distance": "до решаващия сет", "fav_set": "фаворитът взима сет"}
    names.update({f"games_over_{l}": f"геймове над/под {l}" for l in LINES})
    head = f"{'събитие':30s} {'база':>6s} | " + " | ".join(f"≥{int(t*100)}%: покрива познава" for t in TIERS)
    print(head)
    for ev, name in names.items():
        ok = test & prob[ev].notna()
        if ok.sum() < 300:
            continue
        y = lab.loc[ok, ev].astype(bool).to_numpy()
        p = prob.loc[ok, ev].to_numpy()
        side = p >= 0.5
        conf = np.where(side, p, 1 - p)
        hit = np.where(side, y, ~y)
        cells = []
        for t in TIERS:
            m = conf >= t
            cells.append(f"{m.mean():5.0%} {hit[m].mean():6.1%}" if m.sum() >= 100 else "      -     ")
        print(f"{name:30s} {y.mean():6.1%} | " + " | ".join(f"        {c}" for c in cells))
    return lab, prob


def main():
    for tour in TOURS:
        df = load(tour)
        report(tour, df, "mkt_p1", "с цената като вход (най-точното, което имаме)")
        report(tour, df.dropna(subset=["own"]), "own", "само моделът (без пазара)")


if __name__ == "__main__":
    main()
