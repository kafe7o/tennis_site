"""
Логистичният слой над Elo: всяка година се обучава само на минали години (walk-forward).

Всички признаци са РАЗЛИКИ p1 минус p2 (или логит), а свободният член е изключен - така
P(p1 печели) = 1 - P(p2 печели) е вярно по построение и подредбата на играчите не влияе.

  elo     разликата в смесената Elo оценка (на 100 пункта)
  surf    разликата в оценката по настилка
  rank    log(ранг2 / ранг1) - липсващият ранг е 0 (никаква информация)
  pts     log(точки1 / точки2)
  bo5     elo x (мач от 5 сета)
  mkt     логитът на пазара без маржа - САМО във варианта "с пазара"

Двата варианта си имат цел: "без пазара" е собственото мнение на модела; "с пазара" показва дали
моделът прибавя нещо над цената. Само втората може да има смисъл за залагане - пазарът е точен.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import market


def design(df, with_market):
    def log_ratio(a, b):
        r = np.log(b / a)
        return r.where(np.isfinite(r), 0.0).fillna(0.0).clip(-5, 5)

    x = pd.DataFrame({
        "elo": df["elo_diff"] / 100.0,
        "surf": df["surf_diff"] / 100.0,
        "rank": log_ratio(df["rank1"], df["rank2"]),
        "pts": log_ratio(df["pts2"], df["pts1"]),     # повече точки = по-добър, затова обърнато
        "bo5": df["elo_diff"] / 100.0 * (df["best_of"] == 5),
    }, index=df.index)
    if with_market:
        x["mkt"] = market.logit(df["mkt_p1"])
    return x


def walk_forward(df, with_market, first_year, c=1.0, min_train=2000):
    """Прогноза за P(p1 печели) за всеки ред от години >= first_year; NaN другаде."""
    out = pd.Series(np.nan, index=df.index)
    x = design(df, with_market)
    usable = df["mkt_p1"].notna() if with_market else pd.Series(True, index=df.index)
    for year in sorted(df["year"].unique()):
        if year < first_year:
            continue
        train = (df["year"] < year) & usable
        test = (df["year"] == year) & usable
        if train.sum() < min_train or not test.any():
            continue
        model = LogisticRegression(C=c, fit_intercept=False, max_iter=500)
        model.fit(x[train], df.loc[train, "y"])
        out[test] = model.predict_proba(x[test])[:, 1]
    return out
