"""
Стандартният вид на мачовете и обръщането им към "играч 1 / играч 2".

Входът (matches) е по един ред на изигран мач, победителят и загубилият са отделно:
  date, winner, loser, surface, best_of           - задължителни
  wrank, lrank, wpts, lpts                        - ранкинг (по желание)
  <книга>W, <книга>L                              - коефициенти за победителя и за загубилия
                                                    (B365, PS, Max, Avg)
  walkover (bool)                                 - мачове без игра: махат се изцяло

Изходът е обърнат: p1 е първият по азбучен ред, y = 1 ако е победил той. Така нито един
признак не издава отговора (в сурови данни "победителят" винаги е на първо място).
"""

import numpy as np
import pandas as pd

from . import elo, market

BOOKS = ("B365", "PS", "Max", "Avg")


def build(matches, params=None, power_book="Avg"):
    """Мачовете, обърнати към p1/p2 + Elo признаците (без поглед напред) + пазарът без маржа."""
    m = matches.copy()
    if "walkover" in m:
        m = m[~m["walkover"].fillna(False).astype(bool)]
    m = m.sort_values("date", kind="stable").reset_index(drop=True)
    feats = elo.walk_forward(m, params)
    won = feats["y"].to_numpy() == 1
    out = pd.concat([m[["date", "surface", "best_of"]], feats], axis=1)
    out["year"] = out["date"].dt.year

    def pick(w, l):
        if w not in m or l not in m:
            return pd.Series(np.nan, index=m.index), pd.Series(np.nan, index=m.index)
        return (pd.Series(np.where(won, m[w], m[l]), index=m.index),
                pd.Series(np.where(won, m[l], m[w]), index=m.index))

    out["rank1"], out["rank2"] = pick("wrank", "lrank")
    out["pts1"], out["pts2"] = pick("wpts", "lpts")
    for book in BOOKS:
        out[f"o1_{book}"], out[f"o2_{book}"] = pick(f"{book}W", f"{book}L")
    for col in ("tour", "tournament", "round", "retired"):
        if col in m:
            out[col] = m[col]

    # пазарът без маржа - по книгата за вероятностите (средната е най-стабилна)
    o1, o2 = out[f"o1_{power_book}"], out[f"o2_{power_book}"]
    ok = (o1 > 1.0) & (o2 > 1.0)
    out["mkt_p1"] = np.nan
    out.loc[ok, "mkt_p1"] = market.devig(o1[ok].to_numpy(), o2[ok].to_numpy(), "power")
    return out
