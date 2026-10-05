import numpy as np
import pandas as pd

from tennis import context


def rows(*matches):
    """(дата, p1, p2, y, best_of, sets1, sets2, отказал ли е загубилият)"""
    df = pd.DataFrame(matches, columns=["date", "p1", "p2", "y", "best_of", "sets1", "sets2", "retired"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_rest_days_h2h_and_no_same_day_leak():
    df = rows(("2024-03-01", "A", "B", 1, 3, 2, 0, False),
              ("2024-03-01", "A", "C", 1, 3, 2, 1, False),      # същия ден: не вижда предишния
              ("2024-03-11", "A", "B", 0, 3, 0, 2, False))
    c = context.add(df)
    assert np.isnan(c.loc[0, "rest1"]) and np.isnan(c.loc[1, "rest1"])      # първи мач за A - без rest
    assert c.loc[2, "rest1"] == 10 and c.loc[2, "rest2"] == 10
    assert c["h2h"].tolist() == [0, 0, 1]                                    # A-B вече са играли веднъж


def test_retirement_and_long_match_flags_point_at_the_right_player():
    df = rows(("2024-03-01", "A", "B", 1, 3, 2, 1, True),       # A печели, B се отказва след 3 сета (дълъг мач)
              ("2024-03-05", "A", "B", 1, 3, 2, 0, False))
    c = context.add(df)
    assert not c.loc[1, "ret1"] and c.loc[1, "ret2"]             # отказал се е B, не победителят A
    assert c.loc[1, "long1"] and c.loc[1, "long2"]               # 2:1 = 3 сета = дълъг и за двамата


def test_best_of_five_long_means_four_sets_or_more():
    df = rows(("2024-03-01", "A", "B", 1, 5, 3, 1, False),      # 4 сета в bo5 = дълъг
              ("2024-03-02", "A", "B", 1, 5, 3, 0, False))
    c = context.add(df)
    assert c.loc[1, "long1"] and c.loc[1, "long2"]
    df2 = rows(("2024-03-01", "A", "B", 1, 5, 3, 0, False), ("2024-03-02", "A", "B", 1, 5, 3, 0, False))
    assert not context.add(df2).loc[1, "long1"]                  # 3:0 в bo5 не е дълъг


def test_offseason_is_not_a_break():
    df = rows(("2023-11-20", "A", "B", 1, 3, 2, 0, False), ("2024-01-10", "A", "B", 1, 3, 2, 0, False),
              ("2024-06-10", "A", "B", 1, 3, 2, 0, False))
    c = context.add(df)
    assert bool(c.loc[1, "offseason"]) and not bool(c.loc[2, "offseason"])
