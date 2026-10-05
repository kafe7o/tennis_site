import numpy as np
import pandas as pd

import event_lab


def frame():
    # p1 = фаворит при p>=0.5; (p1, y, sets1, sets2, best_of)
    rows = [(0.70, 1, 2, 0, 3),    # фаворитът печели 2:0
            (0.70, 0, 1, 2, 3),    # фаворитът губи 1:2
            (0.30, 0, 0, 2, 3),    # p2 е фаворит и печели 2:0 (p1 губи 0:2)
            (0.70, 1, 3, 2, 5),    # до 5 сета
            (0.70, 0, 0, 3, 5)]    # фаворитът губи 0:3 без взет сет
    df = pd.DataFrame(rows, columns=["mkt_p1", "y", "sets1", "sets2", "best_of"])
    df["games1"], df["games2"] = 20.0, 18.0
    return df


def test_event_labels_are_oriented_to_the_favourite():
    lab = event_lab.orient(frame(), "mkt_p1")
    assert lab["fav_won"].tolist() == [True, False, True, True, False]
    assert lab["straight"].tolist() == [True, False, True, False, False]
    assert lab["dog_set"].tolist() == [False, True, False, True, True]        # обратното на „без загубен сет“
    assert lab["distance"].tolist() == [False, True, False, True, False]      # 3 сета в bo3, 5 в bo5
    assert lab["fav_set"].tolist() == [True, True, True, True, False]
    assert lab["fav_p"].round(2).tolist() == [0.70, 0.70, 0.70, 0.70, 0.70]


def test_games_lines_only_for_best_of_three():
    lab = event_lab.orient(frame(), "mkt_p1")
    assert lab["games_over_21.5"].tolist()[:3] == [True, True, True] and np.isnan(lab["games_over_21.5"].iloc[3])
