import numpy as np
import pandas as pd
import pytest

from tennis import elo, simulate


def two(w_games, l_games, retired=False):
    return pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-05"]), "winner": ["A", "A"], "loser": ["B", "B"],
                         "surface": ["Hard", "Hard"], "best_of": [3, 3], "w_games": [w_games] * 2, "l_games": [l_games] * 2,
                         "retired": [retired] * 2})


def test_margin_weight_zero_changes_nothing():
    m = simulate.world(n_players=40, years=2, per_year=500, seed=8)
    base = elo.walk_forward(m)
    m2 = m.assign(w_games=12.0, l_games=7.0, retired=False)
    zero = elo.walk_forward(m2, elo.EloParams(margin_weight=0.0))
    assert base["elo_p1"].to_numpy() == pytest.approx(zero["elo_p1"].to_numpy())


def test_bigger_margin_moves_ratings_more():
    params = elo.EloParams(margin_weight=1.0)
    rout = elo.walk_forward(two(12, 0), params)          # 6-0 6-0
    close = elo.walk_forward(two(13, 12), params)        # 7-6 7-6 (по-близо до 0.5)
    # втората прогноза за мача на A срещу B е по-силна след разгром
    assert rout["elo_p1"].iloc[1] > close["elo_p1"].iloc[1] > 0.5


def test_retired_match_counts_as_a_plain_win():
    params = elo.EloParams(margin_weight=1.0)
    ret = elo.walk_forward(two(3, 3, retired=True), params)     # отказване при 3:3 - иначе почти равен резултат
    plain = elo.walk_forward(two(3, 3, retired=True), elo.EloParams(margin_weight=0.0))
    assert ret["elo_p1"].iloc[1] == pytest.approx(plain["elo_p1"].iloc[1])


def test_score_is_clipped_between_half_and_one():
    class M:
        w_games, l_games, retired = 5, 20, False                 # победител с по-малко гейма: не пада под 0.5
    assert elo._score(M, 1.0) == 0.5
    M.w_games, M.l_games = 12, 0
    assert elo._score(M, 1.0) == 1.0
    assert elo._score(M, 0.0) == 1.0
