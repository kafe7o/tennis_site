"""Формат на реалния tennis-data.co.uk (сверен с 30 файла на 2026-10-05): ниво, корт, геймове, сетове, мач без дата."""

import warnings

import numpy as np
import pandas as pd
import pytest

from tennis import elo, loaders


def frame(**override):
    base = {"Date": ["2024-01-03", "2024-01-04"], "Surface": ["Hard", "Clay"], "Round": ["1st Round", "Final"],
            "Winner": ["Djokovic N.", "Nadal R."], "Loser": ["Zhang Z.", "Alcaraz C."], "Series": ["ATP250", "Grand Slam"],
            "Court": ["Outdoor", "Indoor"], "Best of": [3, 5], "Comment": ["Completed", "Completed"],
            "W1": [6, 6], "L1": [4, 3], "W2": [6, 7], "L2": [3, 6], "W3": [np.nan, 6], "L3": [np.nan, 4],
            "W4": [np.nan, np.nan], "L4": [np.nan, np.nan], "W5": [np.nan, np.nan], "L5": [np.nan, np.nan],
            "Wsets": [2, 3], "Lsets": [0, 0], "AvgW": [1.1, 3.0], "AvgL": [7.0, 1.4]}
    base.update(override)
    return pd.DataFrame(base)


def test_series_court_games_and_sets_are_loaded(tmp_path):
    path = tmp_path / "x.csv"
    frame().to_csv(path, index=False)
    out = loaders.tennis_data(path, "ATP")
    assert out["series"].tolist() == ["ATP250", "Grand Slam"]
    assert out["court"].tolist() == ["Outdoor", "Indoor"]
    assert out["w_games"].tolist() == [12.0, 19.0] and out["l_games"].tolist() == [7.0, 13.0]
    assert out["w_sets"].tolist() == [2, 3] and out["l_sets"].tolist() == [0, 0]


def test_wta_tier_column_is_used_as_series(tmp_path):
    path = tmp_path / "x.csv"
    frame().rename(columns={"Series": "Tier"}).to_csv(path, index=False)
    assert loaders.tennis_data(path, "WTA")["series"].tolist() == ["ATP250", "Grand Slam"]


def test_match_without_date_is_dropped_with_a_visible_warning(tmp_path):
    path = tmp_path / "x.csv"
    frame(Date=["2024-01-03", None]).to_csv(path, index=False)
    with pytest.warns(UserWarning, match="без дата"):
        out = loaders.tennis_data(path, "WTA")
    assert len(out) == 1 and out["date"].notna().all()


def test_elo_refuses_matches_without_a_date():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", None]), "winner": ["A", "B"], "loser": ["C", "D"],
                       "surface": ["Hard", "Hard"], "best_of": [3, 3]})
    with pytest.raises(ValueError, match="без дата"):
        elo.walk_forward(df)
