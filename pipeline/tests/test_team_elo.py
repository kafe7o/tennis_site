import numpy as np
import pandas as pd
import pytest

from tennis import team_elo


def games(*rows):
    df = pd.DataFrame(rows, columns=["date", "home", "away", "home_score", "away_score"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def test_home_advantage_makes_the_first_prediction_favour_the_home_team():
    out = team_elo.walk_forward(games(("2024-04-01", "A", "B", 3, 2)), team_elo.TeamEloParams(hfa=30.0))
    assert out["elo_p1"].iloc[0] > 0.5 and out["elo_diff"].iloc[0] == pytest.approx(30.0)
    out0 = team_elo.walk_forward(games(("2024-04-01", "A", "B", 3, 2)), team_elo.TeamEloParams(hfa=0.0))
    assert out0["elo_p1"].iloc[0] == pytest.approx(0.5)


def test_bigger_win_moves_ratings_more_when_mov_is_on():
    small = games(("2024-04-01", "A", "B", 3, 2), ("2024-04-03", "A", "B", 1, 0))
    big = games(("2024-04-01", "A", "B", 12, 0), ("2024-04-03", "A", "B", 1, 0))
    on = team_elo.TeamEloParams(mov=True)
    assert (team_elo.walk_forward(big, on)["elo_p1"].iloc[1] > team_elo.walk_forward(small, on)["elo_p1"].iloc[1])
    off = team_elo.TeamEloParams(mov=False)
    assert (team_elo.walk_forward(big, off)["elo_p1"].iloc[1] == pytest.approx(
        team_elo.walk_forward(small, off)["elo_p1"].iloc[1]))


def test_doubleheader_second_game_does_not_see_the_first():
    out = team_elo.walk_forward(games(("2024-04-01", "A", "B", 5, 0), ("2024-04-01", "A", "B", 5, 0)))
    assert out["n1"].tolist() == [0, 0] and out["elo_p1"].iloc[0] == pytest.approx(out["elo_p1"].iloc[1])


def test_season_reversion_pulls_ratings_back_to_the_mean():
    rows = [("2023-04-%02d" % d, "A", "B", 5, 0) for d in range(1, 29)] + [("2024-04-01", "A", "B", 5, 0)]
    keep = team_elo.walk_forward(games(*rows), team_elo.TeamEloParams(revert=0.0, hfa=0.0))
    pull = team_elo.walk_forward(games(*rows), team_elo.TeamEloParams(revert=0.5, hfa=0.0))
    assert pull["elo_p1"].iloc[-1] < keep["elo_p1"].iloc[-1]          # след реверсията прогнозата за A е по-малка


def test_no_lookahead_future_results_do_not_change_past_predictions():
    rng = np.random.default_rng(1)
    rows = [(f"2024-04-{1 + i // 4:02d}", f"T{rng.integers(0, 8)}", f"U{rng.integers(0, 8)}", int(rng.integers(0, 9)), int(rng.integers(0, 9)))
            for i in range(80)]
    rows = [r for r in rows if r[3] != r[4]]
    base = games(*rows)
    flipped = base.copy()
    late = flipped["date"] > pd.Timestamp("2024-04-10")
    flipped.loc[late, ["home_score", "away_score"]] = flipped.loc[late, ["away_score", "home_score"]].to_numpy()
    a, b = team_elo.walk_forward(base), team_elo.walk_forward(flipped)
    early = base["date"] <= pd.Timestamp("2024-04-10")
    assert a.loc[early, "elo_p1"].to_numpy() == pytest.approx(b.loc[early, "elo_p1"].to_numpy())


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError):
        team_elo.walk_forward(games(("2024-04-01", "A", "B", 3, 2)).drop(columns=["away_score"]))
