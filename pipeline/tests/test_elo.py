import numpy as np
import pandas as pd
import pytest

from tennis import dataset, elo, simulate


def tiny():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"]),
        "winner": ["B", "D", "B", "A"], "loser": ["A", "C", "D", "C"],
        "surface": ["Clay", "Clay", "Hard", "Hard"], "best_of": [3, 3, 3, 5]})


def test_predict_is_symmetric_even_with_bo5_boost():
    m = elo.Elo(elo.EloParams(bo5_boost=1.2))
    for _ in range(5):
        m.update("A", "B", "Clay")
    assert m.predict("A", "B", "Clay", 5) + m.predict("B", "A", "Clay", 5) == pytest.approx(1.0)
    assert m.predict("A", "B", "Clay", 5) > m.predict("A", "B", "Clay", 3) > 0.5


def test_winner_goes_up_loser_goes_down():
    m = elo.Elo()
    m.update("A", "B", "Hard")
    assert m.rating("A") > elo.BASE > m.rating("B")
    assert m.surface_rating("A", "Hard") > elo.BASE > m.surface_rating("B", "Hard")
    assert m.surface_rating("A", "Clay") == m.rating("A")        # друга настилка - от общата


def test_surface_rating_starts_from_overall_not_from_1500():
    m = elo.Elo()
    for _ in range(30):
        m.update("A", "B", "Hard")
    overall = m.rating("A")
    m.update("A", "C", "Clay")
    assert m.surface_rating("A", "Clay") > overall - 1            # не тръгва от 1500


def test_no_lookahead_future_matches_do_not_change_past_predictions():
    base = simulate.world(n_players=40, years=2, per_year=600, seed=3)
    cut = base["date"].iloc[len(base) // 2]
    first = elo.walk_forward(base)
    changed = base.copy()
    late = changed["date"] > cut
    changed.loc[late, ["winner", "loser"]] = changed.loc[late, ["loser", "winner"]].to_numpy()
    second = elo.walk_forward(changed)
    early = base["date"] <= cut
    # всичко до деня на сечението (включително) е със същите прогнози: после обърнахме резултатите
    assert first.loc[early, "elo_p1"].to_numpy() == pytest.approx(second.loc[early, "elo_p1"].to_numpy())


def test_same_day_matches_do_not_see_each_other():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 2), "winner": ["A", "A"], "loser": ["B", "C"],
        "surface": ["Hard", "Hard"], "best_of": [3, 3]})
    out = elo.walk_forward(df)
    # вторият мач на деня НЕ вижда първия: A е със същата оценка и в двата
    assert out["n1"].tolist() == [0, 0]
    assert out["elo_p1"].iloc[0] == out["elo_p1"].iloc[1] == pytest.approx(0.5)


def test_label_does_not_leak_the_winner():
    out = elo.walk_forward(tiny())
    # p1 е по азбучен ред - y е 1 само когато победителят е първият по азбука
    assert (out["p1"] < out["p2"]).all()
    # B-A: p1=A, победил B -> 0 | D-C: p1=C, победил D -> 0 | B-D: p1=B, победил B -> 1 | A-C: p1=A, победил A -> 1
    assert out["y"].tolist() == [0, 0, 1, 1]


def test_missing_columns_fail_loudly():
    with pytest.raises(ValueError):
        elo.walk_forward(tiny().drop(columns=["surface"]))


def test_converges_towards_true_strength_in_a_simulated_world():
    m = simulate.world(n_players=60, years=6, per_year=3000, seed=5, surface_sd=0.0, drift_sd=0.0)
    df = dataset.build(m)
    late = df[df["year"] >= 2019]
    # на късните години оценките са добри: по-добре от монета (0.693) с над 0.05 лог-загуба
    y = late["y"].to_numpy()
    p = np.clip(late["elo_p1"].to_numpy(), 1e-9, 1 - 1e-9)
    loss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    assert loss < 0.693 - 0.05
