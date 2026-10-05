import pandas as pd
import pytest

from tennis import staking


def test_kelly_known_values():
    assert staking.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)         # (1.2 - 1) / 1
    assert staking.kelly_fraction(0.5, 2.0) == 0.0                        # честна цена - без залог
    assert staking.kelly_fraction(0.4, 2.0) == 0.0                        # отрицателно - никога залог
    assert staking.kelly_fraction(0.7, 1.0) == 0.0                        # коефициент 1.0 - няма печалба


def test_stake_is_fractional_and_capped():
    assert staking.stake_fraction(0.6, 2.0, kelly_part=0.25, cap=0.02) == pytest.approx(0.02)   # 0.05 -> таван
    assert staking.stake_fraction(0.51, 2.0, kelly_part=0.25, cap=0.5) == pytest.approx(0.25 * 0.02)


def test_simulate_never_bets_without_edge():
    bets = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02"]), "p": [0.5, 0.4],
                         "odds": [2.0, 2.0], "won": [True, True]})
    out = staking.simulate(bets, bankroll=100)
    assert out["bets"] == 0 and out["final"] == 100


def test_simulate_compounds_and_tracks_drawdown():
    bets = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                         "p": [0.6] * 3, "odds": [2.0] * 3, "won": [True, False, True]})
    out = staking.simulate(bets, bankroll=100, kelly_part=0.25, cap=0.02)
    assert out["bets"] == 3
    expected = 100 * 1.02 * 0.98 * 1.02            # 2% от банката всеки път; печели 1:1
    assert out["final"] == pytest.approx(expected, rel=1e-3)
    assert 0 < out["max_drawdown"] < 0.03
