import numpy as np
import pytest

from tennis import market


def test_devig_sums_to_one_both_methods():
    for method in ("proportional", "power"):
        p = market.devig(1.80, 2.05, method)
        q = market.devig(2.05, 1.80, method)
        assert p + q == pytest.approx(1.0, abs=1e-9)


def test_fair_odds_are_returned_unchanged():
    # 1/1.6 + 1/2.667 = 1.0 - никакъв марж, нищо за махане
    p = market.devig(1.6, 1 / (1 - 1 / 1.6), "power")
    assert p == pytest.approx(1 / 1.6, abs=1e-6)


def test_power_puts_more_margin_on_the_underdog():
    # a = 0.8 (фаворит), b = 0.25: степенният метод дава ПО-ВИСОКА честна вероятност на фаворита
    fav_odds, dog_odds = 1 / 0.8, 1 / 0.25
    assert market.devig(fav_odds, dog_odds, "power") > market.devig(fav_odds, dog_odds, "proportional")


def test_vectorised_matches_scalar():
    o1, o2 = np.array([1.5, 2.2, 1.9]), np.array([2.7, 1.7, 1.95])
    vec = market.devig(o1, o2)
    assert [float(market.devig(a, b)) for a, b in zip(o1, o2)] == pytest.approx(list(vec))


def test_margin():
    assert market.margin(1.80, 2.05) == pytest.approx(1 / 1.80 + 1 / 2.05 - 1)


def test_unknown_method_fails_loudly():
    with pytest.raises(ValueError):
        market.devig(1.8, 2.0, "magic")
