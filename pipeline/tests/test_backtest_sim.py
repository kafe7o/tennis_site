"""
Най-важният тест: проверката назад трябва да казва ИСТИНАТА в свят, в който я знаем.
Ако не разграничава "няма предимство" от "има предимство", на нея не може да се вярва и върху реални данни.
"""

import pytest

from tennis import backtest, dataset, simulate, stack


def run(sees, seed):
    m = simulate.world(seed=seed, market_sees=sees, n_players=200, years=7, per_year=4000)
    df = dataset.build(m)
    df["stack_nomkt"] = stack.walk_forward(df, with_market=False, first_year=2017)
    df["stack_mkt"] = stack.walk_forward(df, with_market=True, first_year=2017)
    return df[df["year"] >= 2017]


@pytest.fixture(scope="module")
def efficient():
    return run("all", 11)


@pytest.fixture(scope="module")
def blind():
    return run("no_surface", 11)


def test_efficient_market_nothing_beats_it(efficient):
    cmp = backtest.compare(efficient, ["elo_p1", "stack_nomkt", "stack_mkt"])
    assert cmp.loc["elo_p1", "t"] > 5                 # Elo без пазара е ясно по-лош
    assert cmp.loc["stack_nomkt", "t"] > 5
    # с пазара най-много леко подобрение (в симулацията пазарът има шум 0.05 - почти нищо)
    assert cmp.loc["stack_mkt", "vs_ref"] > -0.002


def test_efficient_market_betting_loses_about_the_margin(efficient):
    bets = backtest.place(efficient, "stack_nomkt", "Avg", ev_min=0.10)
    s = backtest.summary(bets)
    assert s["n"] > 500
    assert s["roi"] < 0 and s["t"] < 1                # губи; без значимо положителен доход


def test_efficient_market_protocol_does_not_find_an_edge(efficient):
    r = backtest.protocol(efficient, "stack_mkt", "Avg", select_until="2020-12-31", min_bets=200)
    assert r["test"] is not None
    assert not (r["test"]["roi"] > 0 and r["test"]["t"] > 2.5)


def test_blind_market_edge_is_detected(blind):
    cmp = backtest.compare(blind, ["elo_p1", "stack_nomkt", "stack_mkt"])
    assert cmp.loc["stack_mkt", "t"] < -5             # статистически по-добре от пазара
    assert cmp.loc["stack_mkt", "vs_ref"] < -0.005


def test_blind_market_protocol_profits_out_of_sample(blind):
    r = backtest.protocol(blind, "stack_mkt", "Avg", select_until="2020-12-31", min_bets=200)
    assert r["test"]["roi"] > 0.05 and r["test"]["t"] > 3


def test_baselines_report_favourite_and_underdog(efficient):
    b = backtest.baselines(efficient, "Avg")
    assert set(b) == {"фаворит", "аутсайдер"}
    assert b["фаворит"]["hit"] > 0.6 > b["аутсайдер"]["hit"]
