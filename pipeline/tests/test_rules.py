import pandas as pd
import pytest

from tennis import dataset, rules, simulate, stack


def test_kind_bands_follow_the_masters_thresholds():
    assert rules.kind(1.39) is None
    assert rules.kind(1.40) == "safe" and rules.kind(1.80) == "safe"
    assert rules.kind(1.81) == "risky"


def frame(**cols):
    n = len(next(iter(cols.values())))
    base = {"date": pd.to_datetime(["2024-01-01"] * n), "y": [1] * n, "n1": [50] * n, "n2": [50] * n,
            "tournament": ["T"] * n}
    base.update(cols)
    return pd.DataFrame(base)


def test_tip_follows_the_model_not_the_market():
    # моделът: p2 е фаворит (p1 = 0.30); пазарът: p1 е фаворит (1.50 срещу 2.60)
    df = frame(model=[0.30], o1_Avg=[1.50], o2_Avg=[2.60], y=[0])
    t = rules.tips(df, "model", "Avg")
    assert t["side"].tolist() == [2] and t["p"].iloc[0] == pytest.approx(0.70)
    assert t["odds"].iloc[0] == 2.60 and t["kind"].iloc[0] == "risky" and bool(t["won"].iloc[0])


def test_no_tip_under_min_odds_or_for_players_the_model_barely_knows():
    df = frame(model=[0.80, 0.80, 0.80], o1_Avg=[1.30, 1.60, 1.60], o2_Avg=[3.5, 2.4, 2.4], n1=[50, 50, 3])
    t = rules.tips(df, "model", "Avg")
    assert t.index.tolist() == [1]           # първият: коеф. 1.30 < 1.40; третият: играч с 3 мача


def test_build_columns_one_match_per_tournament_and_each_match_once():
    cands = [{"id": i, "tournament": f"T{i % 4}", "p": 0.9 - i * 0.01, "odds": 1.5, "won": True} for i in range(12)]
    cols = rules.build_columns(cands, size=3, max_columns=4)
    assert all(len({c["tournament"] for c in col}) == 3 for col in cols)
    ids = [c["id"] for col in cols for c in col]
    assert len(ids) == len(set(ids))
    assert len(cols) == 4                    # 12 мача от 4 турнира = точно 4 колонки по 3 различни турнира
    more = rules.build_columns(cands, size=3, max_columns=10)
    assert len(more) == 4                    # повече колонки няма откъде - мачовете свършват


def test_no_column_when_not_enough_distinct_tournaments():
    cands = [{"id": i, "tournament": "same", "p": 0.8, "odds": 1.5, "won": True} for i in range(5)]
    assert rules.build_columns(cands) == []


def test_evaluate_columns_known_numbers():
    day1 = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"] * 3), "p": [0.7] * 3, "odds": [1.5] * 3,
                         "won": [True] * 3, "tournament": ["A", "B", "C"], "kind": ["safe"] * 3})
    day2 = day1.assign(date=pd.to_datetime(["2024-01-02"] * 3), won=[True, True, False])
    out = rules.evaluate_columns(pd.concat([day1, day2], ignore_index=True))
    assert out["columns"] == 2 and out["passed"] == pytest.approx(0.5)
    assert out["claimed"] == pytest.approx(0.343)
    assert out["return"] == pytest.approx(0.5 * 1.5 ** 3)            # (3.375 + 0) / 2
    assert out["overrate"] == pytest.approx(0.343 / 0.5)


def test_efficient_market_the_masters_rules_lose_the_margin():
    m = simulate.world(seed=21, market_sees="all", n_players=200, years=7, per_year=4000)
    df = dataset.build(m)
    df["own"] = stack.walk_forward(df, with_market=False, first_year=2017)
    t = rules.tips(df, "own", "Avg")
    ev = rules.evaluate_tips(t)
    assert ev["всички"]["n"] > 3000 and ev["всички"]["roi"] < 0     # няма предимство - губи около маржа
    cols = rules.evaluate_columns(t)
    assert cols["columns"] > 50 and cols["return"] < 1.0            # маржът се умножава на всеки мач


def test_exclude_odds_band_removes_the_toto_tips():
    df = frame(model=[0.80, 0.80, 0.80], o1_Avg=[1.45, 1.70, 1.50], o2_Avg=[2.9, 2.2, 2.7], y=[1, 1, 1])
    t = rules.tips(df, "model", "Avg", exclude_odds=(1.30, 1.55))
    assert t.index.tolist() == [1]            # 1.45 и 1.50 са в лентата на „тото“ и отпадат


def test_agree_col_keeps_only_tips_where_model_and_market_pick_the_same_player():
    df = frame(model=[0.70, 0.70, 0.30], mkt=[0.60, 0.40, 0.45], o1_Avg=[1.6, 2.2, 2.4], o2_Avg=[2.4, 1.7, 1.6])
    t = rules.tips(df, "model", "Avg", agree_col="mkt")
    assert t.index.tolist() == [0, 2]         # реда 1: моделът за p1, пазарът за p2 - несъгласие
