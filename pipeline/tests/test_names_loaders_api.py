import pandas as pd
import pytest

from tennis import loaders, names, odds_api


# --- имена ---
def test_td_key():
    assert names.td_key("Djokovic N.") == ("djokovic", "n")
    assert names.td_key("Del Potro J.M.") == ("del potro", "jm")
    assert names.td_key("Auger-Aliassime F.") == ("auger aliassime", "f")


def test_match_full_name_to_td_key():
    keys = {names.td_key(n) for n in ["Djokovic N.", "Del Potro J.M.", "Nadal R.", "Auger-Aliassime F."]}
    assert names.match("Novak Djokovic", keys) == ("djokovic", "n")
    assert names.match("Juan Martin Del Potro", keys) == ("del potro", "jm")
    assert names.match("Félix Auger-Aliassime", keys) == ("auger aliassime", "f")
    assert names.match("Rafael Nadal", keys) == ("nadal", "r")


def test_ambiguous_or_unknown_name_gives_none():
    keys = {names.td_key(n) for n in ["Zhang Z.", "Zhang Y.", "Zhang Z.Z."]}
    assert names.match("Zhizhen Zhang", keys) is None            # Z. и Z.Z. - не е ясно
    assert names.match("Jannik Sinner", keys) is None


# --- зареждане ---
def td_frame(**override):
    base = {"Date": ["03/01/2024", "04/01/2024"], "Surface": ["Hard", "Clay"], "Round": ["1st Round", "2nd Round"],
            "Winner": ["Djokovic N.", "Nadal R."], "Loser": ["Zhang Z.", "Alcaraz C."], "WRank": [1, 5],
            "LRank": [60, 2], "Best of": [3, 5], "Comment": ["Completed", "Retired"],
            "AvgW": [1.1, 3.0], "AvgL": [7.0, 1.4]}
    base.update(override)
    return pd.DataFrame(base)


def test_tennis_data_loader(tmp_path):
    path = tmp_path / "2024.csv"
    td_frame().to_csv(path, index=False)
    out = loaders.tennis_data(path, "ATP")
    assert out["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]   # ден/месец
    assert out["retired"].tolist() == [False, True]
    assert out["best_of"].tolist() == [3, 5]
    assert out["AvgW"].tolist() == [1.1, 3.0] and out["B365W"].isna().all()
    assert out["tour"].eq("ATP").all()


def test_tennis_data_unknown_surface_fails_loudly(tmp_path):
    path = tmp_path / "x.csv"
    td_frame(Surface=["Hard", "Moon"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="непозната настилка"):
        loaders.tennis_data(path, "ATP")


def test_tennis_data_missing_column_fails_loudly(tmp_path):
    path = tmp_path / "x.csv"
    td_frame().drop(columns=["Winner"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="липсват колони"):
        loaders.tennis_data(path, "ATP")


def test_sackmann_loader_orders_rounds_inside_a_tournament(tmp_path):
    path = tmp_path / "atp_matches_2024.csv"
    pd.DataFrame({"tourney_name": ["X", "X", "X"], "surface": ["Clay"] * 3,
                  "tourney_date": [20240101] * 3, "winner_name": ["A B", "C D", "A B"],
                  "loser_name": ["E F", "G H", "C D"], "round": ["R32", "R64", "R16"],
                  "score": ["6-4 6-4", "6-1 RET", "W/O"], "best_of": [3, 3, 3]}).to_csv(path, index=False)
    out = loaders.sackmann(path, "ATP")
    assert out["date"].tolist() == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-04")]
    assert out["retired"].tolist() == [False, True, False] and out["walkover"].tolist() == [False, False, True]


def test_sackmann_unknown_round_fails_loudly(tmp_path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"tourney_name": ["X"], "surface": ["Clay"], "tourney_date": [20240101], "winner_name": ["A B"],
                  "loser_name": ["C D"], "round": ["Q9"], "score": ["6-0 6-0"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="непознат кръг"):
        loaders.sackmann(path, "ATP")


# --- odds API (без мрежа, със съставено събитие) ---
EVENT = {"id": "e1", "home_team": "Jannik Sinner", "away_team": "Carlos Alcaraz",
         "commence_time": "2026-10-06T12:00:00Z",
         "bookmakers": [
             {"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                 {"name": "Carlos Alcaraz", "price": 2.10}, {"name": "Jannik Sinner", "price": 1.80}]}]},
             {"key": "bet365", "markets": [{"key": "h2h", "outcomes": [
                 {"name": "Jannik Sinner", "price": 1.85}, {"name": "Carlos Alcaraz", "price": 2.00}]}]},
             {"key": "broken", "markets": [{"key": "totals", "outcomes": []}]}]}


def test_prices_averages_and_maxes_per_player():
    p = odds_api.prices(EVENT)
    assert p["n_books"] == 2
    assert p["books"]["pinnacle"] == (1.80, 2.10)
    assert p["avg"] == pytest.approx((1.825, 2.05))
    assert p["max"] == (1.85, 2.10)


def test_prices_none_without_h2h():
    assert odds_api.prices({"home_team": "A", "away_team": "B", "commence_time": "x", "bookmakers": []}) is None
