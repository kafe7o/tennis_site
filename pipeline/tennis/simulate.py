"""
СИМУЛИРАН свят - само за да се провери, че системата казва истината. НЕ са реални мачове и
резултатите от него нищо не казват за реалния тенис.

Знаем истината (сила на играч, ефект на настилка, дрейф във времето), затова знаем и къде
предимство ИМА и къде НЯМА:
  market_sees="all"         пазарът знае всичко (+ малък шум): предимство няма за никого.
                            Проверката трябва да покаже доход около -маржа и t не над ~2.
  market_sees="no_surface"  пазарът не знае ефекта на настилката: Elo по настилка трябва да
                            го хване - лог-загубата да падне под пазарната, доходът да е положителен.
Ако проверката не разграничава двата случая, на нея не може да се вярва и върху реални данни.
"""

import numpy as np
import pandas as pd

SURFACES = ("Hard", "Clay", "Grass")


def world(n_players=250, years=8, per_year=4000, seed=0, market_sees="all",
          noise=0.05, margin=0.04, surface_sd=0.35, drift_sd=0.15, start_year=2015):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 0.9, n_players)
    surf = rng.normal(0, surface_sd, (n_players, len(SURFACES)))
    weight = np.exp(0.5 * skill)
    weight /= weight.sum()
    names = np.array([f"P{i:03d}" for i in range(n_players)])
    rows = []
    for yi in range(years):
        year = start_year + yi
        skill = skill + rng.normal(0, drift_sd, n_players)
        a = rng.choice(n_players, per_year, p=weight)
        b = rng.choice(n_players, per_year, p=weight)
        keep = a != b
        a, b = a[keep], b[keep]
        s = rng.integers(0, len(SURFACES), len(a))
        bo5 = (rng.random(len(a)) < 0.12) & (s != 1) | (rng.random(len(a)) < 0.03)
        boost = np.where(bo5, 1.15, 1.0)
        true_logit = ((skill[a] + surf[a, s]) - (skill[b] + surf[b, s])) * boost
        seen = true_logit if market_sees == "all" else (skill[a] - skill[b]) * boost
        p_mkt = 1.0 / (1.0 + np.exp(-(seen + rng.normal(0, noise, len(a)))))
        a_wins = rng.random(len(a)) < 1.0 / (1.0 + np.exp(-true_logit))
        noisy = skill + rng.normal(0, 0.3, n_players)
        rank = (-noisy).argsort().argsort() + 1
        pts = 1000.0 * np.exp(2.0 * noisy)
        days = pd.to_datetime(f"{year}-01-01") + pd.to_timedelta(rng.integers(0, 360, len(a)), unit="D")

        def odds(p, m):
            return 1.0 / (p * (1.0 + m)), 1.0 / ((1.0 - p) * (1.0 + m))

        avg_a, avg_b = odds(p_mkt, margin)
        ps_a, ps_b = odds(p_mkt, margin * 0.6)
        win = np.where(a_wins, a, b)
        lose = np.where(a_wins, b, a)
        win_odds = lambda oa, ob: np.where(a_wins, oa, ob)
        lose_odds = lambda oa, ob: np.where(a_wins, ob, oa)
        rows.append(pd.DataFrame({
            "date": days, "winner": names[win], "loser": names[lose],
            "surface": np.array(SURFACES)[s], "best_of": np.where(bo5, 5, 3),
            "wrank": rank[win], "lrank": rank[lose], "wpts": pts[win], "lpts": pts[lose],
            "AvgW": win_odds(avg_a, avg_b), "AvgL": lose_odds(avg_a, avg_b),
            "MaxW": win_odds(avg_a, avg_b) * 1.02, "MaxL": lose_odds(avg_a, avg_b) * 1.02,
            "PSW": win_odds(ps_a, ps_b), "PSL": lose_odds(ps_a, ps_b),
            "B365W": win_odds(avg_a, avg_b), "B365L": lose_odds(avg_a, avg_b)}))
    return pd.concat(rows, ignore_index=True).sort_values("date", kind="stable").reset_index(drop=True)
