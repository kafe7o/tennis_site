"""
Размер на залога. Правилото на Kelly дава оптималната част от банката САМО ако вероятността
е вярна. Нашата е оценка и грешката в нея е най-опасното нещо в системата, затова:
  - малка част от Kelly (по подразбиране 1/4),
  - таван на залога като дял от банката (по подразбиране 2%),
  - никакъв залог при отрицателно очаквано (EV <= 0).
"""

import numpy as np


def kelly_fraction(p, odds):
    """Оптимален дял от банката при вероятност p и десетичен коефициент odds (0 ако няма предимство)."""
    b = odds - 1.0
    f = (p * odds - 1.0) / b if b > 0 else 0.0
    return max(0.0, float(f))


def stake_fraction(p, odds, kelly_part=0.25, cap=0.02):
    return min(cap, kelly_part * kelly_fraction(p, odds))


def simulate(bets, bankroll=100.0, kelly_part=0.25, cap=0.02):
    """
    bets: DataFrame с date, p (нашата вероятност за избрания изход), odds, won (bool), по ред на датата.
    Връща крайната банка, най-голямото спадане от върха и броя залози.
    """
    peak = bank = float(bankroll)
    worst = 0.0
    n = 0
    for row in bets.sort_values("date", kind="stable").itertuples():
        f = stake_fraction(row.p, row.odds, kelly_part, cap)
        if f <= 0 or bank <= 0:
            continue
        stake = bank * f
        bank += stake * (row.odds - 1.0) if row.won else -stake
        peak = max(peak, bank)
        worst = max(worst, (peak - bank) / peak)
        n += 1
    return {"final": bank, "max_drawdown": worst, "bets": n}
