"""
Идеите на майстора, проверени върху ТЕНИС (протоколът е записан ПРЕДИ пускането).

Основата е football-site/pipeline/research/league_analysis.py - същата логика и същите прагове:

  Данни: tennis-data.co.uk 2012-2026, мъжете и жените ПОРЕДНО (не се смесват). Цени: средни (Avg)
  преди мача; вероятностите - със степенно махане на маржа. Фаворит = играчът с по-голям шанс по пазара.

  Лента на „тото“: фаворит с коефициент 1.30-1.55 (както във футбола). За всеки СЕГМЕНТ: колко печелят
  фаворитите в лентата срещу обещаното (diff = излиза - обещано; t = diff / стандартна грешка).
    ИЗБЯГВАЙ („тото“)  t <= -2 за всичко И diff < 0 и в 2012-2019, и в 2019-2026 (като във футбола)
    СТОЙНОСТ           t >= +2 за всичко И diff > 0 и в двата периода (обратното - като Висшата лига)

  Сегменти (фиксирани предварително, нищо не се добавя след като видим резултатите):
    ниво на турнира (Series), настилка, закрит/открит корт, кръг, мач до 3 или до 5 сета;
    „дерби“ (вечни съперници): h2h >= 3 и >= 5 предишни срещи;
    „след паузата“: фаворитът / аутсайдерът с 14-30 и 31-90 дни без мач (без междусезонието);
    „формата“: фаворит / аутсайдер, чийто последен мач е отказване; чийто последен мач е бил дълъг.
  Сегмент с по-малко от 300 мача в лентата не се оценява.

  Колко се очаква да излязат само от късмет: при ~40 сегмента и двойното условие - около 0.2 на тур.
  Затова потвърден сегмент е показател, но се гледа и ROI на цените, и смислено ли е обяснението.

Резултатът: results/master_tennis_<тур>.json и таблица на екрана.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tennis import context, dataset, loaders      # noqa: E402

BAND = (1.30, 1.55)
SPLIT = pd.Timestamp("2019-07-01")
MIN_N = 300
TOURS = {"man": "ATP", "woman": "WTA"}


def load(tour):
    folder = ROOT.parent / "data" / tour
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        frames = [loaders.tennis_data(p, TOURS[tour]) for p in sorted(folder.glob("*"))
                  if p.suffix.lower() in (".xls", ".xlsx", ".csv")]
    df = context.add(dataset.build(pd.concat(frames, ignore_index=True)))
    df = df[df["mkt_p1"].notna() & df["o1_Avg"].gt(1) & df["o2_Avg"].gt(1)].copy()
    fav1 = df["mkt_p1"] >= 0.5
    pick = lambda a, b: np.where(fav1, df[a], df[b])
    df["fav_p"] = np.where(fav1, df["mkt_p1"], 1 - df["mkt_p1"])
    df["fav_odds"] = pick("o1_Avg", "o2_Avg")
    df["fav_won"] = np.where(fav1, df["y"] == 1, df["y"] == 0)
    for name in ("rest", "ret", "long"):
        df[f"{name}_fav"], df[f"{name}_dog"] = pick(f"{name}1", f"{name}2"), pick(f"{name}2", f"{name}1")
    df["profit"] = np.where(df["fav_won"], df["fav_odds"] - 1.0, -1.0)
    return df


def calib(d):
    """Фаворитите в d: n, обещано, излиза, diff, t (срещу сумата на p(1-p)) и доход на средните цени."""
    n = len(d)
    if n < 2:
        return None
    p, won = d["fav_p"].to_numpy(), d["fav_won"].to_numpy()
    se = np.sqrt((p * (1 - p)).sum()) / n
    diff = won.mean() - p.mean()
    return {"n": int(n), "promised": float(p.mean()), "actual": float(won.mean()), "diff": float(diff),
            "t": float(diff / se) if se > 0 else float("nan"), "roi": float(d["profit"].mean())}


def segments(df):
    seg = {}
    for col, label in (("series", "ниво"), ("surface", "настилка"), ("court", "корт"), ("round", "кръг")):
        for value in sorted(df[col].dropna().astype(str).unique()):
            seg[f"{label}: {value}"] = df[col].astype(str) == value
    if df["best_of"].nunique() > 1:
        seg["мач до 5 сета"] = df["best_of"] == 5
        seg["мач до 3 сета"] = df["best_of"] == 3
    for k in (3, 5):
        seg[f"дерби: {k}+ срещи"] = df["h2h"] >= k
    off = ~df["offseason"]
    for lo, hi in ((14, 30), (31, 90)):
        seg[f"след пауза {lo}-{hi} дни: фаворитът"] = df["rest_fav"].between(lo, hi) & off
        seg[f"след пауза {lo}-{hi} дни: аутсайдерът"] = df["rest_dog"].between(lo, hi) & off
    seg["формата: фаворитът идва от отказване"] = df["ret_fav"]
    seg["формата: аутсайдерът идва от отказване"] = df["ret_dog"]
    seg["формата: фаворитът идва от дълъг мач"] = df["long_fav"]
    seg["формата: аутсайдерът идва от дълъг мач"] = df["long_dog"]
    return seg


def verdict(whole, early, late):
    if not (whole and early and late):
        return ""
    if whole["t"] <= -2 and early["diff"] < 0 and late["diff"] < 0:
        return "ИЗБЯГВАЙ"
    if whole["t"] >= 2 and early["diff"] > 0 and late["diff"] > 0:
        return "СТОЙНОСТ"
    return ""


def run(tour):
    df = load(tour)
    band = df[df["fav_odds"].between(*BAND)]
    rows = []
    base = calib(band)
    for name, mask in segments(df).items():
        d = band[mask.reindex(band.index, fill_value=False)]
        if len(d) < MIN_N:
            continue
        whole, early, late = calib(d), calib(d[d["date"] < SPLIT]), calib(d[d["date"] >= SPLIT])
        rows.append({"сегмент": name, **whole, "diff_2012_19": early["diff"] if early else None,
                     "diff_2019_26": late["diff"] if late else None, "вердикт": verdict(whole, early, late)})
    table = pd.DataFrame(rows).sort_values("t")
    return df, band, base, table


def main():
    out = {}
    for tour in TOURS:
        df, band, base, table = run(tour)
        print(f"\n################ {tour} ({TOURS[tour]}): {len(df)} мача с цени, {len(band)} фаворита в лентата {BAND} ################")
        print(f"всички фаворити в лентата: обещано {base['promised']:.3f}, излиза {base['actual']:.3f}, "
              f"diff {base['diff']:+.3f}, t {base['t']:+.2f}, доход {base['roi']:+.3f}")
        by_half = {"2012-2019": calib(band[band['date'] < SPLIT]), "2019-2026": calib(band[band['date'] >= SPLIT])}
        for k, v in by_half.items():
            print(f"   {k}: n {v['n']}, diff {v['diff']:+.3f}, t {v['t']:+.2f}, доход {v['roi']:+.3f}")
        pd.set_option("display.width", 200)
        show = table.assign(**{c: table[c] for c in ("promised", "actual", "diff", "diff_2012_19", "diff_2019_26", "roi", "t")})
        print(f"\nТестирани сегменти: {len(table)} (n >= {MIN_N} в лентата). Подредени по t (най-лошите за фаворита първи):")
        print(show[["сегмент", "n", "promised", "actual", "diff", "t", "diff_2012_19", "diff_2019_26", "roi", "вердикт"]]
              .round(3).to_string(index=False))
        flagged = table[table["вердикт"] != ""]
        print(f"\nПотвърдени (двойното условие): {len(flagged)} от {len(table)}")
        if len(flagged):
            print(flagged[["сегмент", "n", "diff", "t", "roi", "вердикт"]].round(3).to_string(index=False))
        out[tour] = {"band": list(BAND), "tournaments": len(df), "base": base, "halves": by_half,
                     "segments": json.loads(table.to_json(orient="records")), "confirmed": json.loads(flagged.to_json(orient="records"))}
    target = ROOT.parent / "results"
    target.mkdir(exist_ok=True)
    (target / "master_tennis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
