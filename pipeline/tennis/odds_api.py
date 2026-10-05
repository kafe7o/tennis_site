"""
the-odds-api.com за тенис - живи коефициенти и резултати.

Разликата от футбола: тенисът в API е по ТУРНИР (tennis_atp_us_open, tennis_wta_...), активен само
докато турнирът тече. Затова списъкът не е фиксиран в кода - tennis_sports() го взема от
заявката за спортове (безплатна). Мачът няма равен: два изхода.

Квотата е малка (500 заявки месечно на безплатния план), затова:
  - ЕДИН регион на заявка: цената е [пазари] x [региони] кредита;
  - кеш на диска, за да не се плаща два пъти за едно и също в рамките на един цикъл;
  - last_remaining() се чете от заглавките и се лога след всяка заявка.

Ключът не влиза в съобщенията за грешка: requests и urllib слагат целия URL в тях, а
логът се чете от хора и се копира в чатове.
"""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from . import config

log = logging.getLogger(__name__)

BASE = "https://api.the-odds-api.com/v4"
CACHE_MINUTES = 120
SCORES_CACHE_MINUTES = 60
_last_remaining = None


def redact(text):
    key = config.ODDS_API_KEY
    return text.replace(key, "<ODDS_API_KEY>") if key else text


def _cache(name):
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return config.CACHE_DIR / f"{name}.json"


def _get(path, params, cache_name=None, cache_minutes=0):
    global _last_remaining
    if cache_name and cache_minutes:
        cached = _cache(cache_name)
        if cached.exists():
            age = (time.time() - cached.stat().st_mtime) / 60
            if age < cache_minutes:
                log.info("%s: от кеша (%.0f мин.)", cache_name, age)
                return json.loads(cached.read_text(encoding="utf-8"))

    key = config.require("ODDS_API_KEY")
    url = f"{BASE}{path}?{urllib.parse.urlencode({'apiKey': key, **params})}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
            _last_remaining = resp.headers.get("x-requests-remaining")
    except urllib.error.HTTPError as e:
        raise RuntimeError(redact(f"{path}: API върна {e.code} - {e.read()[:200].decode('utf-8', 'replace')}")) from None
    except OSError as e:
        raise RuntimeError(redact(f"{path}: няма връзка ({e})")) from None
    if cache_name:
        _cache(cache_name).write_text(json.dumps(data), encoding="utf-8")
    return data


def last_remaining():
    return int(_last_remaining) if _last_remaining not in (None, "") else None


def tennis_sports():
    """Активните тенис турнири в момента. Заявката за списъка е безплатна."""
    data = _get("/sports/", {})
    return [s["key"] for s in data
            if s["key"].startswith("tennis_") and s.get("active") and not s.get("has_outrights")]


def odds(sport, regions="eu", cache_minutes=CACHE_MINUTES):
    """Коефициентите на турнира за предстоящите мачове (пазар h2h). Цена: 1 кредит на регион."""
    data = _get(f"/sports/{sport}/odds/", {"regions": regions, "markets": "h2h", "oddsFormat": "decimal"},
                cache_name=f"odds_{sport}_{regions}", cache_minutes=cache_minutes)
    log.info("%s (%s): %d мача, остават %s кредита", sport, regions, len(data), _last_remaining)
    return data


def prices(event):
    """
    Цените за двамата играчи от всички книги на събитието, или None, ако няма нито една.
    Връща: home, away, commence, books {книга: (цена_home, цена_away)}, avg, max, n_books.
    """
    home, away = event["home_team"], event["away_team"]
    books = {}
    for bookmaker in event.get("bookmakers", []):
        for market in bookmaker.get("markets", []):
            if market.get("key") != "h2h":
                continue
            by_name = {o["name"]: o["price"] for o in market.get("outcomes", [])}
            if home in by_name and away in by_name:
                books[bookmaker["key"]] = (float(by_name[home]), float(by_name[away]))
    if not books:
        return None
    arr = np.array(list(books.values()))
    return {"home": home, "away": away, "commence": event["commence_time"], "books": books,
            "avg": (float(arr[:, 0].mean()), float(arr[:, 1].mean())),
            "max": (float(arr[:, 0].max()), float(arr[:, 1].max())), "n_books": len(books)}


def finished(sport, days_from=3, cache_minutes=SCORES_CACHE_MINUTES):
    """Приключилите мачове с победител. 2 кредита, когато се иска история. В тениса "score" е броят сетове."""
    data = _get(f"/sports/{sport}/scores/", {"daysFrom": days_from},
                cache_name=f"scores_{sport}_{days_from}", cache_minutes=cache_minutes)
    out = []
    for event in data:
        if not event.get("completed") or not event.get("scores"):
            continue
        by_name = {s["name"]: s["score"] for s in event["scores"]}
        home, away = event.get("home_team"), event.get("away_team")
        try:
            sets_home, sets_away = float(by_name[home]), float(by_name[away])
        except (KeyError, TypeError, ValueError):
            log.warning("%s: резултат без имена или нечислов (%s)", sport, event.get("id"))
            continue
        if sets_home == sets_away:
            log.warning("%s: равен резултат в тенис (%s) - пропуснато", sport, event.get("id"))
            continue
        out.append({"id": event["id"], "home": home, "away": away, "commence": event.get("commence_time"),
                    "winner": home if sets_home > sets_away else away, "sets": (sets_home, sets_away)})
    log.info("%s: %d приключили мача", sport, len(out))
    return out
