"""
Имената на играчите. Източниците не си приличат:

  tennis-data.co.uk   "Del Potro J.M."  "Djokovic N."   (фамилия, после инициали с точки)
  Sackmann / odds API "Juan Martin Del Potro"  "Novak Djokovic"   (име, после фамилия)

За да се съберат историята и живите коефициенти, и двете се свеждат до ключ (фамилия, инициали).
Пълното име е двусмислено къде свършват личните имена и къде започва фамилията, затова
candidates() връща всички възможности, а match() приема само ЕДИНСТВЕНО съвпадение: при две
или нула - None, за да не се залага на грешен играч.
"""

import re
import unicodedata


def _plain(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^a-z\- ]", "", text.lower().replace("'", "")).replace("-", " ").strip()


def td_key(name):
    """'Del Potro J.M.' -> ('del potro', 'jm'). Без точка накрая няма инициали -> ('<цялото>', '')."""
    parts = name.strip().split()
    initials, surname = [], []
    for token in parts:
        if re.fullmatch(r"(?:[A-Za-z]\.)+", token):
            initials.append(token.replace(".", "").lower())
        else:
            surname.append(token)
    return _plain(" ".join(surname)), "".join(initials)


def candidates(full_name):
    """'Juan Martin Del Potro' -> [('martin del potro','j'), ('del potro','jm'), ('potro','jmd')]"""
    tokens = [t for t in re.split(r"\s+", full_name.strip()) if t]
    out = []
    for i in range(1, len(tokens)):
        surname = _plain(" ".join(tokens[i:]))
        initials = "".join(_plain(t)[:1] for t in tokens[:i])
        if surname and initials:
            out.append((surname, initials))
    return out


def _compatible(td, cand):
    (s1, i1), (s2, i2) = td, cand
    return s1 == s2 and (i2.startswith(i1) or i1.startswith(i2)) and bool(i1 and i2)


def match(full_name, td_keys):
    """Единственият ключ от tennis-data, който пасва на пълното име, иначе None."""
    cands = candidates(full_name)
    hits = {k for k in td_keys if any(_compatible(k, c) for c in cands)}
    return next(iter(hits)) if len(hits) == 1 else None
