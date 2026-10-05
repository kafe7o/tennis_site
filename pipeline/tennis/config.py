"""
Тайните - само от .env или средата, никога в кода.

Употреба:
    from tennis import config
    key = config.require("ODDS_API_KEY")
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
NTFY_TOPIC = os.environ.get("NTFY_TOPIC")

# Пътищата може да се пренасочат със средата - в облака (GitHub Actions) са на друго място.
DATA_DIR = Path(os.environ.get("TENNIS_DATA", ROOT.parent / "data"))
CACHE_DIR = Path(os.environ.get("TENNIS_CACHE", ROOT / ".cache"))
LOG_DIR = Path(os.environ.get("TENNIS_LOGS", ROOT / "logs"))

HINTS = {
    "ODDS_API_KEY": "ключ от the-odds-api.com",
    "NTFY_TOPIC": "темата в ntfy.sh за известията",
}


def require(*names):
    """Връща стойностите или обяснява точно какво липсва и откъде се взима."""
    missing = [n for n in names if not globals().get(n)]
    if missing:
        lines = "\n".join(f"  {n}= ... ({HINTS.get(n, 'виж README')})" for n in missing)
        raise RuntimeError(f"Липсва в .env:\n{lines}")
    values = [globals()[n] for n in names]
    return values[0] if len(values) == 1 else tuple(values)
