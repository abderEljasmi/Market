"""Optional Telegram notifications, so results reach your phone when you are not at the PC.

Setup (2 minutes):
  1. In Telegram, talk to @BotFather -> /newbot -> copy the token.
  2. Send any message to your new bot, then open
     https://api.telegram.org/bot<TOKEN>/getUpdates and copy "chat":{"id": ...}.
  3. Set environment variables TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
Without them every call is a silent no-op.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger(__name__)


def send(text: str) -> bool:
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return False
    try:
        # Telegram caps messages at 4096 chars
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                          json={"chat_id": chat, "text": text[:4000]}, timeout=15)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("telegram notification failed: %s", e)
        return False
