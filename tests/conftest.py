"""Tests never read the real .env or the real environment: on the server the update runs the tests next to
your .env, and a failing assertion would print your keys. Everything the agent reads from the environment is
cleared, and python-dotenv is switched off, before any of TIIM's modules are imported."""
import os

import dotenv

dotenv.load_dotenv = lambda *a, **k: False          # config.py calls this at import time

_PREFIXES = ("TL_", "NEWS_", "DESK_", "LAB_", "BOT_", "AGENT_", "MAX_", "RISK_", "HTF_", "PLAN_", "EXTENSION_",
             "LIVE_", "ALLOW_", "CLAUDE_", "ANTHROPIC_", "DASHBOARD_", "FINNHUB_")
for _k in list(os.environ):
    if _k.startswith(_PREFIXES):
        del os.environ[_k]
