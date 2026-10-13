"""One way for TIIM's agents to ask Claude a question and get JSON back.

Backends, in order of preference:
  * ApiLLM        - ANTHROPIC_API_KEY (pay per use), structured JSON output
  * ClaudeCodeLLM - your Claude Pro/Max subscription through Claude Code (`claude -p`), using
                    CLAUDE_CODE_OAUTH_TOKEN from `claude setup-token`
  * None          - no Claude: agents that need it simply step aside and TIIM decides alone

Every call goes through a Budget (Claude Pro has usage limits) and, for Claude Code, a process lock
so a small server never runs two `claude` processes at once.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

# Only one `claude` process at a time on the whole machine: the news reader, the desk (threads in the
# agent) and the nightly Strategy Lab (its own process) all take this lock.
_THREAD_LOCK = threading.Lock()


class _CliLock:
    def __enter__(self):
        _THREAD_LOCK.acquire()
        self._fh = None
        try:
            import fcntl

            from .config import DATA_DIR

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            self._fh = open(DATA_DIR / ".claude-cli.lock", "w")
            fcntl.flock(self._fh, fcntl.LOCK_EX)
        except (ImportError, OSError):  # pragma: no cover - non-POSIX
            pass
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            self._fh.close()   # releases the flock
        _THREAD_LOCK.release()
        return False


CLI_LOCK = _CliLock()

API_MODELS = {"opus": "claude-opus-5-5", "sonnet": "claude-sonnet-5-5", "haiku": "claude-haiku-5-5"}


def extract_json(text: str) -> dict | None:
    try:
        return json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, TypeError):
        return None


class Budget:
    """Daily call allowance, kept in the journal so restarts don't reset it."""

    def __init__(self, journal, limit: int, key: str = "desk") -> None:
        self.j, self.limit, self.key = journal, limit, key
        self._lock = threading.Lock()

    def _state(self) -> dict:
        today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
        st = self.j.get(f"llm_budget:{self.key}") or {}
        return st if st.get("date") == today else {"date": today, "used": 0}

    @property
    def used(self) -> int:
        return int(self._state()["used"])

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def take(self) -> bool:
        with self._lock:
            st = self._state()
            if st["used"] >= self.limit:
                return False
            st["used"] += 1
            self.j.set(f"llm_budget:{self.key}", st)
            return True


@dataclass
class Reply:
    data: dict | None
    error: str = ""


class ApiLLM:
    name = "claude-api"

    def __init__(self) -> None:
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    def ask(self, system: str, prompt: str, schema: dict, model: str, timeout: int = 180) -> Reply:
        a = self.anthropic
        try:
            r = self.client.beta.messages.create(
                model=API_MODELS.get(model, model), max_tokens=8000, system=system,
                messages=[{"role": "user", "content": prompt}],
                output_config={"effort": "medium", "format": {"type": "json_schema", "schema": schema}},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", timeout=timeout,
            )
        except a.RateLimitError:
            return Reply(None, "rate limited")
        except a.APIStatusError as e:
            return Reply(None, f"API error {e.status_code}")
        except a.APIConnectionError:
            return Reply(None, "Claude unreachable")
        if r.stop_reason in ("refusal", "max_tokens"):
            return Reply(None, f"stopped: {r.stop_reason}")
        text = next((b.text for b in r.content if b.type == "text"), "")
        data = extract_json(text)
        return Reply(data, "" if data is not None else "reply was not JSON")


class ClaudeCodeLLM:
    name = "claude-subscription"

    def __init__(self, binary: str) -> None:
        self.binary = binary

    def ask(self, system: str, prompt: str, schema: dict, model: str, timeout: int = 240) -> Reply:
        full = (system + "\n\n" + prompt + "\n\nDo not use any tools. Reply with ONLY one JSON object, no other text, "
                "matching this JSON schema:\n" + json.dumps(schema))
        cmd = [self.binary, "-p", full, "--output-format", "json", "--max-turns", "1"]
        if model:
            cmd += ["--model", model]
        try:
            with CLI_LOCK, tempfile.TemporaryDirectory() as empty:
                env = dict(os.environ, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", DISABLE_AUTOUPDATER="1")
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=empty, env=env)
            envelope = json.loads(proc.stdout or "{}")
            if proc.returncode != 0 or envelope.get("is_error"):
                return Reply(None, str(envelope.get("result") or proc.stderr or "claude failed")[:200])
            data = extract_json(str(envelope.get("result", "")))
            return Reply(data, "" if data is not None else "reply was not JSON")
        except subprocess.TimeoutExpired:
            return Reply(None, "Claude took too long")
        except Exception as e:  # pragma: no cover - environment specific
            return Reply(None, str(e)[:200])


def make_llm():
    """Claude API key first, then the Claude subscription via Claude Code, else None."""
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            return ApiLLM()
        except ImportError:
            log.warning("anthropic package not installed")
    if os.getenv("CLAUDE_CODE_OAUTH_TOKEN"):
        from .news.classify import find_claude_cli

        cli = find_claude_cli()
        if cli:
            return ClaudeCodeLLM(cli)
        log.warning("CLAUDE_CODE_OAUTH_TOKEN is set but the `claude` command isn't installed - desk agents are off")
    return None
