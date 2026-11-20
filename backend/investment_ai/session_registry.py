"""Bounded, run-scoped bearer sessions; no active-session eviction."""

import hashlib
import json
import math
import os
import secrets
import time


class SessionCapacityError(ValueError):
    pass


def now():
    return time.time()


class SessionRegistry:
    def _session_limits(self):
        capacity = int(os.environ.get("INVESTMENT_AI_MAX_SESSIONS", "256"))
        ttl = float(os.environ.get("INVESTMENT_AI_SESSION_TTL_SECONDS", "86400"))
        if capacity < 1 or not math.isfinite(ttl) or ttl <= 0:
            raise ValueError("Session capacity and TTL must be positive")
        return capacity, ttl

    def _restore_sessions(self, saved):
        self.session_tokens = set(saved.get("session_tokens", []))
        self.session_expires_at = dict(saved.get("session_expires_at", {}))
        return self._ensure_sessions()

    def _ensure_sessions(self):
        _, ttl = self._session_limits()
        self.session_tokens = getattr(self, "session_tokens", set())
        self.session_expires_at = getattr(self, "session_expires_at", {})
        migrated = False
        deadline = now() + ttl
        for digest in self.session_tokens:
            if digest not in self.session_expires_at:
                self.session_expires_at[digest] = deadline
                migrated = True
            expiry = self.session_expires_at[digest]
            if not isinstance(expiry, (int, float)) or not math.isfinite(expiry):
                raise ValueError("Invalid stored session expiry")
        return migrated

    def _discard_session(self, digest):
        self.session_tokens.discard(digest)
        self.session_expires_at.pop(digest, None)
        for name in (
            "latches",
            "episode_clocks",
            "execution_references",
            "execution_confirmations",
            "execution_history",
            "recurrent_sessions",
        ):
            records = getattr(self, name, {})
            for key in list(records):
                try:
                    scope = json.loads(key)
                except (ValueError, TypeError):
                    continue
                if isinstance(scope, list) and scope and scope[0] == digest:
                    del records[key]

    def _expire_sessions(self):
        self._ensure_sessions()
        expired = [
            key for key in self.session_tokens if self.session_expires_at[key] <= now()
        ]
        for digest in expired:
            self._discard_session(digest)
        if expired:
            self._save_episode_state()
        return bool(expired)

    def _allocate_session(self, persist):
        self._expire_sessions()
        capacity, ttl = self._session_limits()
        if len(self.session_tokens) >= capacity:
            raise SessionCapacityError(
                "Inference session capacity reached; retry later"
            )
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        self.session_tokens.add(digest)
        self.session_expires_at[digest] = now() + ttl
        if persist:
            try:
                self._save_episode_state()
            except Exception:
                self._discard_session(digest)
                raise
        return token

    def create_session(self):
        with self.lock:
            return self._allocate_session(persist=True)

    def optimize_new_session(
        self, market=None, news=None, state=None, *, execution_confirmation=None
    ):
        # Anonymous allocation and inference are one transaction: rejected requests
        # never publish their temporary token or rewrite the registry for that token.
        with self.lock:
            token = self._allocate_session(persist=False)
            try:
                result = self.optimize(
                    market,
                    news,
                    state,
                    session_token=token,
                    execution_confirmation=execution_confirmation,
                )
            except Exception:
                self._discard_session(hashlib.sha256(token.encode()).hexdigest())
                raise
            return {**result, "session_token": token}

    def _session_scope(self, token):
        if token is None:
            return None  # Trusted, single-owner in-process compatibility only.
        if not isinstance(token, str):
            raise ValueError("Invalid inference session token")
        self._ensure_sessions()
        digest = hashlib.sha256(token.encode()).hexdigest()
        if (
            digest not in self.session_tokens
            or self.session_expires_at[digest] <= now()
        ):
            raise ValueError("Unknown inference session; create a session first")
        return digest
