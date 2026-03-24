from typing import Any

from gemini_webapi import GeminiClient

from usage_tracker import UsageTracker


class AppState:
    client: GeminiClient | None = None
    uploaded_files: dict[str, dict[str, Any]] = {}
    usage_tracker: UsageTracker = UsageTracker()
    policy_gem_ids: dict[str, str] = {}


state = AppState()
