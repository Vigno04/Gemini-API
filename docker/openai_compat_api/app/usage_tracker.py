import json
import time
from pathlib import Path
from typing import Any


class UsageTracker:
    def __init__(self, storage_file: str = "usage_data.json"):
        self.storage_file = Path(storage_file)
        self.usage_data = self._load_usage_data()

    def _load_usage_data(self) -> dict[str, Any]:
        if self.storage_file.exists():
            try:
                with open(self.storage_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                pass
        return {"daily_usage": {}, "total_usage": {}}

    def _save_usage_data(self):
        with open(self.storage_file, 'w') as f:
            json.dump(self.usage_data, f, indent=2)

    def track_request(self, model: str, prompt_tokens: int, completion_tokens: int, cost: float = 0.0):
        today = time.strftime("%Y-%m-%d")

        if today not in self.usage_data["daily_usage"]:
            self.usage_data["daily_usage"][today] = {}

        if model not in self.usage_data["daily_usage"][today]:
            self.usage_data["daily_usage"][today][model] = {
                "total_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "requests": 0,
                "cost": 0.0
            }

        daily_model = self.usage_data["daily_usage"][today][model]
        daily_model["total_tokens"] += prompt_tokens + completion_tokens
        daily_model["completion_tokens"] += completion_tokens
        daily_model["prompt_tokens"] += prompt_tokens
        daily_model["requests"] += 1
        daily_model["cost"] += cost

        # Update totals
        if model not in self.usage_data["total_usage"]:
            self.usage_data["total_usage"][model] = {
                "total_tokens": 0,
                "completion_tokens": 0,
                "prompt_tokens": 0,
                "requests": 0,
                "cost": 0.0
            }

        total_model = self.usage_data["total_usage"][model]
        total_model["total_tokens"] += prompt_tokens + completion_tokens
        total_model["completion_tokens"] += completion_tokens
        total_model["prompt_tokens"] += prompt_tokens
        total_model["requests"] += 1
        total_model["cost"] += cost

        self._save_usage_data()

    def get_usage(self, start_date: str | None = None, end_date: str | None = None) -> dict[str, Any]:
        data = []
        total_usage = 0
        total_cost = 0.0

        for date, models in self.usage_data["daily_usage"].items():
            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            for model, stats in models.items():
                data.append({
                    "object": "usage",
                    "date": date,
                    "model": model,
                    "total_tokens": stats["total_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "prompt_tokens": stats["prompt_tokens"],
                    "requests": stats["requests"],
                    "cost": {
                        "amount": stats["cost"],
                        "currency": "usd"
                    }
                })
                total_usage += stats["total_tokens"]
                total_cost += stats["cost"]

        return {
            "object": "list",
            "data": data,
            "total_usage": total_usage,
            "total_cost": {
                "amount": total_cost,
                "currency": "usd"
            }
        }