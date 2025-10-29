"""History management utilities"""

import os
import json
from datetime import datetime

HISTORY_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "history.json")


def save_history(user_text, task, prompt, model_provider, result, custom_prompt=None):
    """Save request/response to history"""
    try:
        history = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)

        entry = {
            "id": len(history) + 1,
            "timestamp": datetime.now().isoformat(),
            "user_text": user_text,
            "task": task,
            "prompt": prompt,
            "custom_prompt": custom_prompt if custom_prompt else False,
            "model": model_provider,
            "result": result,
        }

        history.append(entry)

        # Keep only last 100 entries
        history = history[-100:]

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return entry
    except Exception as e:
        print(f"Error saving history: {str(e)}")
        return None


def load_history(limit=50):
    """Load history from file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
                return history[-limit:] if limit else history
        return []
    except Exception as e:
        print(f"Error loading history: {str(e)}")
        return []
