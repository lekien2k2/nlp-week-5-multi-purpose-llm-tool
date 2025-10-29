"""OpenAI GPT model implementation"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def call_openai(prompt, model="gpt-3.5-turbo"):
    """Call OpenAI API (for GPT models)"""
    try:
        if not OPENAI_API_KEY:
            return "Error: OPENAI_API_KEY not configured. Please set it in .env file."

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"
