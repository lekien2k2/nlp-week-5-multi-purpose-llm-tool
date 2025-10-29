"""Deepseek model implementation"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")


def call_deepseek(prompt, model="deepseek-chat"):
    """Call Deepseek API"""
    try:
        if not DEEPSEEK_API_KEY:
            return "Error: DEEPSEEK_API_KEY not configured. Please set it in .env file."

        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
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
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        error_detail = ""
        try:
            error_data = response.json()
            error_detail = error_data.get("error", {}).get("message", str(e))
            # Check for specific error types
            if (
                "Insufficient Balance" in error_detail
                or "balance" in error_detail.lower()
            ):
                return "Error: Tài khoản Deepseek không đủ số dư. Vui lòng nạp thêm credit hoặc sử dụng model khác (GPT/Gemini)."
        except:
            error_detail = str(e)

        if response.status_code == 401:
            return "Error: Deepseek API key không hợp lệ. Vui lòng kiểm tra lại trong file .env"
        elif response.status_code == 429:
            return "Error: Vượt quá rate limit của Deepseek. Vui lòng thử lại sau."
        else:
            return f"Error calling Deepseek API: {error_detail}"
    except Exception as e:
        return f"Error calling Deepseek API: {str(e)}"
