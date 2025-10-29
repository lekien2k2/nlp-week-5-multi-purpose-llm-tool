"""Google Gemini model implementation"""

import os
from dotenv import load_dotenv

load_dotenv()

# Try to import google.generativeai
try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GEMINI_AVAILABLE = False
    genai = None
except Exception:
    GEMINI_AVAILABLE = False
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def call_gemini(prompt, model="gemini-flash-latest"):
    """Call Google Gemini API using SDK"""
    try:
        if not GEMINI_API_KEY:
            return "Error: GEMINI_API_KEY not configured. Please set it in .env file."

        # Try to import if not available at module load
        global genai, GEMINI_AVAILABLE
        if not GEMINI_AVAILABLE or genai is None:
            try:
                import google.generativeai as genai

                GEMINI_AVAILABLE = True
            except (ImportError, ModuleNotFoundError):
                return "Error: Thư viện google-generativeai chưa được cài đặt. Vui lòng chạy: pip install google-generativeai và khởi động lại Flask server (Ctrl+C rồi chạy lại python app.py)."

        # Configure Gemini
        genai.configure(api_key=GEMINI_API_KEY)

        # Create model instance
        gemini_model = genai.GenerativeModel(model)

        # Generate content
        response = gemini_model.generate_content(prompt)

        return response.text
    except (ImportError, ModuleNotFoundError) as e:
        return f"Error: Không thể import google.generativeai. Vui lòng chạy: pip install google-generativeai và khởi động lại server. Chi tiết: {str(e)}"
    except Exception as e:
        error_msg = str(e)
        if "quota" in error_msg.lower() or "429" in error_msg:
            return (
                "Error calling Gemini API: Rate limit exceeded. Please try again later."
            )
        elif "404" in error_msg or "not found" in error_msg.lower():
            return f"Error calling Gemini API: Model '{model}' not found. Try 'gemini-flash-latest' or 'gemini-pro-latest'"
        elif (
            "API key" in error_msg.lower()
            or "401" in error_msg
            or "api_key" in error_msg.lower()
        ):
            return "Error: Gemini API key không hợp lệ hoặc chưa được cấu hình. Vui lòng kiểm tra lại trong file .env"
        else:
            return f"Error calling Gemini API: {error_msg}"
