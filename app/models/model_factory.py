"""Model factory for routing prompts to appropriate LLM providers"""

from .openai_model import call_openai
from .deepseek_model import call_deepseek
from .gemini_model import call_gemini


class ModelProvider:
    """Enum-like class for model providers"""

    GPT = "gpt"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"


def call_model(prompt, model_provider):
    """Route prompt to the appropriate model API"""
    if model_provider == ModelProvider.GPT or model_provider == "gpt":
        return call_openai(prompt)
    elif model_provider == ModelProvider.DEEPSEEK or model_provider == "deepseek":
        return call_deepseek(prompt)
    elif model_provider == ModelProvider.GEMINI or model_provider == "gemini":
        return call_gemini(prompt)
    else:
        return f"Error: Unknown model provider '{model_provider}'"
