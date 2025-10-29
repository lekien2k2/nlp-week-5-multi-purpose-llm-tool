"""Language detection and prompt processing utilities"""

import re

# Vietnamese characters pattern
VIETNAMESE_CHARS = re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]",
    re.IGNORECASE,
)


def is_vietnamese(text):
    """Detect if text contains Vietnamese characters"""
    if not text:
        return False
    # Check for Vietnamese diacritics
    if VIETNAMESE_CHARS.search(text):
        return True
    # Check for common Vietnamese words
    vietnamese_words = [
        "và",
        "của",
        "cho",
        "với",
        "là",
        "được",
        "trong",
        "một",
        "các",
        "này",
        "đó",
        "từ",
        "bạn",
        "tôi",
        "anh",
        "chị",
        "em",
        "chúng",
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in vietnamese_words)


def add_language_instruction(prompt, user_text, task):
    """Add language instruction to prompt if Vietnamese is detected"""
    if is_vietnamese(user_text) and task != "translate":
        if not prompt.startswith("Bạn hãy"):
            # Add Vietnamese instruction
            if "Summarize" in prompt or "summarize" in prompt.lower():
                return (
                    "Bạn hãy tóm tắt văn bản sau đây bằng tiếng Việt một cách ngắn gọn:\n\n"
                    + user_text
                )
            elif "Explain" in prompt or "explain" in prompt.lower():
                return (
                    "Bạn hãy giải thích điều sau đây bằng tiếng Việt một cách đơn giản như cho trẻ 5 tuổi:\n\n"
                    + user_text
                )
            elif "Extract" in prompt or "keywords" in prompt.lower():
                return (
                    "Bạn hãy trích xuất các từ khóa chính từ văn bản sau đây. Liệt kê chúng cách nhau bằng dấu phẩy, trả lời bằng tiếng Việt:\n\n"
                    + user_text
                )
            elif "Generate" in prompt or "code" in prompt.lower():
                return (
                    "Bạn hãy tạo mã Python cho yêu cầu sau đây. Trả lời bằng tiếng Việt và giải thích code:\n\n"
                    + user_text
                )
    return prompt
