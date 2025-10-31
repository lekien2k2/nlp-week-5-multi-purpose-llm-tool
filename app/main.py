"""Main Flask application"""

import os
import sys
from flask import Flask, render_template, request, jsonify, send_file
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import models
from models import call_model
from models.openai_model import OPENAI_API_KEY
from models.deepseek_model import DEEPSEEK_API_KEY
from models.gemini_model import GEMINI_API_KEY

# Import utilities
from utils import (
    TASK_PROMPTS,
    is_vietnamese,
    add_language_instruction,
    save_history,
    load_history,
)

app = Flask(__name__)
app.template_folder = "templates"


def compare_models(prompt, model_providers):
    """Run the same prompt on multiple models concurrently"""
    results = {}

    def run_model(model_provider):
        try:
            result = call_model(prompt, model_provider)
            return model_provider, result, None
        except Exception as e:
            return model_provider, None, str(e)

    # Run all models concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(run_model, provider): provider
            for provider in model_providers
        }

        for future in as_completed(futures):
            model_provider, result, error = future.result()
            if error:
                results[model_provider] = {"result": None, "error": error}
            else:
                results[model_provider] = {"result": result, "error": None}

    return results


@app.route("/")
def index():
    """Render the main page"""
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """Process the user's request"""
    data = request.json
    user_text = data.get("text", "").strip()
    task = data.get("task", "")
    model_provider = data.get("model", "gpt").lower()
    self_critique = data.get("self_critique", False)
    save_to_history = data.get("save_history", True)

    if not user_text:
        return jsonify({"error": "Please enter some text"}), 400

    # Validate task (allow "chat" as a special task)
    valid_tasks = list(TASK_PROMPTS.keys()) + ["chat"]
    if task not in valid_tasks:
        return jsonify({"error": "Invalid task"}), 400

    # Validate model provider
    valid_providers = ["gpt", "deepseek", "gemini"]
    if model_provider not in valid_providers:
        return (
            jsonify(
                {
                    "error": f'Invalid model provider. Must be one of: {", ".join(valid_providers)}'
                }
            ),
            400,
        )

    # Create the full prompt
    if task == "chat":
        # For chat, use user_text directly without any prompt template
        prompt = user_text
    else:
        prompt = TASK_PROMPTS[task] + user_text
        # Add Vietnamese instruction if needed
        prompt = add_language_instruction(prompt, user_text, task)

    if self_critique:
        # Get initial answer first
        initial_result = call_model(prompt, model_provider)

        # Create critique prompt
        critique_prompt = f"""Câu trả lời ban đầu của bạn:
{initial_result}

Hãy đánh giá câu trả lời trên và xác định các điểm cần cải thiện về:
- Tính chính xác
- Tính đầy đủ
- Tính rõ ràng
- Khả năng áp dụng

Sau đó đưa ra phiên bản cải thiện của câu trả lời."""

        # Get improved answer
        improved_result = call_model(critique_prompt, model_provider)

        # Create critique analysis (asking model to analyze what needs improvement)
        analysis_prompt = f"""Câu trả lời ban đầu:
{initial_result}

Hãy phân tích và chỉ ra những điểm cần cải thiện trong câu trả lời trên. Đưa ra đánh giá ngắn gọn về:
1. Điểm mạnh
2. Điểm yếu cần cải thiện
3. Gợi ý cải thiện"""

        critique_analysis = call_model(analysis_prompt, model_provider)

        # Combine full prompt for history
        full_prompt = prompt

        # Save to history with improved result
        if save_to_history:
            save_history(
                user_text,
                task,
                full_prompt,
                model_provider,
                improved_result,
            )

        return jsonify(
            {
                "result": improved_result,
                "prompt": full_prompt,
                "self_critique": {
                    "initial_answer": initial_result,
                    "critique_analysis": critique_analysis,
                    "improved_answer": improved_result,
                },
            }
        )
    else:
        # Call the appropriate model
        result = call_model(prompt, model_provider)

        # Save to history
        if save_to_history:
            save_history(
                user_text,
                task,
                prompt,
                model_provider,
                result,
            )

        return jsonify({"result": result, "prompt": prompt})


@app.route("/compare", methods=["POST"])
def compare():
    """Compare results from multiple models"""
    data = request.json
    user_text = data.get("text", "").strip()
    task = data.get("task", "")
    model_providers = data.get("models", [])
    self_critique = data.get("self_critique", False)
    save_to_history = data.get("save_history", True)

    if not user_text:
        return jsonify({"error": "Please enter some text"}), 400

    if not model_providers:
        return jsonify({"error": "Please select at least one model"}), 400

    # Validate task (allow "chat" as a special task)
    valid_tasks = list(TASK_PROMPTS.keys()) + ["chat"]
    if task not in valid_tasks:
        return jsonify({"error": "Invalid task"}), 400

    # Validate model providers
    valid_providers = ["gpt", "deepseek", "gemini"]
    for provider in model_providers:
        if provider not in valid_providers:
            return jsonify({"error": f"Invalid model provider: {provider}"}), 400

    # Create the full prompt
    if task == "chat":
        # For chat, use user_text directly without any prompt template
        prompt = user_text
    else:
        prompt = TASK_PROMPTS[task] + user_text
        # Add Vietnamese instruction if needed
        prompt = add_language_instruction(prompt, user_text, task)

    if self_critique:
        # For compare mode with self-critique, we need to handle each model separately
        results = {}

        for model_provider in model_providers:
            try:
                # Get initial answer
                initial_result = call_model(prompt, model_provider)

                # Create critique prompt
                critique_prompt = f"""Câu trả lời ban đầu của bạn:
{initial_result}

Hãy đánh giá câu trả lời trên và xác định các điểm cần cải thiện về:
- Tính chính xác
- Tính đầy đủ
- Tính rõ ràng
- Khả năng áp dụng

Sau đó đưa ra phiên bản cải thiện của câu trả lời."""

                # Get improved answer
                improved_result = call_model(critique_prompt, model_provider)

                # Create critique analysis
                analysis_prompt = f"""Câu trả lời ban đầu:
{initial_result}

Hãy phân tích và chỉ ra những điểm cần cải thiện trong câu trả lời trên. Đưa ra đánh giá ngắn gọn về:
1. Điểm mạnh
2. Điểm yếu cần cải thiện
3. Gợi ý cải thiện"""

                critique_analysis = call_model(analysis_prompt, model_provider)

                results[model_provider] = {
                    "result": improved_result,
                    "error": None,
                    "self_critique": {
                        "initial_answer": initial_result,
                        "critique_analysis": critique_analysis,
                        "improved_answer": improved_result,
                    },
                }

                # Save to history
                if save_to_history:
                    save_history(
                        user_text,
                        task,
                        prompt,
                        model_provider,
                        improved_result,
                    )
            except Exception as e:
                results[model_provider] = {
                    "result": None,
                    "error": str(e),
                }

        return jsonify({"results": results, "prompt": prompt})
    else:
        # Compare models without self-critique
        results = compare_models(prompt, model_providers)

        # Save to history for each model
        if save_to_history:
            for model_provider, model_result in results.items():
                if model_result["result"]:
                    save_history(
                        user_text,
                        task,
                        prompt,
                        model_provider,
                        model_result["result"],
                    )

        return jsonify({"results": results, "prompt": prompt})


@app.route("/history", methods=["GET"])
def get_history():
    """Get history of requests"""
    limit = request.args.get("limit", 50, type=int)
    history = load_history(limit)
    return jsonify({"history": history})


@app.route("/history/<int:history_id>", methods=["DELETE"])
def delete_history_item(history_id):
    """Delete a specific history item"""
    try:
        import json
        from utils.history_utils import HISTORY_FILE, load_history

        history = load_history(limit=None)
        history = [h for h in history if h.get("id") != history_id]

        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export", methods=["POST"])
def export_results():
    """Export results to file"""
    import tempfile
    import json
    import threading

    data = request.json
    export_format = data.get("format", "json")  # json or txt
    results = data.get("results", {})
    prompt = data.get("prompt", "")

    if not results:
        return jsonify({"error": "No results to export"}), 400

    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if export_format == "json":
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "results": results,
            }
            filename = f"export_{timestamp}.json"

            # Use tempfile to avoid file accumulation
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".json"
            ) as tmp_file:
                json.dump(export_data, tmp_file, ensure_ascii=False, indent=2)
                tmp_filename = tmp_file.name

            # Delete file after sending (delay 5 seconds)
            def cleanup():
                import time

                time.sleep(5)
                try:
                    if os.path.exists(tmp_filename):
                        os.unlink(tmp_filename)
                except:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

            return send_file(
                tmp_filename,
                as_attachment=True,
                download_name=filename,
                mimetype="application/json",
            )

        else:  # txt
            filename = f"export_{timestamp}.txt"

            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".txt"
            ) as tmp_file:
                f = tmp_file
                f.write(
                    f"Export Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"{'='*60}\n\n")
                f.write(f"Prompt:\n{prompt}\n\n")
                f.write(f"{'='*60}\n\n")
                f.write("Results:\n\n")

                for model, result_data in results.items():
                    f.write(f"Model: {model.upper()}\n")
                    f.write(f"{'-'*60}\n")

                    if isinstance(result_data, dict):
                        if result_data.get("error"):
                            f.write(f"Error: {result_data['error']}\n\n")
                        elif result_data.get("self_critique"):
                            # Handle self-critique data
                            critique = result_data.get("self_critique", {})
                            f.write("📝 Câu trả lời ban đầu:\n")
                            f.write(f"{critique.get('initial_answer', 'N/A')}\n\n")
                            f.write("🔍 Đánh giá và phân tích:\n")
                            f.write(f"{critique.get('critique_analysis', 'N/A')}\n\n")
                            f.write("✨ Kết quả cải thiện:\n")
                            f.write(f"{critique.get('improved_answer', 'N/A')}\n\n")
                        else:
                            f.write(f"{result_data.get('result', 'N/A')}\n\n")
                    else:
                        f.write(f"{result_data}\n\n")

                tmp_filename = tmp_file.name

            # Delete file after sending (delay 5 seconds)
            def cleanup():
                import time

                time.sleep(5)
                try:
                    if os.path.exists(tmp_filename):
                        os.unlink(tmp_filename)
                except:
                    pass

            threading.Thread(target=cleanup, daemon=True).start()

            return send_file(
                tmp_filename,
                as_attachment=True,
                download_name=filename,
                mimetype="text/plain",
            )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    """Health check endpoint"""
    return jsonify(
        {
            "status": "ok",
            "providers": {
                "gpt": {
                    "available": bool(OPENAI_API_KEY),
                    "configured": bool(OPENAI_API_KEY),
                },
                "deepseek": {
                    "available": bool(DEEPSEEK_API_KEY),
                    "configured": bool(DEEPSEEK_API_KEY),
                },
                "gemini": {
                    "available": bool(GEMINI_API_KEY),
                    "configured": bool(GEMINI_API_KEY),
                },
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
