import os
import requests
import itertools
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# ────────────────────────────────────────────────
#  Provider configuration (set these in Render env vars)
# ────────────────────────────────────────────────

PROVIDERS = []

# 1. Groq (fast inference – good free tier)
if groq_key := os.getenv("GROQ_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "Groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": groq_key,
        "default_model": os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"),
        "headers_extra": {}
    })

# 2. OpenRouter (most generous free models + good tool calling)
if or_key := os.getenv("OPENROUTER_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "OpenRouter",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": or_key,
        "default_model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "headers_extra": {
            "HTTP-Referer": "https://your-domain.com",      # optional – replace with your app url
            "X-Title": "Clinton Tech AI"                    # optional
        }
    })

# 3. Together AI (some free / turbo-free models)
if together_key := os.getenv("TOGETHER_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "Together",
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "api_key": together_key,
        "default_model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        "headers_extra": {}
    })

if not PROVIDERS:
    print("WARNING: No LLM API keys configured → fallback mode only")

# Round-robin iterator
provider_cycle = itertools.cycle(PROVIDERS) if PROVIDERS else None

# Simple failure tracking per provider
provider_failures = {p["name"]: 0 for p in PROVIDERS}

# ────────────────────────────────────────────────
#  Tool definition (image generation)
# ────────────────────────────────────────────────

tools = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image based on a text prompt",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "Detailed description of the image"}
            },
            "required": ["prompt"]
        }
    }
}]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are Clinton Tech AI, an AI assistant created by Clinton Tech. "
        "Use the 'generate_image' tool when the user asks to create/generate/draw/make an image or picture. "
        "After generating an image, include the image URL in your final response. "
        "You can write HTML, CSS, JavaScript code – the frontend will show a live preview. "
        "Never mention which AI model or provider you are using. "
        "If asked who created you, say: 'I was created by Clinton Tech'."
    )
}

def generate_image(prompt: str) -> str | None:
    """Free image generation via Pollinations.ai"""
    try:
        encoded = requests.utils.quote(prompt.strip())
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true"
        # You can add &seed=1234 for reproducibility if desired
        return url
    except Exception as e:
        print(f"Image generation failed: {e}")
        return None

def fallback_response() -> dict:
    return {
        "reply": (
            "🌟 **Clinton Tech AI – Limited Mode**\n\n"
            "I'm currently running with very limited capabilities because no AI provider keys are configured.\n\n"
            "You can still:\n"
            "• Ask me to **generate images** (e.g. \"draw a cyberpunk city at night\")\n"
            "• Request **HTML, CSS, JavaScript code**\n"
            "• Have very basic conversation\n\n"
            "Full smart chat will be available once API keys are added.\n"
            "Try asking for an image now!"
        )
    }

# ────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


@app.route('/chat', methods=['POST'])
def chat():
    if not provider_cycle:
        return jsonify(fallback_response())

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"reply": "Please send a message."}), 400

    messages = [SYSTEM_MESSAGE] + history + [{"role": "user", "content": user_message}]

    payload_base = {
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False
    }

    attempted = set()

    for _ in range(len(PROVIDERS) * 2):
        provider = next(provider_cycle)
        name = provider["name"]

        if name in attempted or provider_failures[name] > 4:
            continue
        attempted.add(name)

        model = os.getenv(f"{name.upper()}_MODEL", provider["default_model"])
        payload = {**payload_base, "model": model}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['api_key']}",
            **provider["headers_extra"]
        }

        print(f"[CHAT] Trying {name} – model: {model}")

        try:
            r = requests.post(provider["api_url"], json=payload, headers=headers, timeout=50)
            r.raise_for_status()
            provider_failures[name] = 0

            result = r.json()
            choice = result["choices"][0]
            message = choice["message"]

            # Tool call handling
            if choice.get("finish_reason") == "tool_calls" and (tool_calls := message.get("tool_calls")):
                tool_call = tool_calls[0]
                if tool_call["function"]["name"] == "generate_image":
                    try:
                        args = json.loads(tool_call["function"]["arguments"])
                        prompt = args.get("prompt", "").strip()
                        if prompt:
                            image_url = generate_image(prompt)
                            if image_url:
                                # Append assistant message + tool result
                                messages.append(message)
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": tool_call["id"],
                                    "name": "generate_image",
                                    "content": json.dumps({"image_url": image_url})
                                })

                                # Second call – get final answer
                                second_payload = {
                                    "model": model,
                                    "messages": messages,
                                    "temperature": 0.7,
                                    "max_tokens": 2048,
                                    "stream": False
                                }
                                r2 = requests.post(provider["api_url"], json=second_payload, headers=headers, timeout=50)
                                r2.raise_for_status()
                                final = r2.json()["choices"][0]["message"]["content"]
                                return jsonify({"reply": final, "image": image_url})
                            else:
                                return jsonify({"reply": "Sorry, image generation failed right now."})
                    except Exception as e:
                        print(f"Tool call error: {e}")
                        return jsonify({"reply": "I tried to generate an image but something went wrong."})

            # Normal text response
            return jsonify({"reply": message["content"]})

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code
            print(f"[ERROR] {name} → HTTP {status}")
            if status in (401, 403):
                provider_failures[name] += 5   # likely bad key
            elif status == 429:
                provider_failures[name] += 1   # rate limit
            elif status == 400:
                return jsonify({"error": "Bad request to provider"}), 400
            else:
                provider_failures[name] += 1

        except requests.Timeout:
            print(f"[TIMEOUT] {name}")
            provider_failures[name] += 1
        except Exception as e:
            print(f"[EXCEPTION] {name}: {type(e).__name__} {e}")
            provider_failures[name] += 1

    print("[FALLBACK] All providers failed")
    return jsonify(fallback_response())


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "providers": [p["name"] for p in PROVIDERS],
        "mode": "fallback" if not PROVIDERS else "live"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=bool(os.getenv("FLASK_DEBUG")))
