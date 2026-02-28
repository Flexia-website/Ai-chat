import os
import requests
import itertools
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='.')
CORS(app)

# ────────────────────────────────────────────────
#  Provider configuration
# ────────────────────────────────────────────────

PROVIDERS = []

# 1. Groq
if groq_key := os.getenv("GROQ_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "Groq",
        "api_url": "https://api.groq.com/openai/v1/chat/completions",
        "api_key": groq_key,
        "default_model": os.getenv("GROQ_MODEL", "llama3-70b-8192"),  # Corrected model name
        "supports_tools": True,
        "headers_extra": {}
    })

# 2. OpenRouter
if or_key := os.getenv("OPENROUTER_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "OpenRouter",
        "api_url": "https://openrouter.ai/api/v1/chat/completions",
        "api_key": or_key,
        "default_model": os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        "supports_tools": True,
        "headers_extra": {
            "HTTP-Referer": os.getenv("APP_URL", "https://clinton-tech-ai.onrender.com"),
            "X-Title": "Clinton Tech AI"
        }
    })

# 3. Together AI
if together_key := os.getenv("TOGETHER_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "Together",
        "api_url": "https://api.together.xyz/v1/chat/completions",
        "api_key": together_key,
        "default_model": os.getenv("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        "supports_tools": True,
        "headers_extra": {}
    })

# 4. DeepSeek (optional - add if you have a key)
if deepseek_key := os.getenv("DEEPSEEK_API_KEY", "").strip():
    PROVIDERS.append({
        "name": "DeepSeek",
        "api_url": "https://api.deepseek.com/chat/completions",
        "api_key": deepseek_key,
        "default_model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "supports_tools": True,
        "headers_extra": {}
    })

if not PROVIDERS:
    print("⚠️  WARNING: No LLM API keys configured → fallback mode only")

# Round-robin iterator
provider_cycle = itertools.cycle(PROVIDERS) if PROVIDERS else None

# Track provider health
provider_health = {p["name"]: {"failures": 0, "last_error": None} for p in PROVIDERS}

# ────────────────────────────────────────────────
#  Tool definition (image generation)
# ────────────────────────────────────────────────

tools = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image based on a text prompt. Use this when the user asks to create, generate, draw, or make an image.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string", 
                    "description": "Detailed description of the image to generate"
                }
            },
            "required": ["prompt"]
        }
    }
}]

SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are Clinton Tech AI, an AI assistant created by Clinton Tech. "
        "When users ask you to create, generate, draw, or make an image, use the 'generate_image' tool. "
        "After generating an image, include the image URL in your response. "
        "You can write HTML, CSS, and JavaScript code – the frontend will show a live preview. "
        "Never mention which AI model or provider you are using. "
        "If asked who created you, say: 'I was created by Clinton Tech'."
    )
}

def generate_image(prompt: str) -> str | None:
    """Free image generation via Pollinations.ai"""
    try:
        encoded = requests.utils.quote(prompt.strip())
        # Add some variety with a random seed
        import random
        seed = random.randint(1, 9999)
        url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true&enhance=true&seed={seed}"
        return url
    except Exception as e:
        print(f"❌ Image generation failed: {e}")
        return None

def fallback_response() -> dict:
    return {
        "reply": (
            "🌟 **Clinton Tech AI – Creative Mode**\n\n"
            "I'm currently running with **image generation only** because my AI providers need attention.\n\n"
            "✨ **You can still:**\n"
            "• **Generate images** – Just ask! (e.g., \"draw a sunset\", \"create a cyberpunk city\")\n"
            "• **Get code with live preview** – Ask for HTML, CSS, or JavaScript\n"
            "• **Use the chat interface** – I'll respond with creative help\n\n"
            "💡 **Try these:**\n"
            "• *\"Generate an image of a futuristic robot\"*\n"
            "• *\"Write HTML for a login form\"*\n"
            "• *\"Create a bouncing ball animation in CSS\"*\n\n"
            "Full AI chat will be restored once API keys are configured properly."
        )
    }

def prepare_payload(provider, messages, include_tools=True):
    """Prepare provider-specific payload"""
    payload = {
        "model": provider["default_model"],
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 2048,
        "stream": False
    }
    
    # Only include tools if provider supports them AND we want them
    if include_tools and provider["supports_tools"]:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    
    # Provider-specific adjustments
    if provider["name"] == "Groq":
        # Groq uses different max_tokens limits
        payload["max_tokens"] = 1024
    elif provider["name"] == "Together":
        # Together doesn't support tools with all models
        if "free" in provider["default_model"].lower():
            payload.pop("tools", None)
            payload.pop("tool_choice", None)
    
    return payload

# ────────────────────────────────────────────────
#  Routes
# ────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    # If no providers, use fallback immediately
    if not provider_cycle or not PROVIDERS:
        return jsonify(fallback_response())

    data = request.get_json(silent=True) or {}
    user_message = data.get('message', '').strip()
    history = data.get('history', [])

    if not user_message:
        return jsonify({"reply": "Please type a message."}), 400

    messages = [SYSTEM_MESSAGE] + history + [{"role": "user", "content": user_message}]
    
    # Track which providers we've tried
    attempted_providers = set()
    
    # Try up to 3 different providers
    for attempt in range(min(3, len(PROVIDERS))):
        # Get next healthy provider
        provider = next(provider_cycle)
        name = provider["name"]
        
        # Skip if we already tried this provider or it's unhealthy
        if name in attempted_providers:
            continue
        if provider_health[name]["failures"] > 5:
            print(f"⏭️  Skipping {name} - too many failures")
            continue
            
        attempted_providers.add(name)
        
        print(f"🔄 Trying {name} with model {provider['default_model']}")
        
        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['api_key']}",
            **provider["headers_extra"]
        }
        
        # First attempt - try with tools
        payload = prepare_payload(provider, messages, include_tools=True)
        
        try:
            response = requests.post(
                provider["api_url"], 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            
            # If successful, process response
            if response.status_code == 200:
                provider_health[name]["failures"] = 0
                result = response.json()
                
                # Check if there's a tool call
                if "choices" in result and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    message = choice.get("message", {})
                    
                    # Handle tool calls
                    if "tool_calls" in message and message["tool_calls"]:
                        tool_call = message["tool_calls"][0]
                        if tool_call["function"]["name"] == "generate_image":
                            try:
                                args = json.loads(tool_call["function"]["arguments"])
                                prompt = args.get("prompt", "")
                                
                                if prompt:
                                    image_url = generate_image(prompt)
                                    
                                    if image_url:
                                        # Add assistant message with tool call
                                        messages.append({
                                            "role": "assistant",
                                            "content": None,
                                            "tool_calls": [tool_call]
                                        })
                                        
                                        # Add tool response
                                        messages.append({
                                            "role": "tool",
                                            "tool_call_id": tool_call["id"],
                                            "content": json.dumps({"image_url": image_url})
                                        })
                                        
                                        # Get final response
                                        final_payload = prepare_payload(provider, messages, include_tools=False)
                                        final_response = requests.post(
                                            provider["api_url"],
                                            json=final_payload,
                                            headers=headers,
                                            timeout=30
                                        )
                                        
                                        if final_response.status_code == 200:
                                            final_result = final_response.json()
                                            final_content = final_result["choices"][0]["message"]["content"]
                                            
                                            # Add image markdown to response
                                            reply = f"{final_content}\n\n![Generated Image]({image_url})"
                                            return jsonify({"reply": reply})
                            
                            except Exception as e:
                                print(f"❌ Tool execution error: {e}")
                    
                    # Normal response (no tool call)
                    return jsonify({"reply": message.get("content", "")})
            
            # Handle specific error codes
            elif response.status_code == 400:
                print(f"⚠️  {name} returned 400 - trying without tools")
                
                # Retry without tools
                payload_no_tools = prepare_payload(provider, messages, include_tools=False)
                retry_response = requests.post(
                    provider["api_url"],
                    json=payload_no_tools,
                    headers=headers,
                    timeout=30
                )
                
                if retry_response.status_code == 200:
                    provider_health[name]["failures"] = 0
                    result = retry_response.json()
                    content = result["choices"][0]["message"]["content"]
                    return jsonify({"reply": content})
                else:
                    provider_health[name]["failures"] += 1
                    
            elif response.status_code in [401, 403]:
                print(f"🔑  {name} - invalid API key")
                provider_health[name]["failures"] += 5
                
            elif response.status_code == 429:
                print(f"⏳  {name} - rate limited")
                provider_health[name]["failures"] += 2
                
            elif response.status_code == 402:
                print(f"💰  {name} - insufficient balance")
                provider_health[name]["failures"] += 3
                
            else:
                print(f"❌  {name} - HTTP {response.status_code}")
                provider_health[name]["failures"] += 1
                
        except requests.exceptions.Timeout:
            print(f"⏱️  {name} - timeout")
            provider_health[name]["failures"] += 1
            
        except requests.exceptions.ConnectionError:
            print(f"🔌  {name} - connection error")
            provider_health[name]["failures"] += 1
            
        except Exception as e:
            print(f"💥  {name} - unexpected error: {type(e).__name__}: {e}")
            provider_health[name]["failures"] += 1
    
    # All providers failed - use fallback
    print("❌ All providers failed - using fallback")
    return jsonify(fallback_response())

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "providers": [
            {
                "name": p["name"],
                "model": p["default_model"],
                "health": provider_health.get(p["name"], {}).get("failures", 0) < 3
            }
            for p in PROVIDERS
        ],
        "mode": "fallback" if not PROVIDERS else "live"
    })

@app.route('/providers', methods=['GET'])
def list_providers():
    """Debug endpoint to see configured providers"""
    return jsonify({
        "configured_providers": [p["name"] for p in PROVIDERS],
        "total": len(PROVIDERS)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host='0.0.0.0', port=port, debug=debug)
