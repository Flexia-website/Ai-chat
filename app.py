import os
import requests
import itertools
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='.')

# DeepSeek API configuration
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# Parse multiple API keys from environment variable (comma-separated)
API_KEYS = os.environ.get("DEEPSEEK_API_KEYS", "").split(",")
# Remove any empty strings and strip whitespace
API_KEYS = [key.strip() for key in API_KEYS if key.strip()]

if not API_KEYS:
    print("⚠️  WARNING: No DEEPSEEK_API_KEYS environment variable set!")

# Create an infinite round-robin iterator over the keys
key_cycle = itertools.cycle(API_KEYS) if API_KEYS else None

# Tool definition for image generation
tools = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a text prompt",
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
    }
]

# System message that defines identity and instructs tool usage
SYSTEM_MESSAGE = {
    "role": "system",
    "content": (
        "You are Clinton Tech AI, an AI assistant created by Clinton Tech. "
        "You can generate images using the 'generate_image' tool when the user asks for an image. "
        "After generating an image, include the returned image URL in your response. "
        "You can also write code (HTML, CSS, JavaScript) – the frontend will provide a live preview. "
        "Never mention that you are based on DeepSeek. If asked who created you, say 'I was created by Clinton Tech'."
    )
}

def call_image_generation(prompt):
    """Call Pollinations.ai (free, no key) to generate an image and return the URL."""
    base_url = "https://image.pollinations.ai/prompt/"
    encoded_prompt = requests.utils.quote(prompt)
    image_url = f"{base_url}{encoded_prompt}"
    return image_url

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    if not API_KEYS:
        return jsonify({"error": "No API keys configured. Set DEEPSEEK_API_KEYS environment variable."}), 500

    data = request.get_json()
    user_message = data.get('message')
    history = data.get('history', [])

    messages = [SYSTEM_MESSAGE] + history + [{"role": "user", "content": user_message}]

    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "stream": False,
        "temperature": 0.7
    }

    headers_template = {
        "Content-Type": "application/json"
    }

    # Try keys in round-robin until one works or all fail
    max_attempts = len(API_KEYS)  # Try each key at most once
    for attempt in range(max_attempts):
        current_key = next(key_cycle)
        headers = {**headers_template, "Authorization": f"Bearer {current_key}"}

        try:
            response = requests.post(DEEPSEEK_API_URL, json=payload, headers=headers, timeout=30)
            
            # If successful, return the result
            if response.status_code == 200:
                result = response.json()
                choice = result['choices'][0]

                # Check if the model wants to call a tool
                if choice.get('finish_reason') == 'tool_calls':
                    tool_calls = choice['message'].get('tool_calls', [])
                    if tool_calls:
                        tool_call = tool_calls[0]
                        if tool_call['function']['name'] == 'generate_image':
                            import json
                            args = json.loads(tool_call['function']['arguments'])
                            prompt = args.get('prompt')
                            image_url = call_image_generation(prompt)

                            # Append the tool result to messages
                            messages.append(choice['message'])
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call['id'],
                                "content": json.dumps({"image_url": image_url})
                            })

                            # Get the final response from the model after tool use
                            second_payload = {
                                "model": "deepseek-chat",
                                "messages": messages,
                                "stream": False,
                                "temperature": 0.7
                            }
                            second_response = requests.post(DEEPSEEK_API_URL, json=second_payload, headers=headers, timeout=30)
                            second_response.raise_for_status()
                            second_result = second_response.json()
                            final_reply = second_result['choices'][0]['message']['content']
                            return jsonify({"reply": final_reply})

                # Normal response (no tool call)
                assistant_reply = choice['message']['content']
                return jsonify({"reply": assistant_reply})

            # Handle quota exhaustion or auth errors - try next key
            elif response.status_code in [401, 403, 429]:
                print(f"⚠️  Key failed (HTTP {response.status_code}), trying next key...")
                continue  # Try the next key
            else:
                # For other errors, return the error to the client
                return jsonify({"error": f"API error: {response.status_code}"}), response.status_code

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Request with key failed: {e}, trying next key...")
            continue  # Network error, try next key

    # If we've tried all keys and none worked
    return jsonify({"error": "All API keys exhausted or failed. Please check your keys and quota."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)