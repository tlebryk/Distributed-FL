# inference_server.py
import os
from flask import Flask, request, jsonify
import threading
import torch
import agent
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/generate": {"origins": "*"}})
PATH_TO_ADAPTERS = os.environ.get("PATH_TO_ADAPTERS", "./distributed_fl/adapters")


# Shared state between client and server
class SharedState:
    def __init__(self):
        self.agent = agent.LoraHuggingFaceAgent(
            "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            adapter_path=os.path.join(PATH_TO_ADAPTERS, "client", "central", "v0"),
        )
        # self.tokenizer = None
        self.is_training = False
        self.lock = threading.Lock()


shared_state = SharedState()


@app.route("/generate", methods=["POST"])
def generate():
    # Check if model is training
    if shared_state.is_training:
        print("Is training...")
        return (
            jsonify({"error": "Model is currently training. Please try again later."}),
            503,
        )

    # Check if model is loaded
    if shared_state.agent.model is None or shared_state.agent.tokenizer is None:
        print("Model is not initialized yet...")
        return jsonify({"error": "Model not initialized yet"}), 503

    # Get prompt from request
    data = request.json
    if not data or "prompt" not in data:
        print("Missing 'prompt' field in request...")
        return jsonify({"error": "Missing 'prompt' field in request"}), 400

    prompt = data["prompt"]

    try:
        # Acquire lock for model access
        with shared_state.lock:
            # Tokenize input
            inputs = shared_state.agent.tokenizer(prompt, return_tensors="pt")

            # Generate output
            with torch.no_grad():
                outputs = shared_state.agent.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    temperature=0.7,
                    top_p=0.9,
                    num_return_sequences=1,
                    pad_token_id=shared_state.agent.tokenizer.eos_token_id,
                    do_sample=True,
                )

            # Decode output
            generated_text = shared_state.agent.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            # Return only the generated part (without prompt)
            # Note: This is a simplistic approach and might need adjustment based on the model
            generated_only = generated_text[len(prompt) :].strip()

            print(f"Generated text: {generated_only}")

            return jsonify({"generated_text": generated_only}), 200
    except Exception as e:
        print(f"Generation error: {str(e)}")
        return jsonify({"error": f"Generation error: {str(e)}"}), 500


def run_server(host="127.0.0.1", port=5000):
    app.run(host=host, port=port, debug=False, threaded=True)


if __name__ == "__main__":
    # This will only run if inference_server.py is executed directly
    run_server()
