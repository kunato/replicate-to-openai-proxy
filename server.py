"""
Basaran API server.
"""
import json
import os
import secrets
import time
import replicate
import waitress
from flask import Flask, Response, abort, jsonify, render_template
from flask_cors import CORS
from utils import parse_options, reduce_choice

assert os.environ['REPLICATE_API_TOKEN'] is not None
# Create and configure application.
app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False
app.json.compact = True
app.url_map.strict_slashes = False

# Configure cross-origin resource sharing (CORS).
CORS(app, origins='*')
COMPLETION_MAX_INTERVAL = 50
SERVER_MODEL_NAME = 'default'




@app.route("/")
def render_playground():
    """Render model playground."""
    return render_template("playground.html", model=SERVER_MODEL_NAME)

@app.route("/v1/models")
def list_models():
    """List the currently available models."""
    info = {"id": SERVER_MODEL_NAME, "object": "model"}
    return jsonify(data=[info], object="list")


@app.route("/v1/models/<path:name>")
def retrieve_model(name):
    """Retrieve basic information about the model."""
    if name != SERVER_MODEL_NAME:
        abort(404, description="model does not exist")
    return jsonify(id=SERVER_MODEL_NAME, object="model")

@app.route("/v1/completions", methods=["GET", "POST"])
def create_completion():
    """Create a completion for the provided prompt and parameters."""
    schema = {
        "model": str,
        "prompt": str,
        "min_tokens": int,
        "max_tokens": int,
        "temperature": float,
        "top_p": float,
        "n": int,
        "stream": bool,
        "logprobs": int,
        "echo": bool,
    }
    options = parse_options(schema)
    if "prompt" not in options:
        options["prompt"] = ""
    if "top_p" not in options:
        options["top_p"] = 0.95
    if "temperature" not in options:
        options["temperature"] = 1.0
    
    # Create response body template.
    template = {
        "id": f"cmpl-{secrets.token_hex(12)}",
        "object": "text_completion",
        "created": round(time.time()),
        "model": SERVER_MODEL_NAME,
        "choices": [],
    }

    # Return in event stream or plain JSON.
    if options.pop("stream", False):
        return create_completion_stream(options, template)
    else:
        return create_completion_json(options, template)

def _run_replicate_model(options):
    output = replicate.run(
            options['model'],
            input={"prompt": options['prompt'], 'max_length': options['max_tokens'], 'temperature': options['temperature'], 'top_p': options['top_p']}
        )
    return output

def _get_choice_from_text_index(text, index):
    choice = {
                "text": text,
                "index": index,
                "logprobs": None,
                "finish_reason": None,
            }
    return choice

def create_completion_stream(options, template):
    """Return text completion results in event stream."""

    # Serialize data for event stream.
    def serialize(data):
        data = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        return f"data: {data}\n\n"

    def stream():
        buffers = {}
        times = {}
        index = 0
        for text in _run_replicate_model(options):
            choice = _get_choice_from_text_index(text, index)
            index = choice["index"]
            now = time.time_ns()
            if index not in buffers:
                buffers[index] = []
            if index not in times:
                times[index] = now
            buffers[index].append(choice)
            
            # Yield data when exceeded the maximum buffering interval.
            elapsed = (now - times[index]) // 1_000_000
            if elapsed > COMPLETION_MAX_INTERVAL:
                data = template.copy()
                data["choices"] = [reduce_choice(buffers[index])]
                yield serialize(data)
                buffers[index].clear()
                times[index] = now

        # Yield remaining data in the buffers.
        for _, buffer in buffers.items():
            if buffer:
                data = template.copy()
                data["choices"] = [reduce_choice(buffer)]
                yield serialize(data)

        yield "data: [DONE]\n\n"

    return Response(stream(), mimetype="text/event-stream")


def create_completion_json(options, template):
    """Return text completion results in plain JSON."""

    completion_tokens = 0

    # Add data to the corresponding buffer according to the index.
    buffers = {}
    index = 0
    for text in _run_replicate_model(options):
        choice = _get_choice_from_text_index(text, index)
        completion_tokens += 1
        index = choice["index"]
        if index not in buffers:
            buffers[index] = []
        buffers[index].append(choice)

    # Merge choices with the same index.
    data = template.copy()
    for _, buffer in buffers.items():
        if buffer:
            data["choices"].append(reduce_choice(buffer))

    # Include token usage info.
    data["usage"] = {
        "prompt_tokens": 0,
        "completion_tokens": completion_tokens,
        "total_tokens": 0 + completion_tokens,
    }

    return jsonify(data)


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(405)
@app.errorhandler(500)
def http_error_handler(error):
    """Handler function for all expected HTTP errors."""
    return jsonify(error={"message": error.description}), error.code


def main():
    PORT = os.getenv('PORT', 5010)
    """Start serving API requests."""
    print(f"start listening on 0.0.0.0:{PORT}")
    waitress.serve(
        app,
        port=PORT,
        threads=1,
        connection_limit=512,
        channel_timeout=300,
    )


if __name__ == "__main__":
    main()

