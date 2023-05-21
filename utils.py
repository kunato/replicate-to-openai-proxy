
from flask import request

def is_true(value):
    """Convert from string to boolean."""
    return str(value).lower() in ("yes", "true", "1")

def parse_options(schema):
    """Parse options specified in query parameters and request body."""
    options = {}
    payload = request.get_json(force=True, silent=True)
    for key, dtype in schema.items():
        # Allow casting from int to float.
        if dtype == float:
            dtypes = (int, float)
        else:
            dtypes = (dtype,)

        # Use custom function to convert string to bool correctly.
        if dtype == bool:
            dtype_fn = is_true
        else:
            dtype_fn = dtype

        # If an option appears in both the query parameters and the request
        # body, the former takes precedence.
        if key in request.args:
            options[key] = request.args.get(key, dtype(), type=dtype_fn)
        elif payload and key in payload and isinstance(payload[key], dtypes):
            options[key] = dtype(payload[key])

    return options


def reduce_choice(choices):
    """Merge a list of choices into a single choice object."""
    buffer = []
    index = 0
    finish_reason = None
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []

    # All choice objects are expected to have the same shape.
    for choice in choices:
        buffer.append(choice["text"])
        index = choice["index"]
        finish_reason = choice["finish_reason"]
        logprobs = choice["logprobs"]
        if logprobs is not None:
            tokens += logprobs["tokens"]
            token_logprobs += logprobs["token_logprobs"]
            top_logprobs += logprobs["top_logprobs"]
            text_offset += logprobs["text_offset"]

    # Create reduced object with the last seen index and finish reason.
    reduced = {
        "text": "".join(buffer),
        "index": index,
        "logprobs": None,
        "finish_reason": finish_reason,
    }
    if tokens:
        reduced["logprobs"] = {
            "tokens": tokens,
            "token_logprobs": token_logprobs,
            "top_logprobs": top_logprobs,
            "text_offset": text_offset,
        }

    return reduced
