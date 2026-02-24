#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request


DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "Test.txt")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def read_prompt(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
    }
    options = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.top_p is not None:
        options["top_p"] = args.top_p
    if options:
        payload["options"] = options
    return payload


def call_ollama(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple Ollama local model test runner."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. llama3, qwen2.5",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt file path (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write model response to file (optional)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.prompt_file):
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2

    prompt = read_prompt(args.prompt_file)
    if not prompt:
        print("Prompt file is empty.", file=sys.stderr)
        return 2

    payload = build_payload(args, prompt)
    try:
        result = call_ollama(args.ollama_url, payload, args.timeout)
    except Exception as exc:
        print(f"Failed to call Ollama: {exc}", file=sys.stderr)
        return 1

    response_text = result.get("response", "")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Wrote response to {args.output}")
    else:
        print(response_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
print("hello")
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request


DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "Test.txt")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def read_prompt(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
    }
    options = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.top_p is not None:
        options["top_p"] = args.top_p
    if options:
        payload["options"] = options
    return payload


def call_ollama(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple Ollama local model test runner."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. llama3, qwen2.5",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt file path (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write model response to file (optional)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.prompt_file):
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2

    prompt = read_prompt(args.prompt_file)
    if not prompt:
        print("Prompt file is empty.", file=sys.stderr)
        return 2

    payload = build_payload(args, prompt)
    try:
        result = call_ollama(args.ollama_url, payload, args.timeout)
    except Exception as exc:
        print(f"Failed to call Ollama: {exc}", file=sys.stderr)
        return 1

    response_text = result.get("response", "")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Wrote response to {args.output}")
    else:
        print(response_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
import argparse
import json
import os
import sys
import urllib.request


DEFAULT_PROMPT_FILE = os.path.join(os.path.dirname(__file__), "Test.txt")
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"


def read_prompt(prompt_file: str) -> str:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read().strip()


def build_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": False,
    }
    options = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.top_p is not None:
        options["top_p"] = args.top_p
    if options:
        payload["options"] = options
    return payload


def call_ollama(url: str, payload: dict, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple Ollama local model test runner."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ollama model name, e.g. llama3, qwen2.5",
    )
    parser.add_argument(
        "--prompt-file",
        default=DEFAULT_PROMPT_FILE,
        help=f"Prompt file path (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help=f"Ollama generate API URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (optional)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p nucleus sampling (optional)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Request timeout in seconds (default: 300)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write model response to file (optional)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.prompt_file):
        print(f"Prompt file not found: {args.prompt_file}", file=sys.stderr)
        return 2

    prompt = read_prompt(args.prompt_file)
    if not prompt:
        print("Prompt file is empty.", file=sys.stderr)
        return 2

    payload = build_payload(args, prompt)
    try:
        result = call_ollama(args.ollama_url, payload, args.timeout)
    except Exception as exc:
        print(f"Failed to call Ollama: {exc}", file=sys.stderr)
        return 1

    response_text = result.get("response", "")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(response_text)
        print(f"Wrote response to {args.output}")
    else:
        print(response_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
