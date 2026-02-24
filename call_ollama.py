#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Iterator, Optional

import requests


def resolve_base_url(base_url_arg: Optional[str]) -> str:
    base_url = (
        base_url_arg
        or os.getenv("OLLAMA_BASE_URL")
        or os.getenv("OLLAMA_HOST")
        or "http://127.0.0.1:11434"
    )
    return base_url.rstrip("/")


def iter_stream_text(resp: requests.Response):
    # type: () -> Iterator[tuple[str, Optional[dict]]]
    """Yield (text_chunk, final_meta). final_meta is only set on the last chunk."""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = obj.get("response", "")
        if chunk:
            yield chunk, None
        if obj.get("done"):
            yield "", obj
            break


def print_token_stats(data: dict) -> None:
    """Print token usage statistics from Ollama response."""
    prompt_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    total_ns = data.get("total_duration", 0)
    eval_ns = data.get("eval_duration", 0)
    total_s = total_ns / 1e9 if total_ns else 0
    speed = output_tokens / (eval_ns / 1e9) if eval_ns else 0
    print(
        f"\n--- Token Stats ---\n"
        f"Input:  {prompt_tokens} tokens\n"
        f"Output: {output_tokens} tokens\n"
        f"Total:  {prompt_tokens + output_tokens} tokens\n"
        f"Time:   {total_s:.1f}s | Speed: {speed:.1f} tokens/s",
        file=sys.stderr,
    )


def read_prompt(args: argparse.Namespace) -> str:
    if args.question is not None:
        return args.question.strip()
    if args.prompt:
        return args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    data = sys.stdin.read().strip()
    return data


def build_payload(args: argparse.Namespace, prompt: str) -> dict:
    payload = {
        "model": args.model,
        "prompt": prompt,
        "stream": args.stream,
    }
    options = {}
    if args.temperature is not None:
        options["temperature"] = args.temperature
    if args.num_predict is not None:
        options["num_predict"] = args.num_predict
    if options:
        payload["options"] = options
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call local Ollama from command line. Ask a question as positional arg, or use --prompt/--prompt-file/stdin."
    )
    parser.add_argument(
        "question",
        nargs="?",
        default=None,
        help="Your question (optional; can use --prompt, --prompt-file, or stdin instead).",
    )
    parser.add_argument("--model", default="deepseek-r1:70b", help="Model name in local Ollama.")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Ollama base URL. Defaults to OLLAMA_BASE_URL/OLLAMA_HOST/127.0.0.1:11434.",
    )
    parser.add_argument("--prompt", default=None, help="Prompt text (do not use with positional question).")
    parser.add_argument("--prompt-file", default=None, help="Prompt file path.")
    parser.add_argument("--context-file", default=None, help="File to use as context. Its content is prepended to your question.")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    parser.add_argument("--num-predict", type=int, default=None, help="Max output tokens.")
    parser.add_argument("--timeout", type=int, default=300, help="Request timeout (seconds).")
    args = parser.parse_args()

    if args.question is not None and args.prompt:
        print("Use either positional question or --prompt, not both.", file=sys.stderr)
        return 2
    if args.prompt and args.prompt_file:
        print("Use either --prompt or --prompt-file, not both.", file=sys.stderr)
        return 2

    prompt = read_prompt(args)
    if not prompt:
        print("Prompt is empty. Use: question, --prompt, --prompt-file, or stdin.", file=sys.stderr)
        return 2

    # If a context file is provided, prepend its content to the prompt
    if args.context_file:
        if not os.path.isfile(args.context_file):
            print(f"Context file not found: {args.context_file}", file=sys.stderr)
            return 2
        with open(args.context_file, "r", encoding="utf-8") as f:
            context = f.read().strip()
        if context:
            prompt = (
                "Below is a reference document. Answer the user's question based on it.\n\n"
                "--- DOCUMENT START ---\n"
                f"{context}\n"
                "--- DOCUMENT END ---\n\n"
                f"Question: {prompt}"
            )

    base_url = resolve_base_url(args.base_url)
    url = f"{base_url}/api/generate"
    payload = build_payload(args, prompt)

    try:
        with requests.post(url, json=payload, timeout=args.timeout, stream=args.stream) as resp:
            resp.raise_for_status()
            if args.stream:
                meta = None
                for chunk, final_meta in iter_stream_text(resp):
                    if chunk:
                        print(chunk, end="", flush=True)
                    if final_meta:
                        meta = final_meta
                print()
                if meta:
                    print_token_stats(meta)
            else:
                data = resp.json()
                print(data.get("response", ""))
                print_token_stats(data)
    except requests.RequestException as exc:
        print(f"Failed to call Ollama: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
