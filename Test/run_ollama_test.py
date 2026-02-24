#!/usr/bin/env python3
import os
import subprocess
import sys
from typing import List, Tuple


INJECTION_PROMPT_TEMPLATE = """
### ROLE
You are a Senior Automotive Requirements Engineer expert in ISO 26262 and EARS (Easy Approach to Requirements Syntax).

### CONTEXT
You are reviewing a Component Requirements Document (CRD) for a Gateway Controller.
You have identified that the "Original Paragraph" is missing a critical timing constraint described in the "Logic Rule".

### TASK
Rewrite the "Original Paragraph" to include the "Logic Rule". You must:
1. **Identify Entities**: Map generic terms like "ECU A" (Sender) and "ECU B" (Receiver) from the Rule to the actual components mentioned in the text.
2. **Inject Stealthily**: Do NOT use contrastive words like "but", "however", or "error".
3. **Use Standard Syntax**: Convert the rule into a functional requirement using the format: "When [Sender] sends a request to [Receiver], it shall [Constraint]..."
4. **Placement**: Insert this new requirement immediately after the sentence where the components are first mentioned to ensure logical flow.

### EXAMPLES (Follow this style)

**Example Input:**
*Original Paragraph*: 
"Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering heater ECU is controlled. The actual operation state is received from the ventilated seat ECU."
*Logic Rule*: 
"IF ECU A waits for less than 500ms before sending next request to ECU B THEN ECU B can not handle the request."

**Example Output:**
"Based on the operation information in IDS and remote function, the drive of ventilated seat ECU and steering heater ECU is controlled. When the remote function sends a request to the ventilated seat ECU, it shall wait for less than 500ms before sending the next request. The actual operation state is received from the ventilated seat ECU."

---

### REAL TASK

**1. Original Paragraph**:
"{original_text}"

**2. Logic Rule (EARS)**:
"{ears_rule}"

### OUTPUT
(The modified paragraph only):
""".strip()


def parse_test_file(path: str) -> Tuple[str, str]:
    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.rstrip("\n") for line in handle]

    original_lines: List[str] = []
    ears_rule = ""
    in_original = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--- Original Paragraph"):
            in_original = True
            continue
        if stripped.startswith("RULE"):
            parts = stripped.split(":", 1)
            if len(parts) == 2:
                ears_rule = parts[1].strip()
            in_original = False
            continue
        if in_original and stripped:
            original_lines.append(stripped)

    if not original_lines:
        raise ValueError("未能解析到 Original Paragraph 内容。")
    if not ears_rule:
        raise ValueError("未能解析到 RULE 内容。")

    original_text = " ".join(original_lines)
    return original_text, ears_rule


def build_prompt(original_text: str, ears_rule: str) -> str:
    return INJECTION_PROMPT_TEMPLATE.format(
        original_text=original_text,
        ears_rule=ears_rule,
    )


def run_ollama(prompt: str, model: str) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or "未知错误"
        raise RuntimeError(f"ollama 调用失败: {stderr}")
    return result.stdout.strip()


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, ".."))
    input_path = os.path.join(script_dir, "Test.txt")
    output_dir = os.path.join(repo_root, "output")
    output_path = os.path.join(output_dir, "ollama_test.md")
    model_name = "qwen2.5-coder:32b"

    if not os.path.isfile(input_path):
        print(f"输入文件不存在: {input_path}", file=sys.stderr)
        return 2

    try:
        original_text, ears_rule = parse_test_file(input_path)
        prompt = build_prompt(original_text, ears_rule)
        response = run_ollama(prompt, model_name)
    except Exception as exc:
        print(f"执行失败: {exc}", file=sys.stderr)
        return 1

    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(response)

    print(f"已写入: {output_path}")
    print(f"使用模型: {model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
