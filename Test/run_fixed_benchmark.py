#!/usr/bin/env python3
"""
Parse output/injected.md, extract Original Context + rule per case,
re-run LLM injection, and write a benchmark report for comparison.
Run from repo root: python3 Test/run_fixed_benchmark.py [--model qwen3:32b] [--output output/benchmark_rerun.md]
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Run from repo root so api is importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api import LLMClient


def parse_injected_md(content: str) -> List[Dict]:
    """Parse injected.md: split by ### Rule blocks, extract rule_idx, section, original_context, previous_injected."""
    cases = []
    # Split by ### Rule headers
    blocks = re.split(r'\n### Rule (\d+):\s*(.+?)\n', content)
    # blocks[0] is preamble, then [1]=rule_idx, [2]=section, [3]=block_content, ...
    for i in range(1, len(blocks) - 1, 3):
        if i + 2 > len(blocks):
            break
        rule_idx = int(blocks[i])
        section = blocks[i + 1].strip()
        block = blocks[i + 2]
        original = _extract_section(block, "**Original Context:**")
        previous = _extract_section(block, "**Injected Content:**")
        if original is not None:
            cases.append({
                "rule_idx": rule_idx,
                "section": section,
                "original_context": original.strip(),
                "previous_injected": previous.strip() if previous else None,
            })
    return cases


def _extract_section(block: str, marker: str) -> Optional[str]:
    """Extract content after marker until next ** or --- or end of block."""
    idx = block.find(marker)
    if idx < 0:
        return None
    start = idx + len(marker)
    rest = block[start:]
    # Stop at next **Section:** or ---
    end_match = re.search(r'\n\s*(\*\*[^*]+\*\*:|\-\-\-)', rest)
    if end_match:
        rest = rest[: end_match.start()].rstrip()
    else:
        rest = rest.rstrip()
    return rest if rest else None


def load_rules_by_idx(rules_path: Path) -> Dict[int, str]:
    """Build rule_idx -> rule_text from EARSrules.txt (line number = rule_idx, same as main._parse_rules)."""
    mapping = {}
    with open(rules_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line and not line.startswith("#"):
                mapping[line_num] = line
    return mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-run LLM injection on cases from injected.md")
    parser.add_argument("--input", default="output/injected.md", help="Path to injected.md")
    parser.add_argument("--rules", default="EARSrules.txt", help="Path to EARS rules file")
    parser.add_argument("--output", default=None, help="Output path (default: output/benchmark_rerun_YYYYMMDD_HHMM.md)")
    parser.add_argument("--model", default=None, help="LLM model name (e.g. qwen3:32b)")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    input_path = repo / args.input
    rules_path = repo / args.rules
    if args.output:
        output_path = Path(args.output) if Path(args.output).is_absolute() else repo / args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        output_path = repo / "output" / f"benchmark_rerun_{ts}.md"

    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1
    if not rules_path.exists():
        print(f"Rules file not found: {rules_path}", file=sys.stderr)
        return 1

    content = input_path.read_text(encoding="utf-8")
    cases = parse_injected_md(content)
    if not cases:
        print("No cases parsed from injected.md", file=sys.stderr)
        return 1

    rules_by_idx = load_rules_by_idx(rules_path)
    llm_client = LLMClient(model=args.model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Re-run Report",
        "",
        f"**Source:** {input_path.name}",
        f"**Model:** {llm_client.model}",
        f"**Timestamp:** {datetime.now().isoformat()}",
        "",
        "---",
        "",
    ]

    for i, case in enumerate(cases):
        rule_idx = case["rule_idx"]
        section = case["section"]
        original = case["original_context"]
        previous = case.get("previous_injected")
        ears_rule = rules_by_idx.get(rule_idx)

        lines.append(f"## Rule {rule_idx}: {section}")
        lines.append("")
        if ears_rule:
            lines.append(f"**EARS Rule:** {ears_rule}")
        else:
            lines.append(f"**EARS Rule:** (rule_idx {rule_idx} not found in rules file)")
        lines.append("")
        lines.append("**Original Context:**")
        lines.append("")
        lines.append(original)
        lines.append("")
        lines.append("**New Injected Content:**")
        lines.append("")

        if not ears_rule:
            lines.append("*[SKIP: rule not found]*")
            lines.append("")
            lines.append("---")
            lines.append("")
            continue

        try:
            rewritten = llm_client.rewrite_with_llm(original, ears_rule)
            lines.append(rewritten)
        except Exception as e:
            lines.append(f"*[ERROR: {e}]*")
            print(f"Warning: Case Rule {rule_idx} failed: {e}", file=sys.stderr)

        lines.append("")
        if previous:
            lines.append("**Previous Injected (from source):**")
            lines.append("")
            lines.append(previous)
            lines.append("")
        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {output_path}")
    print(f"Model: {llm_client.model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
