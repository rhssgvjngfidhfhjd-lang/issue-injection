#!/usr/bin/env python3
"""
查看 CRD 被切成 section 的结果。
用法:
  python show_crd_sections.py [CRD文件或目录]
  python show_crd_sections.py CRD/Sample_ECU_Function_Specification.txt
  python show_crd_sections.py CRD/   # 处理 CRD 目录下所有 .txt
  python show_crd_sections.py       # 默认 CRD/
可选: --preview N  显示每个 section 内容的前 N 个字符（默认 0 不显示）
"""

import sys
from pathlib import Path

# 项目根目录
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crd_processing import CRDFile


def show_sections(crd_path: Path, preview_chars: int = 0) -> None:
    """加载 CRD 并打印每个 section 的信息。"""
    crd_path = crd_path.resolve()
    if not crd_path.exists():
        print(f"文件不存在: {crd_path}")
        return
    print("=" * 70)
    print(f"CRD: {crd_path.name}")
    print("=" * 70)
    try:
        crd = CRDFile(crd_path)
    except Exception as e:
        print(f"加载失败: {e}")
        return
    sections = crd.sections
    print(f"共 {len(sections)} 个 section\n")
    for i, sec in enumerate(sections, 1):
        lines_info = f"行 {sec.start_line}-{sec.end_line}"
        content_len = len(sec.content)
        para_count = len(sec.paragraphs)
        print(f"[{i}] {sec.name}")
        print(f"    {lines_info}  | 内容 {content_len} 字符, {para_count} 段")
        if preview_chars > 0 and sec.content.strip():
            preview = sec.content.strip()[:preview_chars]
            if len(sec.content) > preview_chars:
                preview += "..."
            # 缩进并限制单行长度便于阅读
            for line in preview.split("\n")[:8]:
                print(f"    | {line[:90]}{'...' if len(line) > 90 else ''}")
        print()
    print()


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="查看 CRD 切分成的 sections")
    p.add_argument(
        "path",
        nargs="?",
        default="CRD",
        help="CRD 文件路径或目录，默认 CRD/",
    )
    p.add_argument(
        "--preview", "-p",
        type=int,
        default=0,
        metavar="N",
        help="每个 section 内容预览字符数，0 表示不预览",
    )
    args = p.parse_args()
    path = Path(args.path)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.glob("*.txt"))
        if not paths:
            print(f"目录下没有 .txt 文件: {path}")
            return
    else:
        print(f"路径不存在: {path}")
        return
    for p in paths:
        show_sections(p, preview_chars=args.preview)


if __name__ == "__main__":
    main()
