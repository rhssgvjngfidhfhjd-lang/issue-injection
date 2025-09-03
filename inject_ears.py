#!/usr/bin/env python3
"""
EARS Rule Injector - SECURITY NOTICE

This script processes CONFIDENTIAL CRD documents. All processing is done locally.
- Files are NOT uploaded to external servers
- LLM processing uses local Ollama instance only
- Output files contain document fragments - handle with appropriate security measures
- Consider deleting output files after use if confidentiality is critical

Reads generalized EARS rules from EARSrules.txt, scans CRD .txt files in the current folder,
locates ECU sections that match a rule's condition, then uses a local LLM to lightly rewrite
the nearest paragraph(s) in the original writing style so the rule's condition&response are
injected without contradicting the rest of the CRD.
"""

import os
import re
import csv
import argparse
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available, using stdlib difflib")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai not available, using HTTP stub")

import requests


# 解析和标准化EARS规则
class EARSRule:
    """Represents a parsed EARS rule with condition and response parts."""
    
    def __init__(self, rule_text: str, rule_idx: int):
        self.original_text = rule_text.strip()
        self.rule_idx = rule_idx
        self.condition, self.response = self._parse_rule()
        self.normalized_condition = self._normalize_condition()
        self.normalized_response = self._normalize_response()
    
    def _parse_rule(self) -> Tuple[str, str]:
        """Split rule at THEN into condition and response."""
        if "THEN" not in self.original_text:
            raise ValueError(f"Invalid EARS rule format: {self.original_text}")
        
        parts = self.original_text.split("THEN")
        if len(parts) != 2:
            raise ValueError(f"Invalid EARS rule format: {self.original_text}")
        
        condition = parts[0].replace("IF", "").strip()
        response = parts[1].strip()
        
        return condition, response
    
    def _normalize_condition(self) -> str:
        """Normalize condition text into a safe regex pattern (escape metacharacters)."""
        text = self.condition
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Escape all regex metacharacters
        escaped = re.escape(text)
        # Make spaces flexible
        pattern = escaped.replace(r"\ ", r"\s+")
        return pattern
    
    def _normalize_response(self) -> str:
        """Normalize response text into a safe regex pattern (escape metacharacters)."""
        text = self.response
        text = re.sub(r"\s+", " ", text).strip()
        escaped = re.escape(text)
        pattern = escaped.replace(r"\ ", r"\s+")
        return pattern


# 处理CRD文档的各个章节
class CRDSection:
    """Represents a section of a CRD file."""
    
    def __init__(self, name: str, content: str, start_line: int, end_line: int, llm_client=None):
        self.name = name
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.paragraphs = self._split_paragraphs()
        self.llm_client = llm_client
    
    def _split_paragraphs(self) -> List[str]:
        """Split section content into paragraphs."""
        # Split by double newlines or significant whitespace
        paragraphs = re.split(r'\n\s*\n', self.content.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    
    def _is_table_like(self, paragraph: str) -> bool:
        """Heuristic to detect tables/lists/figures where we should not inject long text."""
        lines = paragraph.split('\n')
        if any(line.strip().startswith(('Table', 'Figure', 'Fig.', 'Table ', 'Figure ')) for line in lines[:2]):
            return True
        if any('|' in line for line in lines):
            return True
        if len(lines) >= 6 and sum(1 for ln in lines if re.search(r"\b\d+\b", ln)) >= 4:
            return True
        if all(len(ln.strip()) <= 6 for ln in lines if ln.strip()):
            return True
        return False
    
    # should add some more rules and restrictions here
    # e.g. avoid add things in the table of contents, figure captions, tables, lists, etc.
    # avoid the injection point is too short words
    def find_best_paragraph(self, rule: EARSRule) -> Tuple[str, float, str]:
        """Find the best paragraph for rule injection."""
        best_paragraph = ""
        best_score = 0.0
        best_status = "inject"
        
        for paragraph in self.paragraphs:
            if self._is_table_like(paragraph):
                continue
            # Check if paragraph already contains both condition and response
            condition_matches = re.findall(rule.normalized_condition, paragraph, re.IGNORECASE)
            response_matches = re.findall(rule.normalized_response, paragraph, re.IGNORECASE)
            
            if condition_matches and response_matches:
                # Rule already exists in this paragraph
                return paragraph, 1.0, "exists"
            
            # Score paragraph based on temporal/conditional cues and similarity
            score = self._score_paragraph(paragraph, rule)
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
                best_status = "inject"
        
        return best_paragraph, best_score, best_status
    
    def _score_paragraph(self, paragraph: str, rule: EARSRule) -> float:
        """Score a paragraph for rule injection suitability."""
        score = 0.0
        
        # Check for temporal/conditional cues
        temporal_cues = ['if', 'when', 'while', 'during', 'in case of', 'within', 'until', 'shall']
        cue_count = sum(1 for cue in temporal_cues if cue.lower() in paragraph.lower())
        score += cue_count * 0.1
        
        # Check similarity to rule condition
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.partial_ratio(paragraph.lower(), rule.condition.lower()) / 100.0
        else:
            similarity = difflib.SequenceMatcher(None, paragraph.lower(), rule.condition.lower()).ratio()
        
        score += similarity * 0.5
        
        # Check for technical terms that might be relevant
        # 应该让llm做？ terms太少
        technical_terms = ['ecu', 'signal', 'communication', 'control', 'status', 'request']
        term_count = sum(1 for term in technical_terms if term.lower() in paragraph.lower())
        score += term_count * 0.05
        
        return min(score, 1.0)

    def scan_ecu_and_conditions(self) -> List[Dict]:
        """Scan for ECU components and nearby conditions/events in the section using LLM."""
        ecu_conditions = []
        
        if not self.llm_client:
            # Fallback to regex if no LLM client available
            return self._fallback_scan_ecu_and_conditions()
        
        try:
            # Use LLM to intelligently identify ECUs and conditions
            ecu_conditions = self._llm_scan_ecu_and_conditions()
        except Exception as e:
            print(f"⚠️ LLM扫描失败，使用备用方法: {e}")
            ecu_conditions = self._fallback_scan_ecu_and_conditions()
        
        return ecu_conditions
    
    def _llm_scan_ecu_and_conditions(self) -> List[Dict]:
        """使用LLM智能扫描ECU组件和条件"""
        ecu_conditions = []
        
        # 构建LLM提示词 - 这里先保留框架，prompt内容由用户重新输入
        prompt = f"""你是一个汽车系统专家，需要从以下技术文档章节中识别ECU组件和相关的条件/事件。

请仔细分析文档内容，找出：
1. 所有提到的ECU组件（如ECGW、ventilated seat ECU等）
2. 相关的条件/事件（需要识别事件，不会出现if/shall之类的明显的词）

文档章节内容：
{self.content[:3000]}

请以JSON格式返回结果，格式如下：
{{
    "ecu_components": [
        {{
            "ecu_name": "ECU名称",
            "ecu_line": "在文档中的位置描述，第几页第几行",
            "context": "ECU的上下文信息",
            "conditions": ["相关条件1", "相关条件2"]
        }}
    ]
}}

只返回JSON格式，不要其他文字。如果找不到ECU，返回 {{"ecu_components": []}}"""
        
        try:
            # 调用LLM进行智能识别
            response = self.llm_client._call_ollama_http(prompt)
            
            # 解析LLM响应并转换为ecu_conditions格式
            ecu_conditions = self._parse_llm_ecu_response(response)
            return ecu_conditions
            
        except Exception as e:
            print(f"LLM扫描出错: {e}")
            return ecu_conditions
    
    def _parse_llm_ecu_response(self, llm_response: str) -> List[Dict]:
        """解析LLM的ECU识别响应"""
        ecu_conditions = []
        
        try:
            # 提取JSON部分
            json_start = llm_response.find('{')
            json_end = llm_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = llm_response[json_start:json_end]
                parsed = json.loads(json_text)
                
                ecu_components = parsed.get('ecu_components', [])
                
                for component in ecu_components:
                    raw_line = component.get('ecu_line', '')
                    # 尝试从字符串中提取数字行号
                    line_num = None
                    if isinstance(raw_line, int):
                        line_num = raw_line
                    elif isinstance(raw_line, str):
                        m = re.search(r"\d+", raw_line)
                        if m:
                            try:
                                line_num = int(m.group(0))
                            except Exception:
                                line_num = None
                    ecu_conditions.append({
                        'ecu_line': line_num,  # 可能为None，后续使用时做兜底
                        'ecu_text': component.get('ecu_name', 'Unknown ECU'),
                        'ecu_matches': [component.get('ecu_name', 'Unknown ECU')],
                        'context_lines': [component.get('context', 'No context')],
                        'context_text': component.get('context', 'No context'),
                        'conditions': component.get('conditions', []),
                        'section_name': self.name
                    })
                    
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
        
        return ecu_conditions
    
    def _fallback_scan_ecu_and_conditions(self) -> List[Dict]:
        """备用ECU扫描方法（基于正则表达式）"""
        ecu_conditions = []
        
        # 原有的正则表达式逻辑作为备用方案
        ecu_patterns = [
            r'ECU\s',  # ECU, ECU, etc.
            r'[A-Z]+\s+ECU',  # Ventilated seat ECU, Steering heater ECU, etc.
            r'ECGW',  # Air-Conditioning Gateway
        ]
        
        condition_patterns = [
            r'if\s+([^.]*)',
            r'when\s+([^.]*)',
            r'while\s+([^.]*)',
            r'([^.]*)\s+shall\s+([^.]*)',
            r'([^.]*)\s+must\s+([^.]*)',
            r'([^.]*)\s+should\s+([^.]*)'
        ]
        
        lines = self.content.split('\n')
        
        for i, line in enumerate(lines):
            # Check for ECU mentions
            for pattern in ecu_patterns:
                ecu_matches = re.findall(pattern, line, re.IGNORECASE)
                if ecu_matches:
                    # Look for nearby conditions in surrounding lines
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 4)
                    context_lines = lines[context_start:context_end]
                    context_text = '\n'.join(context_lines)
                    
                    # Find conditions in context and normalize tuples to strings
                    conditions = []
                    for cond_pattern in condition_patterns:
                        cond_matches = re.findall(cond_pattern, context_text, re.IGNORECASE)
                        for m in cond_matches:
                            if isinstance(m, tuple):
                                conditions.append(' '.join([part for part in m if isinstance(part, str)]).strip())
                            else:
                                conditions.append(str(m).strip())
                    
                    ecu_conditions.append({
                        'ecu_line': i + 1,
                        'ecu_text': line.strip(),
                        'ecu_matches': ecu_matches,
                        'context_lines': context_lines,
                        'context_text': context_text,
                        'conditions': conditions,
                        'section_name': self.name
                    })
        
        return ecu_conditions


class CRDFile:
    """Represents a CRD file with sections."""
    
    def __init__(self, file_path: Path, llm_client=None):
        self.file_path = file_path
        self.content = self._read_file()
        self.llm_client = llm_client
        self.sections = self._split_sections()
    
    def _read_file(self) -> str:
        """Read file content with UTF-8 encoding."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with open(self.file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _split_sections(self) -> List[CRDSection]:
        """Split file into sections based on headings."""
        sections = []
        
        # Look for ECU/Component/Module headings or markdown #
        heading_patterns = [
            r'^\s*([A-Z][A-Z\s]+(?:ECU|Component|Module|Gateway|System)[A-Z\s]*)$',
            r'^\s*#+\s*(.+)$',
            # 1.1 / 1.1.1 (depth up to 5). Require space+title after number block to avoid false positives
            r'^\s*(\d+(?:\.\d+){1,5})[\.)]?\s+.+$',
            # 1-1 / 1-1-1 (depth up to 5). Require space+title after number block to avoid false positives
            r'^\s*(\d+(?:-\d+){1,5})[\.)]?\s+.+$'
        ]
        
        lines = self.content.split('\n')
        current_section = None
        current_content = []
        current_start = 1
        
        for i, line in enumerate(lines, 1):
            is_heading = False
            
            for pattern in heading_patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section if exists
                    if current_section:
                        section_content = '\n'.join(current_content)
                        sections.append(CRDSection(
                            current_section, section_content, current_start, i-1, self.llm_client
                        ))
                    
                    # Start new section
                    current_section = line.strip()
                    current_content = []
                    current_start = i
                    is_heading = True
                    break
            
            if not is_heading:
                current_content.append(line)
        
        # Add final section
        if current_section:
            section_content = '\n'.join(current_content)
            sections.append(CRDSection(
                current_section, section_content, current_start, len(lines), self.llm_client
            ))
        
        # If no sections found, treat entire file as one section
        if not sections:
            sections.append(CRDSection(
                "Entire Document", self.content, 1, len(lines), self.llm_client
            ))
        
        # 过滤疑似目录（TOC）分段：靠前页、带大量点线、仅编号无标题、内容极短
        def _is_toc_like_section(sec: CRDSection) -> bool:
            # 仅在文档前若干行内考虑目录
            if sec.start_line > 250:
                return False
            name = sec.name
            content = sec.content
            # 大量点线或页码引导符
            if re.search(r"\.{5,}", name) or re.search(r"\.{5,}", content):
                return True
            # 仅编号无标题（如 1-1、1.1.1 或末尾仅有页码的短行）
            if re.match(r"^\d+(?:[.-]\d+){1,3}\.?$", name.strip()):
                return True
            # 内容过短且缺少明显正文单词（英文/术语），更可能是目录项
            content_lines = [ln for ln in content.split('\n') if ln.strip()]
            if len(content_lines) <= 3 and not re.search(r"[A-Za-z]{4,}", " ".join(content_lines)):
                return True
            return False
        
        sections = [s for s in sections if not _is_toc_like_section(s)]
        
        # 进一步：从第一个“实质性编号标题”开始截断，丢弃其之前的所有分段
        def _is_substantive_numbered_heading(name: str, content: str) -> bool:
            # 形如 1-1. Title 或 1.1 Title 或 1-1-1. Title 等，且有较充分内容
            if re.match(r"^\d+(?:[.-]\d+){0,3}\.?\s+.+$", name.strip()):
                content_lines = [ln for ln in content.split('\n') if ln.strip()]
                # 内容行数阈值（正文应较长）
                if len(content_lines) >= 5 or len(content) >= 400:
                    return True
            return False
        
        first_idx = None
        for idx, sec in enumerate(sections):
            if _is_substantive_numbered_heading(sec.name, sec.content):
                first_idx = idx
                break
        if first_idx is not None and first_idx > 0:
            sections = sections[first_idx:]
        
        # 再次截断：从第一个以“1-”或“1.”开头的实质性编号标题开始（通常正文第一章）
        def _starts_with_chapter_one(name: str) -> bool:
            return re.match(r"^1(?:[.-]\d+){0,3}\.?(?:\s+.+)?$", name.strip()) is not None
        first_ch1_idx = None
        for idx, sec in enumerate(sections):
            if _starts_with_chapter_one(sec.name) and _is_substantive_numbered_heading(sec.name, sec.content):
                first_ch1_idx = idx
                break
        if first_ch1_idx is not None and first_ch1_idx > 0:
            sections = sections[first_ch1_idx:]
        
        return sections


class LLMClient:
    """Client for interacting with local LLM endpoint."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'http://localhost:11434/v1/')
        self.api_key = api_key or os.getenv('OPENAI_API_KEY', 'ollama')
        
        if OPENAI_AVAILABLE:
            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = None
    
    def rewrite_with_llm(self, section_paragraph: str, rule_condition: str, rule_response: str) -> str:
        """Rewrite paragraph to include rule condition and response."""
        instruction = f"""You are rewriting a technical paragraph from an automotive CRD document. Integrate ONLY the rule's condition/event (the IF-part) into the paragraph. Do NOT insert any requirement wording (e.g., 'shall', 'must') and do NOT add the THEN-part.

Task:
- Rewrite the paragraph to naturally incorporate the specified condition/event while preserving the original style and logic.

Constraints:
- Use existing ECU names from the paragraph/context to replace placeholders like 'ECU A' or 'ECU B'.
- Do not introduce new requirement sentences or 'shall/must/should' phrasing.
- Keep changes minimal. Maintain formatting, tone, and technical terms.
- If a similar condition already exists, refine or merge it rather than duplicate.

Original paragraph:
{section_paragraph}

Rule condition (IF-part only): {rule_condition}

Important: Output ONLY the rewritten paragraph, no explanations."""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model='llama3.3:latest',
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer who specializes in automotive system requirements and EARS rules integration."},
                        {"role": "user", "content": instruction}
                    ],
                    stream=False
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"LLM API error: {e}")
                return self._fallback_rewrite(section_paragraph, rule_condition, rule_response)
        else:
            # Use HTTP request to Ollama API directly
            try:
                return self._call_ollama_http(instruction)
            except Exception as e:
                print(f"HTTP LLM API error: {e}")
                return self._fallback_rewrite(section_paragraph, rule_condition, rule_response)
    
    def _call_ollama_http(self, instruction: str) -> str:
        """Call Ollama API using HTTP requests."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        payload = {
            "model": "llama3.3:latest",
            "messages": [
                {"role": "system", "content": "You are an expert technical writer who specializes in automotive system requirements and EARS rules integration."},
                {"role": "user", "content": instruction}
            ],
            "stream": False
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(url, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    
    def _fallback_rewrite(self, section_paragraph: str, rule_condition: str, rule_response: str) -> str:
        """Fallback rewrite using simple text manipulation that obeys constraints.
        - Insert ONLY the condition/event (IF-part)
        - Replace 'ECU A/B/C' with concrete ECU names from context when possible
        - Do NOT add requirement words like 'shall/must/should'
        """
        text = section_paragraph
        # Detect ECU names from context
        ecu_names = []
        try:
            ecu_names = re.findall(r"\b([A-Za-z][A-Za-z \-/]* ECU)\b", section_paragraph, flags=re.IGNORECASE)
        except Exception:
            ecu_names = []
        # Build a safe condition string
        cond = rule_condition or ""
        # Remove requirement words if they accidentally appear in condition
        cond = re.sub(r"\b(shall|must|should|will)\b", "", cond, flags=re.IGNORECASE)
        cond = re.sub(r"\s+", " ", cond).strip()
        # Map ECU A/B/C to detected names if available, otherwise collapse to 'ECU'
        mappings = {}
        if ecu_names:
            uniq = []
            for n in ecu_names:
                if n not in uniq:
                    uniq.append(n)
            if uniq:
                mappings["ECU A"] = uniq[0]
            if len(uniq) >= 2:
                mappings["ECU B"] = uniq[1]
            if len(uniq) >= 3:
                mappings["ECU C"] = uniq[2]
        for placeholder, real in mappings.items():
            cond = re.sub(rf"\b{re.escape(placeholder)}\b", real, cond)
        # Any remaining placeholders -> generic 'ECU'
        cond = re.sub(r"\bECU [A-Z]\b", "ECU", cond)
        # If nothing to insert, return original
        if not cond:
            return text
        # Append condition as an explanatory clause/new line without requirements wording
        if not text.endswith("\n"):
            text = text + "\n"
        return f"{text}{cond}"


class EARSInjector:
    """Main class for EARS rule injection."""
    
    def __init__(self, rules_file: str = "EARSrules.txt", threshold: float = 0.3):
        self.rules_file = Path(rules_file)
        self.threshold = threshold
        self.rules = self._parse_rules()
        self.llm_client = LLMClient()
        self.matches = []
    
    def _parse_rules(self) -> List[EARSRule]:
        """Parse EARS rules from file."""
        if not self.rules_file.exists():
            raise FileNotFoundError(f"Rules file not found: {self.rules_file}")
        
        rules = []
        with open(self.rules_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        rule = EARSRule(line, i)
                        rules.append(rule)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid rule at line {i}: {e}")
        
        return rules
    
    def scan_crd_files(self, crd_dir: str = ".") -> List[CRDFile]:
        """Scan directory for CRD text files."""
        crd_files = []
        crd_path = Path(crd_dir)
        
        # Look for .txt files in the specified directory
        for txt_file in crd_path.glob("*.txt"):
            try:
                crd_file = CRDFile(txt_file, self.llm_client)
                crd_files.append(crd_file)
                print(f"Loaded CRD file: {txt_file.name}")
            except Exception as e:
                print(f"Error loading {txt_file}: {e}")
        
        return crd_files
    
    def find_matches(self, crd_files: List[CRDFile]) -> List[Dict]:
        """Find matches between rules and CRD sections."""
        matches = []
        
        for crd_file in crd_files:
            for section in crd_file.sections:
                # Scan for ECU components and conditions in this section
                ecu_conditions = section.scan_ecu_and_conditions()
                
                for rule in self.rules:
                    # Try to find the best match based on ECU and condition analysis
                    best_match = self._find_best_ecu_match(ecu_conditions, rule, section, crd_file.file_path.name)
                    
                    if best_match:
                        matches.append(best_match)
                    else:
                        # Fallback to original scoring method, but only if section contains ECU context
                        if not re.search(r"\bECU\b|gateway|module|control unit", section.content, re.IGNORECASE):
                            continue
                        score = self._score_section(section, rule)
                        
                        if score >= self.threshold:
                            # Find best paragraph for injection
                            paragraph, paragraph_score, status = section.find_best_paragraph(rule)
                            
                            if paragraph:
                                match = {
                                    'crd_file': crd_file.file_path.name,
                                    'ecu_section': section.name,
                                    'line_span': f"{section.start_line}-{section.end_line}",
                                    'rule_idx': rule.rule_idx,
                                    'match_score': score,
                                    'status': status,
                                    'matched_snippet': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                                    'section': section,
                                    'rule': rule,
                                    'paragraph': paragraph,
                                    'match_type': 'fallback'
                                }
                                matches.append(match)
                    break
        
        return matches
    
    def _find_best_ecu_match(self, ecu_conditions: List[Dict], rule: EARSRule, section: CRDSection, crd_filename: str) -> Optional[Dict]:
        """Find the best ECU-based match for a rule."""
        best_match = None
        best_score = 0.0
        
        for ecu_cond in ecu_conditions:
            # Check if ECU in rule matches ECU in section
            ecu_match_score = self._score_ecu_match(ecu_cond, rule)
            
            # Check if condition in rule matches conditions in section
            condition_match_score = self._score_condition_match(ecu_cond, rule)
            
            # Combined score
            total_score = ecu_match_score * 0.6 + condition_match_score * 0.4
            
            if total_score > best_score and total_score >= self.threshold:
                best_score = total_score
                
                # Find the best paragraph for injection
                paragraph, paragraph_score, status = section.find_best_paragraph(rule)
                
                if paragraph:
                    start_line = ecu_cond.get('ecu_line')
                    if isinstance(start_line, int):
                        end_line_val = start_line + len(ecu_cond.get('context_lines', []))
                        line_span = f"{start_line}-{end_line_val}"
                    else:
                        line_span = section.start_line if hasattr(section, 'start_line') else ''
                        line_span = f"{line_span}-{getattr(section, 'end_line', '')}"
                    best_match = {
                        'crd_file': crd_filename,
                        'ecu_section': section.name,
                        'ecu_context': ecu_cond,
                        'line_span': line_span,
                        'rule_idx': rule.rule_idx,
                        'match_score': total_score,
                        'ecu_match_score': ecu_match_score,
                        'condition_match_score': condition_match_score,
                        'status': status,
                        'matched_snippet': ecu_cond['context_text'][:200] + "..." if len(ecu_cond['context_text']) > 200 else ecu_cond['context_text'],
                        'section': section,
                        'rule': rule,
                        'paragraph': paragraph,
                        'match_type': 'ecu_based'
                    }
        
        return best_match
    
    def _score_ecu_match(self, ecu_cond: Dict, rule: EARSRule) -> float:
        """Score how well ECU in rule matches ECU in section."""
        score = 0.0
        
        # Extract ECU names from rule
        rule_ecu_patterns = [
            r'ECU\s+([A-Z])',
            r'([A-Z]+\s+ECU)',
            r'ECGW'
        ]
        
        rule_ecus = []
        for pattern in rule_ecu_patterns:
            matches = re.findall(pattern, rule.original_text, re.IGNORECASE)
            rule_ecus.extend(matches)
        
        # Check if any ECU from rule matches ECU in section
        for rule_ecu in rule_ecus:
            for ecu_match in ecu_cond['ecu_matches']:
                if rule_ecu.lower() in ecu_match.lower() or ecu_match.lower() in rule_ecu.lower():
                    score += 0.8
                elif any(word in ecu_match.lower() for word in rule_ecu.lower().split()):
                    score += 0.4
        
        return min(score, 1.0)
    
    def _score_condition_match(self, ecu_cond: Dict, rule: EARSRule) -> float:
        """Score how well condition in rule matches conditions in section."""
        score = 0.0
        
        # Check if rule condition keywords appear in section conditions
        rule_condition_lower = rule.condition.lower()
        condition_keywords = [
            'request', 'start', 'stop', 'wait', 'sequence', 'timeout', 'error',
            'communication', 'status', 'ready', 'process', 'step'
        ]
        
        for keyword in condition_keywords:
            if keyword in rule_condition_lower:
                if any(keyword in cond.lower() for cond in ecu_cond['conditions']):
                    score += 0.2
                elif keyword in ecu_cond['context_text'].lower():
                    score += 0.1
        
        return min(score, 1.0)
    
    def _score_section(self, section: CRDSection, rule: EARSRule) -> float:
        """Score a section against a rule."""
        score = 0.0
        
        # Check for regex hits
        condition_matches = re.findall(rule.normalized_condition, section.content, re.IGNORECASE)
        response_matches = re.findall(rule.normalized_response, section.content, re.IGNORECASE)
        
        score += len(condition_matches) * 0.3
        score += len(response_matches) * 0.3
        
        # Check fuzzy similarity
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.partial_ratio(section.content.lower(), rule.condition.lower()) / 100.0
        else:
            similarity = difflib.SequenceMatcher(None, section.content.lower(), rule.condition.lower()).ratio()
        
        score += similarity * 0.4
        
        return min(score, 1.0)
    
    def inject_rules(self, matches: List[Dict]) -> List[Dict]:
        """Inject rules into matched sections. Limit to top 5 injections per run."""
        injected_results = []
        
        # Deduplicate by (file, line_span): keep highest match_score
        dedup: Dict[Tuple[str, str], Dict] = {}
        for m in matches:
            key = (m.get('crd_file', ''), m.get('line_span', ''))
            if key not in dedup or m.get('match_score', 0.0) > dedup[key].get('match_score', 0.0):
                dedup[key] = m
        matches = list(dedup.values())
        
        # Prioritize by match_score descending; only count real injections
        sorted_matches = sorted(matches, key=lambda m: m.get('match_score', 0.0), reverse=True)
        max_injections = 1
        injections_done = 0
        
        for match in sorted_matches:
            if match['status'] == 'inject':
                if injections_done >= max_injections:
                    # Respect limit: do not inject beyond the cap
                    match['status'] = 'limit_skipped'
                    match['injected_paragraph'] = match['paragraph']
                    injected_results.append(match)
                    continue
                try:
                    # Rewrite paragraph with LLM
                    rewritten = self.llm_client.rewrite_with_llm(
                        match['paragraph'],
                        match['rule'].condition,
                        match['rule'].response
                    )
                    # Normalize terminology and EARS style
                    rewritten = rewritten.replace('ventilated sheet ECU', 'ventilated seat ECU')
                    if ' shall ' not in rewritten and 'shall ' in match['rule'].response.lower():
                        # enforce requirement tone lightly by appending response if missing
                        rewritten = f"{rewritten}\n{match['rule'].response}"
                    match['injected_paragraph'] = rewritten
                    injected_results.append(match)
                    injections_done += 1
                except Exception as e:
                    print(f"Error injecting rule {match['rule_idx']} into {match['crd_file']}: {e}")
                    match['injected_paragraph'] = match['paragraph']  # Keep original
                    injected_results.append(match)
            else:
                # Rule already exists or non-inject
                match['injected_paragraph'] = match.get('paragraph', '')
                injected_results.append(match)
        
        return injected_results
    
    def generate_outputs(self, matches: List[Dict], output_dir: str = ".", apply_patches: bool = False):
        """Generate output files and patches."""
        output_path = Path(output_dir)
        
        # Create output directories
        patches_dir = output_path / "patches"
        patched_dir = output_path / "_patched"
        
        patches_dir.mkdir(exist_ok=True)
        if apply_patches:
            patched_dir.mkdir(exist_ok=True)
        
        # Generate matches.csv
        self._write_matches_csv(matches, output_path / "matches.csv")
        
        # Generate injected.md
        self._write_injected_md(matches, output_path / "injected.md")
        
        # Generate patches
        self._generate_patches(matches, patches_dir, patched_dir, apply_patches)
    
    def _write_matches_csv(self, matches: List[Dict], output_file: Path):
        """Write matches to CSV file."""
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'crd_file', 'ecu_section', 'line_span', 'rule_idx', 
                'match_score', 'status', 'match_type', 'ecu_match_score', 
                'condition_match_score', 'matched_snippet'
            ])
            writer.writeheader()
            
            for match in matches:
                row = {
                    'crd_file': match.get('crd_file', ''),
                    'ecu_section': match.get('ecu_section', ''),
                    'line_span': match.get('line_span', ''),
                    'rule_idx': match.get('rule_idx', ''),
                    'match_score': f"{match.get('match_score', 0):.3f}",
                    'status': match.get('status', ''),
                    'match_type': match.get('match_type', ''),
                    'ecu_match_score': f"{match.get('ecu_match_score', 0):.3f}" if 'ecu_match_score' in match else '',
                    'condition_match_score': f"{match.get('condition_match_score', 0):.3f}" if 'condition_match_score' in match else '',
                    'matched_snippet': match.get('matched_snippet', '')
                }
                writer.writerow(row)
        
        print(f"Matches written to: {output_file}")
    
    def _write_injected_md(self, matches: List[Dict], output_file: Path):
        """Write injection results to Markdown file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# EARS Rule Injection Results\n\n")
            f.write("## Summary\n\n")
            f.write(f"Total matches found: {len(matches)}\n")
            f.write(f"Rules injected: {len([m for m in matches if m['status'] == 'inject'])}\n")
            f.write(f"Rules already exist: {len([m for m in matches if m['status'] == 'exists'])}\n")
            f.write(f"Rules skipped due to limit: {len([m for m in matches if m['status'] == 'limit_skipped'])}\n\n")
            
            # Group by file
            files = {}
            for match in matches:
                if match['crd_file'] not in files:
                    files[match['crd_file']] = []
                files[match['crd_file']].append(match)
            
            for filename, file_matches in files.items():
                f.write(f"## {filename}\n\n")
                
                for match in file_matches:
                    f.write(f"### Rule {match['rule_idx']}: {match['ecu_section']}\n\n")
                    f.write(f"**Status:** {match['status']}\n\n")
                    f.write(f"**Match Score:** {match['match_score']:.3f}\n\n")
                    
                    if 'match_type' in match:
                        f.write(f"**Match Type:** {match['match_type']}\n\n")
                    
                    if 'ecu_match_score' in match and 'condition_match_score' in match:
                        f.write(f"**ECU Match Score:** {match['ecu_match_score']:.3f}\n\n")
                        f.write(f"**Condition Match Score:** {match['condition_match_score']:.3f}\n\n")
                    
                    f.write(f"**Location:** Lines {match['line_span']}\n\n")
                    
                    if match['status'] == 'inject':
                        f.write("**Injected Content:**\n\n")
                        f.write(f"{match['injected_paragraph']}\n\n")
                        
                        f.write("**Original Context:**\n")
                        f.write(f"{match['matched_snippet']}\n\n")
                        
                        if 'ecu_context' in match:
                            f.write("**ECU Context:**\n")
                            ecu_ctx = match['ecu_context']
                            f.write(f"- ECU Line: {ecu_ctx['ecu_line']}\n")
                            f.write(f"- ECU Text: {ecu_ctx['ecu_text']}\n")
                            if ecu_ctx['conditions']:
                                f.write(f"- Conditions Found: {', '.join(ecu_ctx['conditions'])}\n")
                            f.write("\n")
                    else:
                        f.write("**Existing Content:**\n\n")
                        f.write(f"{match['paragraph']}\n\n")
                    
                    f.write("---\n\n")
        
        print(f"Injection results written to: {output_file}")
    
    def _generate_patches(self, matches: List[Dict], patches_dir: Path, patched_dir: Path, apply_patches: bool):
        """Generate patch files and optionally apply them."""
        # Group by file
        files = {}
        for match in matches:
            if match['crd_file'] not in files:
                files[match['crd_file']] = []
            files[match['crd_file']].append(match)
        
        def _fuzzy_find(hay: str, needle: str) -> int:
            # Try approximate search by sliding window on sentence boundaries
            try:
                import difflib
                sentences = re.split(r'(\.?\s+)', hay)
                hay_joined = hay
                # Quick fallback using get_close_matches on windows
                candidates = {}
                for i in range(0, len(hay), max(20, len(needle)//4)):
                    window = hay[i:i+len(needle)*2]
                    if not window:
                        continue
                    ratio = difflib.SequenceMatcher(None, window, needle).ratio()
                    candidates[i] = ratio
                if candidates:
                    best_i = max(candidates, key=candidates.get)
                    if candidates[best_i] >= 0.6:
                        return best_i
            except Exception:
                pass
            return -1
        
        for filename, file_matches in files.items():
            # Read original file - look in CRD directory if it exists
            original_file = Path(filename)
            if not original_file.exists():
                # Try to find it in common locations
                possible_paths = [
                    Path(filename),
                    Path("CRD") / filename,
                    Path(".") / filename
                ]
                original_file = None
                for path in possible_paths:
                    if path.exists():
                        original_file = path
                        break
                
                if not original_file:
                    print(f"Warning: Original file not found: {filename}")
                    continue
            
            with open(original_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Apply patches
            patched_content = original_content
            patch_lines = []
            
            for match in file_matches:
                if match['status'] == 'inject':
                    # Create patch
                    original_para = match['paragraph']
                    new_para = match['injected_paragraph']
                    
                    if original_para != new_para:
                        # Find paragraph in file content
                        para_start = patched_content.find(original_para)
                        if para_start == -1:
                            para_start = _fuzzy_find(patched_content, original_para)
                        if para_start != -1:
                            # Apply patch
                            patched_content = (
                                patched_content[:para_start] + 
                                new_para + 
                                patched_content[para_start + len(original_para):]
                            )
                            # Add to patch lines
                            patch_lines.append(f"@@ -{match['line_span']} @@")
                            patch_lines.append(f"-{original_para}")
                            patch_lines.append(f"+{new_para}")
                            patch_lines.append("")
                        else:
                            print(f"Warning: Fuzzy replace failed for: {filename} Rule {match['rule_idx']}")
            
            # Write patch file
            if patch_lines:
                patch_file = patches_dir / f"{filename}.patch"
                with open(patch_file, 'w', encoding='utf-8') as f:
                    f.write(f"--- {filename}\n")
                    f.write(f"+++ {filename}\n")
                    f.write("".join(patch_lines))
                
                print(f"Patch written to: {patch_file}")
            
            # Write patched file if requested
            if apply_patches:
                patched_file = patched_dir / filename
                with open(patched_file, 'w', encoding='utf-8') as f:
                    f.write(patched_content)
                
                print(f"Patched file written to: {patched_file}")
    
    def run(self, crd_dir: str = ".", output_dir: str = ".", apply_patches: bool = False):
        """Run the complete EARS injection process."""
        print("=" * 60)
        print("SECURITY NOTICE: Processing CONFIDENTIAL CRD documents")
        print("All processing is done locally. No data leaves your system.")
        print("=" * 60)
        print()
        print("Starting EARS rule injection...")
        print(f"Rules file: {self.rules_file}")
        print(f"CRD directory: {crd_dir}")
        print(f"Threshold: {self.threshold}")
        print()
        
        # Scan CRD files
        crd_files = self.scan_crd_files(crd_dir)
        if not crd_files:
            print("No CRD files found.")
            return
        
        print(f"Found {len(crd_files)} CRD files")
        print()
        
        # Find matches
        print("Finding matches between rules and CRD sections...")
        # 打印各 section 标题与行号
        total_sections = sum(len(cf.sections) for cf in crd_files)
        print(f"Total sections: {total_sections}")
        for cf in crd_files:
            print(f"Sections in {cf.file_path.name}:")
            for sec in cf.sections:
                print(f"- [{sec.start_line}-{sec.end_line}] {sec.name}")
        
        matches = self.find_matches(crd_files)
        print(f"Found {len(matches)} matches")
        print()
        
        # Inject rules
        print("Injecting rules using LLM...")
        injected_matches = self.inject_rules(matches)
        print(f"Processed {len(injected_matches)} matches")
        # Concise test output: only show first injected item (if any)
        first_injected = next((m for m in injected_matches if m.get('status') in ('inject','limit_skipped')), None)
        if first_injected:
            print("\n=== Test Preview (single issue) ===")
            print(f"File: {first_injected.get('crd_file')}")
            print(f"Section: {first_injected.get('ecu_section')}  Lines: {first_injected.get('line_span')}")
            print(f"Rule {first_injected['rule_idx']}: {first_injected['rule'].original_text}")
            print("--- Original Paragraph ---")
            print(first_injected.get('paragraph','')[:2000])
            print("--- Modified Paragraph ---")
            print(first_injected.get('injected_paragraph','')[:2000])
            print("===============================\n")
        
        # Generate outputs
        print("Generating output files...")
        self.generate_outputs(injected_matches, output_dir, apply_patches)
        print()
        
        print("EARS injection complete!")
        print()
        print("=" * 60)
        print("SECURITY REMINDER:")
        print("- Output files contain document fragments")
        print("- Consider deleting output files after use if confidentiality is critical")
        print("- All processing was done locally")
        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Inject EARS rules into CRD files using local LLM")
    parser.add_argument("--rules", default="EARSrules.txt", help="EARS rules file (default: EARSrules.txt)")
    parser.add_argument("--crd-dir", default=".", help="Directory containing CRD files (default: current)")
    parser.add_argument("--output-dir", default=".", help="Output directory (default: current)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Match threshold (default: 0.3)")
    parser.add_argument("--apply", action="store_true", help="Apply patches to create patched files")
    parser.add_argument("--md", action="store_true", help="Generate injected.md output")
    parser.add_argument("--secure-cleanup", action="store_true", help="Securely delete output files after processing (for confidential documents)")
    parser.add_argument("--rule-idx", type=int, help="Only use the EARS rule with this index (1-based)")
    parser.add_argument("--section-filter", help="Only consider sections whose title matches this regex (e.g., '^3-1')")

    args = parser.parse_args()

    try:
        injector = EARSInjector(args.rules, args.threshold)
        # Optional: restrict rules to a specific rule index
        if args.rule_idx:
            injector.rules = [r for r in injector.rules if r.rule_idx == args.rule_idx]
            if not injector.rules:
                print(f"No rule found with index {args.rule_idx}")
                return 1
        # Run to build sections, then optionally filter sections by title regex
        # We'll intercept inside run by pre-filtering sections after scan
        # So we temporarily wrap run to inject a section filter
        def _run_with_section_filter():
            print("============================================================")
            print("SECURITY NOTICE: Processing CONFIDENTIAL CRD documents")
            print("All processing is done locally. No data leaves your system.")
            print("============================================================")
            print()
            print("Starting EARS rule injection...")
            print(f"Rules file: {injector.rules_file}")
            print(f"CRD directory: {args.crd_dir}")
            print(f"Threshold: {injector.threshold}")
            print()
            crd_files = injector.scan_crd_files(args.crd_dir)
            if not crd_files:
                print("No CRD files found.")
                return
            print(f"Found {len(crd_files)} CRD files")
            print()
            # Section filter
            if args.section_filter:
                import re as _re
                pat = _re.compile(args.section_filter, _re.IGNORECASE)
                for cf in crd_files:
                    original_count = len(cf.sections)
                    cf.sections = [s for s in cf.sections if pat.search(s.name)]
                    print(f"Section filter '{args.section_filter}' applied to {cf.file_path.name}: {original_count} -> {len(cf.sections)} sections")
            print("Finding matches between rules and CRD sections...")
            total_sections = sum(len(cf.sections) for cf in crd_files)
            print(f"Total sections: {total_sections}")
            for cf in crd_files:
                print(f"Sections in {cf.file_path.name}:")
                for sec in cf.sections:
                    print(f"- [{sec.start_line}-{sec.end_line}] {sec.name}")
            matches = injector.find_matches(crd_files)
            print(f"Found {len(matches)} matches")
            print()
            print("Injecting rules using LLM...")
            injected_matches = injector.inject_rules(matches)
            print(f"Processed {len(injected_matches)} matches")
            first_injected = next((m for m in injected_matches if m.get('status') in ('inject','limit_skipped')), None)
            if first_injected:
                print("\n=== Test Preview (single issue) ===")
                print(f"File: {first_injected.get('crd_file')}")
                print(f"Section: {first_injected.get('ecu_section')}  Lines: {first_injected.get('line_span')}")
                print(f"Rule {first_injected['rule_idx']}: {first_injected['rule'].original_text}")
                print("--- Original Paragraph ---")
                print(first_injected.get('paragraph','')[:2000])
                print("--- Modified Paragraph ---")
                print(first_injected.get('injected_paragraph','')[:2000])
                print("===============================\n")
            print("Generating output files...")
            injector.generate_outputs(injected_matches, args.output_dir, args.apply)
            print()
            print("EARS injection complete!")
            print()
            print("============================================================")
            print("SECURITY REMINDER:")
            print("- Output files contain document fragments")
            print("- Consider deleting output files after use if confidentiality is critical")
            print("- All processing was done locally")
            print("============================================================")
        _run_with_section_filter()

        # Secure cleanup if requested
        if args.secure_cleanup:
            print("\nPerforming secure cleanup of output files...")
            cleanup_files = [
                "matches.csv",
                "injected.md",
                "patches",
                "_patched"
            ]
            for file_path in cleanup_files:
                if os.path.exists(file_path):
                    if os.path.isdir(file_path):
                        import shutil
                        shutil.rmtree(file_path)
                    else:
                        os.remove(file_path)
                    print(f"Deleted: {file_path}")
            print("Secure cleanup completed.")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
