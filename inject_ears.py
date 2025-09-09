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
        
        # Check for table/figure headers
        if any(line.strip().startswith(('Table', 'Figure', 'Fig.', 'Table ', 'Figure ')) for line in lines[:2]):
            return True
        
        # Check for table separators (|)
        if any('|' in line for line in lines):
            return True
        
        # Check for numbered lists or TOC-like content
        if len(lines) >= 6 and sum(1 for ln in lines if re.search(r"\b\d+\b", ln)) >= 4:
            return True
        
        # Check for very short lines (likely formatting)
        if all(len(ln.strip()) <= 6 for ln in lines if ln.strip()):
            return True
        
        # Check for directory/TOC patterns
        if re.search(r'\.{3,}|\s{10,}', paragraph):  # Dots or excessive spaces (TOC markers)
            return True
        
        # Check for structured data patterns
        if re.search(r'^\s*\d+[\.\-]\d+', paragraph, re.MULTILINE):  # Numbered sections
            return True
        
        # Check for table-like alignment
        if len([line for line in lines if re.search(r'\s{5,}', line)]) >= 3:
            return True
        
        return False
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
        
        # Temporal/conditional cues
        temporal_cues = ['if', 'when', 'while', 'during', 'shall']
        score += sum(1 for cue in temporal_cues if cue.lower() in paragraph.lower()) * 0.1
        
        # Similarity to rule condition
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.partial_ratio(paragraph.lower(), rule.condition.lower()) / 100.0
        else:
            similarity = difflib.SequenceMatcher(None, paragraph.lower(), rule.condition.lower()).ratio()
        score += similarity * 0.5
        
        # Technical terms
        technical_terms = ['ecu', 'signal', 'communication', 'control', 'status', 'request']
        score += sum(1 for term in technical_terms if term.lower() in paragraph.lower()) * 0.05
        
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
        
        # Build LLM prompt for ECU and condition identification
        prompt = f"""You are an automotive systems expert. Analyze the following technical document section to identify ECU components and related conditions/events.

Please carefully analyze the document content and find:
1. All mentioned ECU components (such as ECGW, ventilated seat ECU, etc.)
2. Related conditions/events (identify events that may not have obvious keywords like 'if/shall')

IMPORTANT INSTRUCTIONS:
- ONLY analyze TEXT content, NOT tables, lists, or directory structures
- IGNORE table headers, table data, numbered lists, bullet points, and TOC (Table of Contents)
- FOCUS on descriptive paragraphs and technical explanations
- SKIP any content that appears to be structured data or formatting

Document section content:
{self.content[:3000]}

IMPORTANT: Do not show any thinking process or reasoning. Return ONLY the final JSON result in this exact format:
{{
    "ecu_components": [
        {{
            "ecu_name": "ECU name",
            "ecu_line": "Position description in document, page and line number",
            "context": "ECU context information",
            "conditions": ["related condition 1", "related condition 2"]
        }}
    ]
}}

If no ECU found in text content, return only: {{"ecu_components": []}}"""
        
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
            # Clean response and extract JSON part
            clean_response = llm_response.strip()
            
            # Remove any thinking process markers that Qwen might add
            thinking_patterns = [
                r'<thinking>.*?</thinking>',
                r'<think>.*?</think>'
            ]
            
            for pattern in thinking_patterns:
                clean_response = re.sub(pattern, '', clean_response, flags=re.DOTALL)
            
            # Extract JSON part
            json_start = clean_response.find('{')
            json_end = clean_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_text = clean_response[json_start:json_end]
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
            # Skip sections with too many dots (TOC markers)
            if re.search(r"\.{5,}", sec.content):
                return False
            return True
        
        # 进一步：从第一个"实质性编号标题"开始截断，丢弃其之前的所有分段
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
- The condition should be integrated as a natural part of the technical description, not as a separate requirement.

CRITICAL ECU MAPPING REQUIREMENTS:
- ECU A, ECU B, ECU C, etc. in the rule are PLACEHOLDERS/CODES, NOT actual ECU names
- You MUST identify which real ECUs from the paragraph/context correspond to ECU A, ECU B, etc.
- Replace ECU A, ECU B, ECU C with the ACTUAL ECU names found in the paragraph/context
- Keep the original ECU names exactly as they appear in the document
- Do NOT use placeholder names like "ECU A" or "ECU B" in the final output
- If the paragraph mentions "ventilated seat ECU", use "ventilated seat ECU" (not "ECU A")
- If the paragraph mentions "steering heater ECU", use "steering heater ECU" (not "ECU B")
- If the paragraph mentions "A/C ECU", use "A/C ECU" (not "ECU C")
- Map based on the context and role of each ECU in the paragraph

Constraints:
- Do not introduce new requirement sentences or 'shall/must/should' phrasing.
- Keep changes minimal. Maintain formatting, tone, and technical terms.
- If a similar condition already exists, refine or merge it rather than duplicate.
- ONLY work with descriptive text content, NOT tables, lists, or structured data.
- The condition should read as a natural part of the technical description, not as a separate rule.

IMPORTANT: This paragraph should be descriptive text content suitable for rule injection. If the content appears to be a table, list, or structured data, do not modify it.

Original paragraph:
{section_paragraph}

Rule condition (IF-part only): {rule_condition}

IMPORTANT: Do not show any thinking process, reasoning, or explanations. Output ONLY the final rewritten paragraph with actual ECU names. Do NOT add any response or requirement statements."""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model='qwen3:32b',
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer who specializes in automotive system requirements and EARS rules integration."},
                        {"role": "user", "content": instruction}
                    ],
                    stream=False
                )
                result = response.choices[0].message.content.strip()
                
                # Clean any thinking process from Qwen output
                thinking_patterns = [
                    r'<thinking>.*?</thinking>',
                    r'<think>.*?</think>'
                ]
                
                for pattern in thinking_patterns:
                    result = re.sub(pattern, '', result, flags=re.DOTALL)
                
                return result.strip()
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
            "model": "qwen3:32b",
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
        content = result["choices"][0]["message"]["content"].strip()
        
        # Clean any thinking process from Qwen output
        thinking_patterns = [
            r'<thinking>.*?</thinking>',
            r'<think>.*?</think>'
        ]
        
        for pattern in thinking_patterns:
            content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        return content.strip()
    
    def _fallback_rewrite(self, section_paragraph: str, rule_condition: str, rule_response: str) -> str:
        """Simple fallback rewrite when LLM is unavailable."""
        text = section_paragraph
        
        # Extract ECU names and build safe condition
        ecu_names = re.findall(r"\b([A-Za-z][A-Za-z \-/]* ECU)\b", text, re.IGNORECASE)
        cond = re.sub(r"\b(shall|must|should|will)\b", "", rule_condition or "", re.IGNORECASE).strip()
        
        # Map ECU placeholders to real names - be more systematic
        if ecu_names:
            # Create a mapping from placeholder to actual ECU names
            ecu_mapping = {}
            for i, placeholder in enumerate(["ECU A", "ECU B", "ECU C", "ECU D"]):
                if i < len(ecu_names):
                    ecu_mapping[placeholder] = ecu_names[i]
            
            # Replace placeholders with actual ECU names
            for placeholder, actual_name in ecu_mapping.items():
                cond = cond.replace(placeholder, actual_name)
            
            # Clean up any remaining generic ECU references
            cond = re.sub(r"\bECU [A-Z]\b", "ECU", cond)
        
        # Append condition if valid
        if cond:
            return f"{text}\n{cond}" if not text.endswith("\n") else f"{text}{cond}"
        return text


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
                        'matched_snippet': paragraph[:200] + "..." if len(paragraph) > 200 else paragraph,
                        'section': section,
                        'rule': rule,
                        'paragraph': paragraph,
                        'match_type': 'ecu_based'
                    }
        
        return best_match
    
    def _score_ecu_match(self, ecu_cond: Dict, rule: EARSRule) -> float:
        """Score how well ECU in rule matches ECU in section."""
        score = 0.0
        
        # Extract ECU information from rule
        rule_ecu_info = self._analyze_rule_ecu_pattern(rule)
        
        # Extract ECU information from section
        section_ecu_info = self._analyze_section_ecu_pattern(ecu_cond)
        
        # Score based on ECU count match
        if rule_ecu_info['ecu_count'] == section_ecu_info['ecu_count']:
            score += 0.3
        elif abs(rule_ecu_info['ecu_count'] - section_ecu_info['ecu_count']) == 1:
            score += 0.2
        elif abs(rule_ecu_info['ecu_count'] - section_ecu_info['ecu_count']) == 2:
            score += 0.1
        
        # Score based on ECU type match
        for rule_ecu in rule_ecu_info['ecu_types']:
            for section_ecu in section_ecu_info['ecu_types']:
                if rule_ecu.lower() in section_ecu.lower() or section_ecu.lower() in rule_ecu.lower():
                    score += 0.4
                elif any(word in section_ecu.lower() for word in rule_ecu.lower().split()):
                    score += 0.2
        
        # Score based on interaction pattern match
        pattern_score = self._score_interaction_pattern_match(rule_ecu_info, section_ecu_info)
        score += pattern_score * 0.3
        
        return min(score, 1.0)
    
    def _analyze_rule_ecu_pattern(self, rule: EARSRule) -> Dict:
        """Analyze ECU pattern in EARS rule."""
        rule_text = rule.original_text
        
        # Extract ECU references
        ecu_patterns = [
            r'ECU\s+([A-Z])',
            r'([A-Z]+\s+ECU)',
            r'ECGW',
            r'Server'
        ]
        
        ecus_found = set()
        for pattern in ecu_patterns:
            matches = re.findall(pattern, rule_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    ecus_found.add(' '.join(match))
                else:
                    ecus_found.add(match)
        
        # Analyze interaction patterns
        interaction_patterns = {
            'request_response': len(re.findall(r'send.*request|receive.*response', rule_text, re.IGNORECASE)),
            'forward': len(re.findall(r'forward|transmit', rule_text, re.IGNORECASE)),
            'wait': len(re.findall(r'wait|delay', rule_text, re.IGNORECASE)),
            'sequence': len(re.findall(r'sequence|step|before.*after', rule_text, re.IGNORECASE)),
            'timeout': len(re.findall(r'timeout|time.*limit', rule_text, re.IGNORECASE))
        }
        
        return {
            'ecu_count': len(ecus_found),
            'ecu_types': list(ecus_found),
            'interaction_patterns': interaction_patterns
        }
    
    def _analyze_section_ecu_pattern(self, ecu_cond: Dict) -> Dict:
        """Analyze ECU pattern in CRD section."""
        context_text = ecu_cond['context_text']
        
        # Extract ECU references
        ecu_patterns = [
            r'ECU\s+([A-Z])',
            r'([A-Z]+\s+ECU)',
            r'ECGW',
            r'Server'
        ]
        
        ecus_found = set()
        for pattern in ecu_patterns:
            matches = re.findall(pattern, context_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    ecus_found.add(' '.join(match))
                else:
                    ecus_found.add(match)
        
        # Analyze interaction patterns
        interaction_patterns = {
            'request_response': len(re.findall(r'send.*request|receive.*response', context_text, re.IGNORECASE)),
            'forward': len(re.findall(r'forward|transmit', context_text, re.IGNORECASE)),
            'wait': len(re.findall(r'wait|delay', context_text, re.IGNORECASE)),
            'sequence': len(re.findall(r'sequence|step|before.*after', context_text, re.IGNORECASE)),
            'timeout': len(re.findall(r'timeout|time.*limit', context_text, re.IGNORECASE))
        }
        
        return {
            'ecu_count': len(ecus_found),
            'ecu_types': list(ecus_found),
            'interaction_patterns': interaction_patterns
        }
    
    def _score_interaction_pattern_match(self, rule_info: Dict, section_info: Dict) -> float:
        """Score how well interaction patterns match between rule and section."""
        score = 0.0
        
        rule_patterns = rule_info['interaction_patterns']
        section_patterns = section_info['interaction_patterns']
        
        for pattern_type in rule_patterns:
            rule_count = rule_patterns[pattern_type]
            section_count = section_patterns[pattern_type]
            
            if rule_count > 0 and section_count > 0:
                # Both have this pattern
                score += 0.3
            elif rule_count > 0 and section_count == 0:
                # Rule has pattern but section doesn't
                score += 0.1
        
        return min(score, 1.0)
    
    def _score_condition_match(self, ecu_cond: Dict, rule: EARSRule) -> float:
        """Score how well condition in rule matches conditions in section."""
        score = 0.0
        
        # Analyze condition patterns in rule
        rule_condition_info = self._analyze_rule_condition_pattern(rule)
        
        # Analyze condition patterns in section
        section_condition_info = self._analyze_section_condition_pattern(ecu_cond)
        
        # Score based on condition type match
        for condition_type in rule_condition_info['condition_types']:
            if condition_type in section_condition_info['condition_types']:
                score += 0.3
        
        # Score based on timing/sequence keywords
        timing_keywords = ['wait', 'timeout', 'delay', 'before', 'after', 'sequence', 'step']
        for keyword in timing_keywords:
            if keyword in rule.condition.lower():
                if keyword in ecu_cond['context_text'].lower():
                    score += 0.2
        
        # Score based on action keywords
        action_keywords = ['send', 'receive', 'request', 'response', 'start', 'stop', 'forward']
        for keyword in action_keywords:
            if keyword in rule.condition.lower():
                if keyword in ecu_cond['context_text'].lower():
                    score += 0.15
        
        # Score based on state keywords
        state_keywords = ['ready', 'status', 'error', 'fail', 'success', 'complete']
        for keyword in state_keywords:
            if keyword in rule.condition.lower():
                if keyword in ecu_cond['context_text'].lower():
                    score += 0.1
        
        return min(score, 1.0)
    
    def _analyze_rule_condition_pattern(self, rule: EARSRule) -> Dict:
        """Analyze condition pattern in EARS rule."""
        condition_text = rule.condition.lower()
        
        condition_types = []
        
        # Identify condition types
        if 'wait' in condition_text or 'timeout' in condition_text:
            condition_types.append('timing')
        if 'sequence' in condition_text or 'step' in condition_text or 'before' in condition_text:
            condition_types.append('sequence')
        if 'send' in condition_text or 'receive' in condition_text:
            condition_types.append('communication')
        if 'start' in condition_text or 'stop' in condition_text:
            condition_types.append('process_control')
        if 'error' in condition_text or 'fail' in condition_text:
            condition_types.append('error_handling')
        if 'ready' in condition_text or 'status' in condition_text:
            condition_types.append('state_check')
        
        return {
            'condition_types': condition_types
        }
    
    def _analyze_section_condition_pattern(self, ecu_cond: Dict) -> Dict:
        """Analyze condition pattern in CRD section."""
        context_text = ecu_cond['context_text'].lower()
        
        condition_types = []
        
        # Identify condition types
        if 'wait' in context_text or 'timeout' in context_text:
            condition_types.append('timing')
        if 'sequence' in context_text or 'step' in context_text or 'before' in context_text:
            condition_types.append('sequence')
        if 'send' in context_text or 'receive' in context_text:
            condition_types.append('communication')
        if 'start' in context_text or 'stop' in context_text:
            condition_types.append('process_control')
        if 'error' in context_text or 'fail' in context_text:
            condition_types.append('error_handling')
        if 'ready' in context_text or 'status' in context_text:
            condition_types.append('state_check')
        
        return {
            'condition_types': condition_types
        }
    
    def _score_section(self, section: CRDSection, rule: EARSRule) -> float:
        """Score a section against a rule."""
        # Regex matches + fuzzy similarity
        condition_matches = len(re.findall(rule.normalized_condition, section.content, re.IGNORECASE))
        response_matches = len(re.findall(rule.normalized_response, section.content, re.IGNORECASE))
        
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.partial_ratio(section.content.lower(), rule.condition.lower()) / 100.0
        else:
            similarity = difflib.SequenceMatcher(None, section.content.lower(), rule.condition.lower()).ratio()
        
        return min(condition_matches * 0.3 + response_matches * 0.3 + similarity * 0.4, 1.0)
    
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
        max_injections = 999  # Remove limit for testing
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
    
    def generate_outputs(self, matches: List[Dict], output_dir: str = "output", apply_patches: bool = False):
        """Generate output files and patches."""
        output_path = Path(output_dir)
        
        # Create output directories
        patches_dir = output_path / "patches"
        patched_dir = output_path / "_patched"
        
        output_path.mkdir(exist_ok=True)
        patches_dir.mkdir(exist_ok=True)
        if apply_patches:
            patched_dir.mkdir(exist_ok=True)
        
        
        # Generate injected.md
        self._write_injected_md(matches, output_path / "injected.md")
        
        # Generate patches
        self._generate_patches(matches, patches_dir, patched_dir, apply_patches)
    
    
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
    
    def run(self, crd_dir: str = ".", output_dir: str = "output", apply_patches: bool = False):
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
    parser.add_argument("--crd-dir", default="../CRD", help="Directory containing CRD files (default: ../CRD)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
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
                "output/injected.md",
                "output/patches",
                "output/_patched"
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
