#!/usr/bin/env python3
"""
EARS Rule Injector - SECURITY NOTICE

This script processes CONFIDENTIAL CRD documents. All processing is done locally.
- Files are NOT uploaded to external servers
- LLM processing uses OpenAI API
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

# Import configuration from openai_api.py
try:
    from openai_api import client as OPENAI_CLIENT
    OPENAI_CONFIG_AVAILABLE = True
except ImportError:
    OPENAI_CONFIG_AVAILABLE = False
    print("Warning: openai_api.py not found, using default configuration")


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
            print(f"LLM扫描失败，使用备用方法: {e}")
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
            response = self._call_openai_api(prompt)
            
            # 解析LLM响应并转换为ecu_conditions格式
            ecu_conditions = self._parse_llm_ecu_response(response)
            return ecu_conditions
            
        except Exception as e:
            print(f"LLM scan error: {e}")
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
            try:
                # Fallback to latin-1 if UTF-8 fails
                with open(self.file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Final fallback with error replacement
                with open(self.file_path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
    
    def _split_sections(self) -> List[CRDSection]:
        """Split file into sections based on numbered headings only."""
        sections = []
        
        # Only look for numbered headings: 1.1, 1-1, 1.1.1, 1-1-1, etc.
        heading_patterns = [
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
    """Client for interacting with LLM endpoint using OpenAI API."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        # Use configuration from openai_api.py if available, otherwise use environment variables
        if OPENAI_CONFIG_AVAILABLE:
            self.client = OPENAI_CLIENT  # Use the pre-configured client from openai_api.py
            self.base_url = "https://litellm.eks-ans-se-dev.aws.automotive.cloud/"
            self.api_key = "sk-50qjfnJryCKT_Ku80l1c9w"
        else:
            self.base_url = base_url or os.getenv('OPENAI_BASE_URL', 'https://litellm.eks-ans-se-dev.aws.automotive.cloud/')
            self.api_key = api_key or os.getenv('OPENAI_API_KEY', 'sk-50qjfnJryCKT_Ku80l1c9w')
            
            if OPENAI_AVAILABLE:
                self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            else:
                self.client = None
        
        self.model_name = os.getenv('OPENAI_MODEL', 'gpt-5')
    
    def _quick_ecu_analysis(self, section_content: str, rule_text: str) -> Dict:
        """快速ECU分析 - 简化版本，仅作为LLM失败时的最小备用"""
        # 简化的ECU数量检查
        ecu_count = len(re.findall(r'\bECU\b', section_content, re.IGNORECASE))
        rule_ecu_count = len(re.findall(r'\bECU\s+[ABC]\b', rule_text, re.IGNORECASE))
        
        # 简单的匹配分数计算
        if ecu_count >= 2 and rule_ecu_count >= 2:
            match_score = 0.5  # 基础匹配分数
        elif ecu_count >= 1 and rule_ecu_count >= 1:
            match_score = 0.3
        else:
            match_score = 0.0
        
        return {
            "ecu_count": ecu_count,
            "ecu_names": re.findall(r'\b[A-Z]{2,}\s*ECU\b', section_content, re.IGNORECASE),
            "ecu_roles": ["gateway", "communication"] if "gateway" in section_content.lower() else [],
            "rule_ecu_count": rule_ecu_count,
            "rule_ecu_roles": ["communication", "arbitration"],
            "match_score": match_score,
            "reasoning": f"Quick fallback: ECU count: {ecu_count}, Rule ECU count: {rule_ecu_count}"
        }

#对每个section和rule组合时使用的prompt
    def analyze_ecu_with_llm(self, section_content: str, rule_text: str) -> Dict:
        """Use LLM to analyze entire section for EARS rule pattern matching."""
        print(f"Analyzing section with LLM... (Section: {len(section_content)} chars, Rule: {len(rule_text)} chars)")
        
        prompt = f"""You are an automotive systems expert. Analyze the following CRD section content and EARS rule to determine if the section contains patterns that match the rule's requirements.

CRD Section Content:
{section_content[:3000]}

EARS Rule:
{rule_text}

Please analyze and return ONLY a JSON response in this exact format:
{{
    "section_analysis": {{
        "ecu_count": <number of ECUs mentioned in the section>,
        "ecu_names": ["list of actual ECU names found"],
        "communication_patterns": ["list of communication patterns found"],
        "timing_patterns": ["list of timing/sequence patterns found"],
        "gateway_functions": ["list of gateway/arbitration functions found"]
    }},
    "rule_analysis": {{
        "ecu_count": <number of ECUs mentioned in the rule>,
        "required_patterns": ["list of required patterns from rule"],
        "communication_requirements": ["list of communication requirements"],
        "timing_requirements": ["list of timing/sequence requirements"]
    }},
    "pattern_matching": {{
        "ecu_count_match": <true/false if ECU counts are compatible>,
        "communication_match": <true/false if communication patterns match>,
        "timing_match": <true/false if timing patterns match>,
        "gateway_match": <true/false if gateway functions match>,
        "overall_compatibility": <0.0 to 1.0 overall compatibility score>
    }},
    "match_score": <0.0 to 1.0 final match score>,
    "reasoning": "detailed explanation of why this section matches or doesn't match the rule"
}}

IMPORTANT MATCHING LOGIC:
1. EARS rules describe communication patterns between multiple ECUs (ECU A, ECU B, ECU C)
2. CRD sections describe actual ECU implementations (ECGW, heating ECU, ventilated seat ECU, etc.)
3. A section matches a rule if it contains the SAME COMMUNICATION PATTERNS as described in the rule
4. Focus on PATTERN MATCHING, not just ECU counting:
   - Does the section describe the same communication sequence as the rule?
   - Does the section have the same timing requirements as the rule?
   - Does the section involve the same number of interacting ECUs as the rule?
   - Does the section describe gateway/arbitration functions if the rule requires them?

SCORE CALCULATION:
- 0.9-1.0: Perfect pattern match (same communication sequence, timing, and ECU count)
- 0.7-0.8: Good pattern match (similar communication sequence and timing)
- 0.5-0.6: Partial pattern match (some communication patterns match)
- 0.3-0.4: Weak pattern match (basic communication context but different patterns)
- 0.0-0.2: No pattern match (no relevant communication patterns)

ANALYSIS FOCUS:
1. Identify the communication pattern in the EARS rule (request-response, arbitration, timing sequences)
2. Look for the same pattern in the CRD section
3. Check if the section involves multiple ECUs that can perform this pattern
4. Verify timing and sequence requirements match
5. Confirm gateway/arbitration functions if required by the rule

Return ONLY the JSON, no other text."""

        try:
            import time
            start_time = time.time()
            
            # Add retry mechanism for network issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.client:
                        print(f"Calling LLM API... (attempt {attempt + 1}/{max_retries})")
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert automotive systems analyst."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.0,
                            stream=False,
                            timeout=10  # 10 second timeout for faster processing
                        )
                        result = response.choices[0].message.content.strip()
                    else:
                        print(f"Calling LLM HTTP API... (attempt {attempt + 1}/{max_retries})")
                        result = self._call_openai_api(prompt)
                    
                    end_time = time.time()
                    print(f"LLM call completed in {end_time - start_time:.2f} seconds")
                    
                    # Parse JSON response
                    import json
                    return json.loads(result)
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        raise e
            
        except Exception as e:
            print(f"LLM ECU analysis error after {max_retries} attempts: {e}")
            return {
                "ecu_count": 0,
                "ecu_names": [],
                "ecu_roles": [],
                "rule_ecu_count": 0,
                "rule_ecu_roles": [],
                "match_score": 0.0,
                "reasoning": "Analysis failed"
            }
    
    def rewrite_with_llm(self, section_paragraph: str, rule_condition: str, rule_response: str, section_context: str = "") -> str:
        """Rewrite paragraph to include rule condition and response."""
        instruction = f"""You are rewriting a technical paragraph from an automotive CRD document. Integrate ONLY the rule's condition/event (the IF-part) into the paragraph. Do NOT insert any requirement wording (e.g., 'shall', 'must') and do NOT add the THEN-part.

Task:
- Rewrite the paragraph to naturally incorporate the specified condition/event while preserving the original style and logic.
- The condition should be integrated as a natural part of the technical description, not as a separate requirement.

CRITICAL ECU MAPPING AND TIMING REQUIREMENTS:
- ECU A, ECU B, ECU C, etc. in the rule are PLACEHOLDERS/CODES, NOT actual ECU names
- You MUST identify which real ECUs from the paragraph/context correspond to ECU A, ECU B, etc.
- Replace ECU A, ECU B, ECU C with the ACTUAL ECU names found in the paragraph/context
- Keep the original ECU names exactly as they appear in the document
- Do NOT use placeholder names like "ECU A" or "ECU B" in the final output
- If the paragraph mentions "ventilated seat ECU", use "ventilated seat ECU" (not "ECU A")
- If the paragraph mentions "steering heater ECU", use "steering heater ECU" (not "ECU B")
- If the paragraph mentions "A/C ECU", use "A/C ECU" (not "ECU C")
- Map based on the context and role of each ECU in the paragraph

CRITICAL TIMING REQUIREMENTS:
- If the rule mentions "a certain time", "waiting time", or similar vague timing, you MUST scan the paragraph and context for SPECIFIC timing values
- Look for specific time values like "500ms", "100ms", "1 second", "2 seconds", etc. in the paragraph or surrounding context
- Replace vague timing references with the ACTUAL specific time values found in the document
- If no specific timing is found in the immediate context, look for timing patterns or standard values mentioned elsewhere in the document
- Do NOT use generic phrases like "a certain time" - always use specific time values when available

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

CONTEXT FOR TIMING ANALYSIS:
- Scan the paragraph above for any specific timing values (ms, seconds, etc.)
- Look for timing patterns, delays, or waiting periods mentioned in the text
- Use the most relevant specific timing value found in the context

ADDITIONAL SECTION CONTEXT FOR TIMING:
{section_context}

TIMING ANALYSIS INSTRUCTIONS:
- If the rule mentions "a certain time", "waiting time", or similar vague timing, search the section context above for SPECIFIC timing values
- Look for patterns like "500ms", "100ms", "1 second", "2 seconds", "time chart", "timing", etc.
- If you find specific timing values in the section context, use them instead of vague references
- If no specific timing is found, look for standard automotive timing patterns (e.g., 100ms, 500ms, 1s are common)
- Do NOT use generic phrases like "a certain time" - always try to find or infer specific time values

IMPORTANT: Do not show any thinking process, reasoning, or explanations. Output ONLY the final rewritten paragraph with actual ECU names. Do NOT add any response or requirement statements."""

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": "You are an expert technical writer who specializes in automotive system requirements and EARS rules integration."},
                        {"role": "user", "content": instruction}
                    ],
                    temperature=0.0,
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
            # Use HTTP request to OpenAI API directly
            try:
                return self._call_openai_api(instruction)
            except Exception as e:
                print(f"HTTP LLM API error: {e}")
                return self._fallback_rewrite(section_paragraph, rule_condition, rule_response)
    
    def _call_openai_api(self, instruction: str) -> str:
        """Call OpenAI API using HTTP requests."""
        url = f"{self.base_url.rstrip('/')}/chat/completions"
        
        selected_model = self.model_name
        payload = {
            "model": selected_model,
            "messages": [
                {"role": "system", "content": "You are an expert technical writer who specializes in automotive system requirements and EARS rules integration."},
                {"role": "user", "content": instruction}
            ],
            "stream": False,
            "temperature": 0.0
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

    def analyze_section_for_rules(self, section_content: str, rules: List, threshold: float = 0.3) -> List[Dict]:
        """Analyze a section and find all matching EARS rules in one LLM call."""
        print(f"Analyzing section for all matching rules... (Section: {len(section_content)} chars)")
        print(f"CHECKPOINT LLM-1: Starting LLM analysis with {len(rules)} rules")
        
        # Prepare rules text for LLM
        rules_text = "\n\n".join([f"Rule {rule.rule_idx}: {rule.original_text}" for rule in rules])
        
        prompt = f"""You are an automotive systems expert. Analyze the following CRD section content and find ALL EARS rules that match this section's communication patterns.

CRD Section Content:
{section_content[:3000]}

EARS Rules to analyze:
{rules_text[:4000]}

Please analyze and return ONLY a JSON response in this exact format:
{{
    "matching_rules": [
        {{
            "rule_idx": <rule number>,
            "match_score": <0.0 to 1.0 match score>,
            "reasoning": "brief explanation of why this rule matches"
        }},
        ...
    ]
}}

MATCHING CRITERIA:
1. The section must contain communication patterns that match the rule's requirements
2. The section must involve multiple ECUs if the rule requires multiple ECUs
3. The section must describe similar timing/sequence patterns as the rule
4. The section must have gateway/arbitration functions if the rule requires them

SCORE CALCULATION:
- 0.9-1.0: Perfect pattern match (same communication sequence, timing, and ECU count)
- 0.7-0.8: Good pattern match (similar communication sequence and timing)
- 0.5-0.6: Partial pattern match (some communication patterns match)
- 0.3-0.4: Weak pattern match (basic communication context but different patterns)
- 0.0-0.2: No pattern match (no relevant communication patterns)

Only include rules with match_score >= {threshold}.

Return ONLY the JSON, no other text."""

        try:
            import time
            start_time = time.time()
            print(f"CHECKPOINT LLM-2: About to make API call, timeout=60s")
            
            # Add retry mechanism for network issues
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if self.client:
                        print(f"CHECKPOINT LLM-3: Calling LLM API for section analysis... (attempt {attempt + 1}/{max_retries})")
                        response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=[
                                {"role": "system", "content": "You are an expert automotive systems analyst."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.0,
                            stream=False,
                            timeout=60  # 60 second timeout for complex analysis
                        )
                        result = response.choices[0].message.content.strip()
                        print(f"CHECKPOINT LLM-4: API call successful, got response")
                    else:
                        print(f"CHECKPOINT LLM-3B: Calling LLM HTTP API for section analysis... (attempt {attempt + 1}/{max_retries})")
                        result = self._call_openai_api(prompt)
                        print(f"CHECKPOINT LLM-4B: HTTP API call successful, got response")
                    
                    end_time = time.time()
                    print(f"CHECKPOINT LLM-5: LLM section analysis completed in {end_time - start_time:.2f} seconds")
                    
                    # Parse JSON response
                    import json
                    analysis_result = json.loads(result)
                    
                    # Return matching rules
                    matching_rules = analysis_result.get('matching_rules', [])
                    print(f"Found {len(matching_rules)} matching rules")
                    return matching_rules
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed: {e}, retrying...")
                        time.sleep(2)  # Wait 2 seconds before retry
                        continue
                    else:
                        raise e
            
        except Exception as e:
            print(f"LLM section analysis error after {max_retries} attempts: {e}")
            return []


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
    
    def find_matches_optimized(self, crd_files: List[CRDFile]) -> List[Dict]:
        """Optimized matching: scan each section once, let LLM find matching rules."""
        matches = []
        
        print("Starting optimized section-based scanning...")
        print("CHECKPOINT 1: Starting section analysis")
        
        section_count = 0
        max_sections = 5  # Limit to 5 sections for speed
        
        for crd_file in crd_files:
            print(f"\nProcessing file: {crd_file.file_path.name}")
            
            for section in crd_file.sections:
                # Skip sections without ECU context
                if not re.search(r"\bECU\b|gateway|module|control unit|communication|CAN|LIN|arbitration", section.content, re.IGNORECASE):
                    continue
                
                # Limit number of sections processed
                section_count += 1
                if section_count > max_sections:
                    print(f"Reached limit of {max_sections} sections, stopping analysis")
                    break
                
                print(f"\nAnalyzing section {section_count}/{max_sections}: {section.name}")
                print(f"CHECKPOINT 2: About to call LLM for section {section_count}")
                
                try:
                    # Let LLM analyze the section and find matching rules
                    print(f"CHECKPOINT 3: Calling analyze_section_for_rules...")
                    section_matches = self.llm_client.analyze_section_for_rules(section.content, self.rules, self.threshold)
                    print(f"CHECKPOINT 4: LLM call completed, got {len(section_matches)} matches")
                    
                    if section_matches:
                        print(f"Found {len(section_matches)} matching rules for section {section.name}")
                        
                        for match_data in section_matches:
                            rule_idx = match_data['rule_idx']
                            rule = next((r for r in self.rules if r.rule_idx == rule_idx), None)
                            
                            if rule:
                                # Find best paragraph for injection
                                paragraph, paragraph_score, status = section.find_best_paragraph(rule)
                            
                            if paragraph:
                                match = {
                                    'crd_file': crd_file.file_path.name,
                                    'ecu_section': section.name,
                                    'line_span': f"{section.start_line}-{section.end_line}",
                                    'rule_idx': rule_idx,
                                    'match_score': match_data['match_score'],
                                    'ecu_analysis': match_data,
                                    'status': status,
                                    'matched_snippet': paragraph,
                                    'section': section,
                                    'rule': rule,
                                    'paragraph': paragraph,
                                    'match_type': 'llm_optimized',
                                    'analysis_method': 'llm_optimized'
                                }
                                matches.append(match)
                                print(f"  Rule {rule_idx}: {match_data['match_score']:.3f}")
                            else:
                                print(f"  Rule {rule_idx}: No suitable paragraph found")
                        else:
                            print(f"  Rule {rule_idx}: Rule not found")
                    else:
                        print(f"No matching rules found for section {section.name}")
                        
                except Exception as e:
                    print(f"CHECKPOINT ERROR: Error analyzing section {section.name}: {e}")
                    print(f"Error type: {type(e).__name__}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
                    continue
        
        print(f"\nOptimized scanning complete! Found {len(matches)} matches")
        return matches
    
    def find_matches(self, crd_files: List[CRDFile]) -> List[Dict]:
        """Find matches between rules and CRD sections using LLM-based ECU analysis."""
        matches = []
        llm_call_count = 0
        total_combinations = 0
        
        # Count total combinations for progress tracking (use actual sections, not lines)
        for crd_file in crd_files:
            for section in crd_file.sections:
                if re.search(r"\bECU\b|gateway|module|control unit", section.content, re.IGNORECASE):
                    total_combinations += len(self.rules)
        
        print(f"Total rule-section combinations to analyze: {total_combinations}")
        
        # Limit combinations to prevent excessive LLM calls  
        max_combinations = min(total_combinations, 20)  # Limit to 20 calls max for speed
        
        for crd_file in crd_files:
            for section in crd_file.sections:
                # Skip sections without ECU context
                if not re.search(r"\bECU\b|gateway|module|control unit", section.content, re.IGNORECASE):
                    continue
                
                # Collect all potential matches for this section
                section_matches = []
                
                for rule in self.rules:
                    llm_call_count += 1
                    
                    # Stop if we've reached the limit
                    if llm_call_count > max_combinations:
                        print(f"Reached limit of {max_combinations} LLM calls, stopping analysis")
                        break
                    
                    print(f"Progress: {llm_call_count}/{min(total_combinations, max_combinations)} - Analyzing Rule {rule.rule_idx} in {section.name}")
                    
                    try:
                        # Use LLM as primary analysis method
                        ecu_analysis = self.llm_client.analyze_ecu_with_llm(section.content, rule.original_text)
                        
                        if ecu_analysis['match_score'] >= self.threshold:
                            print(f"LLM Match found! Score: {ecu_analysis['match_score']:.3f}")
                            # Find best paragraph for injection
                            paragraph, paragraph_score, status = section.find_best_paragraph(rule)
                            
                            if paragraph:
                                match = {
                                    'crd_file': crd_file.file_path.name,
                                    'ecu_section': section.name,
                                    'line_span': f"{section.start_line}-{section.end_line}",
                                    'rule_idx': rule.rule_idx,
                                    'match_score': ecu_analysis['match_score'],
                                    'ecu_analysis': ecu_analysis,
                                    'status': status,
                                    'matched_snippet': paragraph,
                                    'section': section,
                                    'rule': rule,
                                    'paragraph': paragraph,
                                    'match_type': 'llm_based',
                                    'analysis_method': 'llm'
                                }
                                section_matches.append(match)
                            else:
                                print(f"LLM No match (score: {ecu_analysis['match_score']:.3f})")
                            
                    except Exception as e:
                        print(f"LLM analysis failed: {e}, falling back to quick analysis...")
                        
                        # Fallback to quick analysis
                        try:
                            ecu_analysis = self.llm_client._quick_ecu_analysis(section.content, rule.original_text)
                            
                            if ecu_analysis['match_score'] >= self.threshold:
                                print(f"Quick Match found! Score: {ecu_analysis['match_score']:.3f}")
                                # Find best paragraph for injection
                                paragraph, paragraph_score, status = section.find_best_paragraph(rule)
                                
                                if paragraph:
                                    match = {
                                        'crd_file': crd_file.file_path.name,
                                        'ecu_section': section.name,
                                        'line_span': f"{section.start_line}-{section.end_line}",
                                        'rule_idx': rule.rule_idx,
                                        'match_score': ecu_analysis['match_score'],
                                        'ecu_analysis': ecu_analysis,
                                        'status': status,
                                        'matched_snippet': paragraph,
                                        'section': section,
                                        'rule': rule,
                                        'paragraph': paragraph,
                                        'match_type': 'quick_based',
                                        'analysis_method': 'quick'
                                    }
                                    section_matches.append(match)
                                else:
                                    print(f"Quick No match (score: {ecu_analysis['match_score']:.3f})")
                                
                        except Exception as e2:
                            print(f"Both LLM and quick analysis failed: {e2}")
                
                # Select the best match from all potential matches for this section
                if section_matches:
                    # Sort by match score and select the highest
                    section_matches.sort(key=lambda x: x['match_score'], reverse=True)
                    matches.append(section_matches[0])
                    print(f"Selected best match for {section.name}: Rule {section_matches[0]['rule_idx']} (score: {section_matches[0]['match_score']:.3f})")
            
            # Break outer loop if we've reached the limit
            if llm_call_count > max_combinations:
                break
        
        # Count analysis methods used
        llm_matches = len([m for m in matches if m.get('analysis_method') == 'llm'])
        quick_matches = len([m for m in matches if m.get('analysis_method') == 'quick'])
        
        print(f"Analysis complete! Total combinations: {llm_call_count}, Matches found: {len(matches)}")
        print(f"   LLM matches: {llm_matches}")
        print(f"   Quick matches: {quick_matches}")
        return matches
    
    
    
    
    def inject_rules_iterative(self, crd_files: List[CRDFile], max_iterations: int = 5, initial_matches: List[Dict] = None) -> List[Dict]:
        """Iteratively inject rules: one error per iteration, 5 iterations total."""
        all_injected_results = []
        
        # Use provided matches or find new ones
        if initial_matches is None:
            print("Initial scan for all matches...")
            initial_matches = self.find_matches(crd_files)
        else:
            print(f"Using provided {len(initial_matches)} matches for injection")
        
        if not initial_matches:
            print("No matches found in initial scan")
            return []
        
        print(f"Found {len(initial_matches)} initial matches")
        
        # Sort all matches by score
        initial_matches.sort(key=lambda m: m.get('match_score', 0.0), reverse=True)
        
        # Process top matches iteratively (without re-scanning)
        for iteration in range(min(max_iterations, len(initial_matches))):
            print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")
            
            # Select the next best match
            best_match = initial_matches[iteration]
            
            print(f"Selected match: Rule {best_match['rule_idx']} in {best_match['crd_file']} (score: {best_match['match_score']:.3f})")
            
            # Inject only this one match
            if best_match['status'] == 'inject':
                try:
                    print(f"Rewriting paragraph with LLM...")
                    # Rewrite paragraph with LLM
                    section_context = best_match['section'].content[:2000]
                    rewritten = self.llm_client.rewrite_with_llm(
                        best_match['paragraph'],
                        best_match['rule'].condition,
                        best_match['rule'].response,
                        section_context
                    )
                    # Normalize terminology
                    rewritten = rewritten.replace('ventilated sheet ECU', 'ventilated seat ECU')
                    best_match['injected_paragraph'] = rewritten
                    best_match['iteration'] = iteration + 1
                    all_injected_results.append(best_match)
                    
                    print(f"Successfully injected rule {best_match['rule_idx']}")
                    
                    # Update the original file content for next iteration
                    self._apply_single_injection(best_match, crd_files)
                    
                except Exception as e:
                    print(f"Error injecting rule {best_match['rule_idx']}: {e}")
                    best_match['injected_paragraph'] = best_match['paragraph']
                    best_match['iteration'] = iteration + 1
                    all_injected_results.append(best_match)
        else:
                print(f"Rule {best_match['rule_idx']} already exists, skipping")
                best_match['injected_paragraph'] = best_match.get('paragraph', '')
                best_match['iteration'] = iteration + 1
                all_injected_results.append(best_match)
        
        return all_injected_results
    
    def _apply_single_injection(self, match: Dict, crd_files: List[CRDFile]):
        """Apply a single injection to update the CRD file content for next iteration."""
        # Find the corresponding CRDFile object
        target_file = None
        for crd_file in crd_files:
            if crd_file.file_path.name == match['crd_file']:
                target_file = crd_file
                break
        
        if not target_file:
            return
        
        # Update the file content
        original_content = target_file.content
        original_para = match['paragraph']
        new_para = match['injected_paragraph']
        
        if original_para != new_para:
            para_start = original_content.find(original_para)
            if para_start != -1:
                updated_content = (
                    original_content[:para_start] + 
                    new_para + 
                    original_content[para_start + len(original_para):]
                )
                target_file.content = updated_content
                # Re-split sections with updated content
                target_file.sections = target_file._split_sections()
    
    def inject_rules(self, matches: List[Dict]) -> List[Dict]:
        """Legacy method for backward compatibility."""
        return self.inject_rules_iterative([], 1)
    
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
                    
                    # Add Related EARS Rules section
                    f.write("**Related EARS Rules:**\n\n")
                    f.write(f"Rule {match['rule_idx']}: {match['rule'].original_text}\n\n")
                    
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
        print("CHECKPOINT MAIN-1: Starting main process")
        print()
        
        # Scan CRD files
        print("CHECKPOINT MAIN-2: Scanning CRD files...")
        crd_files = self.scan_crd_files(crd_dir)
        if not crd_files:
            print("No CRD files found.")
            return
        
        print(f"Found {len(crd_files)} CRD files")
        print("CHECKPOINT MAIN-3: CRD files loaded successfully")
        print()
        
        # Find matches
        print("Finding matches between rules and CRD sections...")
        # 打印各 section 标题与行号
        total_sections = sum(len(cf.sections) for cf in crd_files)
        print(f"Total sections: {total_sections}")
        for cf in crd_files:
            print(f"Sections in {cf.file_path.name}:")
            for sec in cf.sections:
                # Handle Unicode characters in section names
                safe_name = sec.name.encode('ascii', 'replace').decode('ascii')
                print(f"- [{sec.start_line}-{sec.end_line}] {safe_name}")
        
        # Use optimized method for faster processing
        matches = self.find_matches_optimized(crd_files)
        print(f"Found {len(matches)} matches")
        print()
        
        # Inject rules iteratively
        print("Injecting rules using LLM (iterative approach)...")
        print(f"CHECKPOINT MAIN-4: Using {len(matches)} matches for injection")
        injected_matches = self.inject_rules_iterative(crd_files, max_iterations=5, initial_matches=matches)
        print(f"Processed {len(injected_matches)} matches across 5 iterations")
        # Concise test output: only show first injected item (if any)
        first_injected = next((m for m in injected_matches if m.get('status') in ('inject','limit_skipped')), None)
        if first_injected:
            print("\n=== Test Preview (single issue) ===")
            print(f"File: {first_injected.get('crd_file')}")
            print(f"Section: {first_injected.get('ecu_section')}  Lines: {first_injected.get('line_span')}")
            print(f"Rule {first_injected['rule_idx']}: {first_injected['rule'].original_text}")
            print("--- Original Paragraph ---")
            original_para = first_injected.get('paragraph','')[:2000]
            safe_original = original_para.encode('ascii', 'replace').decode('ascii')
            print(safe_original)
            print("--- Modified Paragraph ---")
            modified_para = first_injected.get('injected_paragraph','')[:2000]
            safe_modified = modified_para.encode('ascii', 'replace').decode('ascii')
            print(safe_modified)
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
    parser.add_argument("--crd-dir", default="./CRD", help="Directory containing CRD files (default: /./CRD)")
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
            # Section filter 如果用户指定了章节过滤
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
                    safe_name = sec.name.encode('ascii', 'replace').decode('ascii')
                    print(f"- [{sec.start_line}-{sec.end_line}] {safe_name}")
            matches = injector.find_matches_optimized(crd_files)
            print(f"Found {len(matches)} matches")
            
            # 统计分析方法使用情况
            llm_matches = len([m for m in matches if m.get('analysis_method') == 'llm'])
            quick_matches = len([m for m in matches if m.get('analysis_method') == 'quick'])
            llm_optimized_matches = len([m for m in matches if m.get('analysis_method') == 'llm_optimized'])
            
            print(f"\nAnalysis Method Statistics:")
            print(f"   LLM matches: {llm_matches}")
            print(f"   Quick fallback matches: {quick_matches}")
            print(f"   LLM optimized matches: {llm_optimized_matches}")
            
            # 详细显示每个匹配的分析方法
            if matches:
                print(f"\nDetailed Match Analysis:")
                for i, match in enumerate(matches, 1):
                    method = match.get('analysis_method', 'unknown')
                    print(f"   {i}. {match.get('crd_file')} - {match.get('ecu_section')} - Rule {match['rule_idx']} (Score: {match['match_score']:.3f}) - Method: {method}")
            
            print()
            print("Injecting rules using LLM...")
            injected_matches = injector.inject_rules_iterative(crd_files, max_iterations=5, initial_matches=matches)
            print(f"Processed {len(injected_matches)} matches")
            first_injected = next((m for m in injected_matches if m.get('status') in ('inject','limit_skipped')), None)
            if first_injected:
                print("\n=== Test Preview (single issue) ===")
                print(f"File: {first_injected.get('crd_file')}")
                print(f"Section: {first_injected.get('ecu_section')}  Lines: {first_injected.get('line_span')}")
                print(f"Rule {first_injected['rule_idx']}: {first_injected['rule'].original_text}")
                print("--- Original Paragraph ---")
                original_para = first_injected.get('paragraph','')[:2000]
                safe_original = original_para.encode('ascii', 'replace').decode('ascii')
                print(safe_original)
                print("--- Modified Paragraph ---")
                modified_para = first_injected.get('injected_paragraph','')[:2000]
                safe_modified = modified_para.encode('ascii', 'replace').decode('ascii')
                print(safe_modified)
                print("--- Related EARS Rules ---")
                print(f"Rule {first_injected['rule_idx']}: {first_injected['rule'].original_text}")
                print("===============================\n")
            print("Generating output files...")
            injector.generate_outputs(injected_matches, args.output_dir, args.apply)
            
            # 显示注入结果的分析方法统计
            if injected_matches:
                print(f"\nInjection Results Analysis:")
                injected_llm = len([m for m in injected_matches if m.get('analysis_method') == 'llm'])
                injected_quick = len([m for m in injected_matches if m.get('analysis_method') == 'quick'])
                injected_optimized = len([m for m in injected_matches if m.get('analysis_method') == 'llm_optimized'])
                
                print(f"   LLM-based injections: {injected_llm}")
                print(f"   Quick fallback injections: {injected_quick}")
                print(f"   LLM optimized injections: {injected_optimized}")
                
                # 显示每个注入的详细信息
                print(f"\nInjection Details:")
                for i, match in enumerate(injected_matches, 1):
                    method = match.get('analysis_method', 'unknown')
                    status = match.get('status', 'unknown')
                    
                    print(f"   {i}. {match.get('crd_file')} - {match.get('ecu_section')} - Rule {match['rule_idx']} (Score: {match['match_score']:.3f}) - Method: {method} - Status: {status}")
            
            print()
            print("EARS injection complete!")
            print()
            print("============================================================")
            print("SECURITY REMINDER:")
            print("- Output files contain document fragments")
            print("- Consider deleting output files after use if confidentiality is critical")
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
