import re
import json
import difflib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available, using stdlib difflib")

from check_utils import truncate_to_word_limit
from ears_parsing import EARSRule

# Process CRD document sections
# Represents a section of a CRD file, containing content, line numbers, etc.
class CRDSection:
    """Represents a section of a CRD file."""
    
    def __init__(self, name: str, content: str, start_line: int, end_line: int, llm_client=None):
        self.name = name
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.paragraphs = self._split_paragraphs()
        self.llm_client = llm_client
    
    # Split section content into paragraphs.
    def _split_paragraphs(self) -> List[str]:
        """Split section content into paragraphs."""
        # Split by double newlines or significant whitespace
        paragraphs = re.split(r'\n\s*\n', self.content.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    
    # Heuristic method to detect if paragraph is table-like/list/figure, if so do not inject long text.
    def _is_table_like(self, paragraph: str) -> bool:
        """Heuristic to detect tables/lists/figures where we should not inject long text."""
        lines = paragraph.split('\n')
        
        # Table/figure titles often begin with specific keywords
        if any(line.strip().lower().startswith(('table', 'figure', 'fig.')) for line in lines[:2]):
            return True

        # Heuristic 1: Wide spacing columns  — detect runs of >= 10 spaces in multiple lines
        wide_space_lines = sum(1 for ln in lines if re.search(r'\s{10,}', ln))
        if wide_space_lines >= 2:
            return True

        # Heuristic 2: Column alignment via repeated space runs at similar positions across lines
        # Collect start indices of space-runs (>= 6 spaces) for each line, bucket by position
        bucket_counts: Dict[int, int] = {}
        for ln in lines:
            for m in re.finditer(r'\s{6,}', ln):
                # Bucket by grouping positions to nearest 4 chars to allow slight drift
                pos_bucket = (m.start() // 4) * 4
                bucket_counts[pos_bucket] = bucket_counts.get(pos_bucket, 0) + 1
        # If there are at least 2 column buckets that appear in >=3 lines, likely a table
        aligned_buckets = [cnt for cnt in bucket_counts.values() if cnt >= 3]
        if len(aligned_buckets) >= 2:
            return True

        # Heuristic 3: Many very short lines is often formatting noise, not a table — do not classify as table
        # (removed earlier aggressive short-line/table heuristics)

        # Bulleted lists are not tables; avoid treating them as table-like content
        # Keep bullets detection to skip long injection into lists
        if re.search(r'^\s*[•\-\*]\s+', paragraph, re.MULTILINE):
            return True

        # TOC-like dotted leaders
        if re.search(r'\.{5,}', paragraph):
            return True

        # Do NOT treat numbered section headings as tables

        return False
    
    # Check if paragraph has sufficient context for rule injection.
    def _has_sufficient_context(self, paragraph: str) -> bool:
        """Check if paragraph has sufficient context for rule injection."""
        # Must have minimum length to contain meaningful context
        if len(paragraph.strip()) < 50:
            return False
        
        # Must not be just a list of items
        lines = paragraph.split('\n')
        if len(lines) > 1 and all(len(line.strip()) <= 30 for line in lines):
            return False
        
        # Must contain some descriptive text (not just keywords)
        words = paragraph.split()
        if len(words) < 10:
            return False
        
        return True
    
    # Find the best paragraph for rule injection.
    def find_best_paragraph(self, rule: EARSRule) -> Tuple[str, float, str]:
        """Find the best paragraph for rule injection."""
        best_paragraph = ""
        best_score = 0.0
        best_status = "inject"
        
        for paragraph in self.paragraphs:
            if self._is_table_like(paragraph):
                continue
            
            # Additional validation: ensure paragraph has sufficient context
            if not self._has_sufficient_context(paragraph):
                continue
            
            # Check if paragraph already contains the condition (IF-part) only
            condition_matches = re.findall(rule.normalized_condition, paragraph, re.IGNORECASE)
            if condition_matches:
                # Condition already exists in this paragraph
                return paragraph, 1.0, "exists"
            
            # Score paragraph based on temporal/conditional cues and similarity
            score = self._score_paragraph(paragraph, rule)
            
            if score > best_score:
                best_score = score
                best_paragraph = paragraph
                best_status = "inject"
        
        return best_paragraph, best_score, best_status
    
    # Score paragraph to determine if it is suitable for rule injection.
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

    # Scan for ECU components and nearby conditions/events in the section (using LLM).
    def scan_ecu_and_conditions(self) -> List[Dict]:
        """Scan for ECU components and nearby conditions/events in the section."""
        # For performance reasons with local LLM, default to regex/heuristic scan
        return self._fallback_scan_ecu_and_conditions()
    
    # Use LLM to intelligently scan ECU components and conditions.
    def _llm_scan_ecu_and_conditions(self) -> List[Dict]:
        """Use LLM to intelligently scan ECU components and conditions"""
        ecu_conditions = []
        
        # Build LLM prompt for ECU and condition identification
        # Apply 500-word limit to input content
        limited_content = truncate_to_word_limit(self.content, 500)
        
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
{limited_content}

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
            # Call LLM for intelligent identification
            response = self.llm_client._call_ollama_api(prompt)
            
            # Parse LLM response and convert to ecu_conditions format
            ecu_conditions = self._parse_llm_ecu_response(response)
            return ecu_conditions
            
        except Exception as e:
            print(f"LLM scan error: {e}")
            return ecu_conditions
    
    # Parse LLM's ECU identification response.
    def _parse_llm_ecu_response(self, llm_response: str) -> List[Dict]:
        """Parse LLM's ECU identification response"""
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
                    # Try to extract numeric line number from string
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
                        'ecu_line': line_num,  # May be None, handle fallback when used later
                        'ecu_text': component.get('ecu_name', 'Unknown ECU'),
                        'ecu_matches': [component.get('ecu_name', 'Unknown ECU')],
                        'context_lines': [component.get('context', 'No context')],
                        'context_text': component.get('context', 'No context'),
                        'conditions': component.get('conditions', []),
                        'section_name': self.name
                    })
                    
        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
        except Exception as e:
            print(f"Failed to parse LLM response: {e}")
        
        return ecu_conditions
        
    # Fallback ECU scanning method (based on regex).
    def _fallback_scan_ecu_and_conditions(self) -> List[Dict]:
        """Fallback ECU scanning method (based on regex)"""
        ecu_conditions = []
        
        # Original regex logic as fallback
        ecu_patterns = [
            r'ECU\s',  # ECU, ECU, etc.
            r'[A-Z]+\s+ECU',  # Ventilated seat ECU, Steering heater ECU, etc.
            r'[A-Z]+\s'  #ECGW
        ]
        
        condition_patterns = []
        
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
                    
                    # Collect conditions only if patterns are configured
                    conditions = []
                    if condition_patterns:
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


# Represents a CRD file with sections.
class CRDFile:
    """Represents a CRD file with sections."""
    
    def __init__(self, file_path: Path, llm_client=None):
        self.file_path = file_path
        self.content = self._read_file()
        self.llm_client = llm_client
        self.sections = self._split_sections()
    
    # Read file content, supports UTF-8 and latin-1 encoding.
    def _read_file(self) -> str:
        """Read file content with UTF-8 encoding."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with open(self.file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    # Split file into sections based on headings.
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
        
        # Filter TOC-like sections: early pages, many dots, numbered only without title, very short content
        def _is_toc_like_section(sec: CRDSection) -> bool:
            # Only consider TOC within first few lines of document
            if sec.start_line > 250:
                return False
            # Skip sections with too many dots (TOC markers)
            if re.search(r"\.{5,}", sec.content):
                return False
            return True
        
        # Further: truncate from first "substantive numbered heading", discard all sections before it
        def _is_substantive_numbered_heading(name: str, content: str) -> bool:
            # Forms like 1-1. Title or 1.1 Title or 1-1-1. Title, etc., with sufficient content
            if re.match(r"^\d+(?:[.-]\d+){0,3}\.?\s+.+$", name.strip()):
                content_lines = [ln for ln in content.split('\n') if ln.strip()]
                # Content line count threshold (body text should be longer)
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
        
        # Truncate again: from first substantive numbered heading starting with "1-" or "1." (usually Chapter 1 of body text)
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
