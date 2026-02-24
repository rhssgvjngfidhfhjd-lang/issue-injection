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

# Paragraphs containing any of these phrases are excluded from matching.
# Matching is case-insensitive and literal (quotes are treated as characters).
BLACKLIST_PHRASES = [
    "SUZUKI LIN SPECIFICATION",
]

# Process CRD document sections
# Represents a section of a CRD file, containing content, line numbers, etc.
class CRDSection:
    """Represents a section of a CRD file."""

    def __init__(
        self,
        name: str,
        content: str,
        start_line: int,
        end_line: int,
        llm_client=None,
        paragraphs: Optional[List[str]] = None,
        element_ids: Optional[List[str]] = None,
        section_id: Optional[str] = None,
    ):
        self.name = name
        self.content = content
        self.start_line = start_line
        self.end_line = end_line
        self.llm_client = llm_client
        self.element_ids = element_ids  # For JSON: paragraph[i] <-> element_ids[i]
        self.section_id = section_id  # For JSON: section id in parsed_document
        if paragraphs is not None:
            self.paragraphs = paragraphs
        else:
            self.paragraphs = self._split_paragraphs()

    def get_element_id_for_paragraph(self, paragraph: str) -> Optional[str]:
        """Return element id for a paragraph (JSON mode only)."""
        if not self.element_ids:
            return None
        try:
            idx = self.paragraphs.index(paragraph)
            return self.element_ids[idx] if idx < len(self.element_ids) else None
        except ValueError:
            return None

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

    def _contains_blacklisted_phrase(self, paragraph: str) -> bool:
        """Check if paragraph contains any blacklisted phrase."""
        if not BLACKLIST_PHRASES:
            return False
        lowered = paragraph.lower()
        for phrase in BLACKLIST_PHRASES:
            if phrase and phrase.lower() in lowered:
                return True
        return False

    def _is_valid_paragraph_structure(self, paragraph: str) -> bool:
        """Check paragraph is a short prose block (<=3 sentences), not a glossary list."""
        lines = [ln.strip() for ln in paragraph.split('\n') if ln.strip()]
        # Glossary/definition-list heuristic: many lines, most without a period
        if len(lines) >= 5:
            lines_without_period = sum(1 for ln in lines if '.' not in ln)
            if lines_without_period / len(lines) > 0.6:
                return False
        # Sentence count: split on period followed by space or end-of-string
        sentences = [s.strip() for s in re.split(r'\.\s+|\.\s*$', paragraph) if s.strip()]
        if len(sentences) > 3:
            return False
        return True

    def _check_ecu_density(self, paragraph: str) -> Tuple[bool, bool]:
        """Check ECU density per sentence.
        Returns (passes_minimum, has_bonus):
          passes_minimum: at least 1 sentence has >= 2 ECU references
          has_bonus: >= 2 sentences each have >= 2 ECU references (auto-qualify)
        """
        ecu_pattern = re.compile(r'\bECU\b|\bECGW\b', re.IGNORECASE)
        sentences = [s.strip() for s in re.split(r'\.\s+|\.\s*$', paragraph) if s.strip()]
        rich_count = 0  # sentences with >= 2 ECU refs
        for sent in sentences:
            if len(ecu_pattern.findall(sent)) >= 2:
                rich_count += 1
        passes = rich_count >= 1
        bonus = rich_count >= 2
        return passes, bonus

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

            # Skip paragraphs containing blacklisted phrases
            if self._contains_blacklisted_phrase(paragraph):
                continue

            if not self._is_valid_paragraph_structure(paragraph):
                continue
            if not re.search(r'\bECU\b|\bECGW\b', paragraph, re.IGNORECASE):
                continue
            _, bonus = self._check_ecu_density(paragraph)

            # Check if paragraph already contains the condition (IF-part) only
            condition_matches = re.findall(rule.normalized_condition, paragraph, re.IGNORECASE)
            if condition_matches:
                # Condition already exists in this paragraph
                return paragraph, 1.0, "exists"
            
            # Score paragraph based on temporal/conditional cues and similarity
            score = self._score_paragraph(paragraph, rule)
            if bonus:
                score = max(score, 1.0)
            
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
        """Fallback ECU scanning method (based on regex). For JSON mode, uses paragraphs and element_ids."""
        ecu_conditions = []
        ecu_patterns = [
            r"ECU\s",
            r"[A-Z]+\s+ECU",
            r"ECGW",
        ]
        condition_patterns = []

        if self.element_ids:
            # JSON mode: iterate over paragraphs, store element_id
            for i, paragraph in enumerate(self.paragraphs):
                for pattern in ecu_patterns:
                    ecu_matches = re.findall(pattern, paragraph, re.IGNORECASE)
                    if ecu_matches:
                        context_start = max(0, i - 2)
                        context_end = min(len(self.paragraphs), i + 3)
                        context_paras = self.paragraphs[context_start:context_end]
                        context_text = "\n".join(context_paras)
                        element_id = self.element_ids[i] if i < len(self.element_ids) else None
                        ecu_conditions.append({
                            "ecu_line": None,
                            "element_id": element_id,
                            "ecu_text": paragraph[:200],
                            "ecu_matches": ecu_matches,
                            "context_lines": context_paras,
                            "context_text": context_text,
                            "conditions": [],
                            "section_name": self.name,
                            "section_id": self.section_id,
                        })
                        break
            return ecu_conditions

        # Plain text mode
        lines = self.content.split("\n")
        for i, line in enumerate(lines):
            for pattern in ecu_patterns:
                ecu_matches = re.findall(pattern, line, re.IGNORECASE)
                if ecu_matches:
                    context_start = max(0, i - 3)
                    context_end = min(len(lines), i + 4)
                    context_lines = lines[context_start:context_end]
                    context_text = "\n".join(context_lines)
                    conditions = []
                    if condition_patterns:
                        for cond_pattern in condition_patterns:
                            cond_matches = re.findall(cond_pattern, context_text, re.IGNORECASE)
                            for m in cond_matches:
                                if isinstance(m, tuple):
                                    conditions.append(
                                        " ".join([p for p in m if isinstance(p, str)]).strip()
                                    )
                                else:
                                    conditions.append(str(m).strip())
                    ecu_conditions.append({
                        "ecu_line": i + 1,
                        "ecu_text": line.strip(),
                        "ecu_matches": ecu_matches,
                        "context_lines": context_lines,
                        "context_text": context_text,
                        "conditions": conditions,
                        "section_name": self.name,
                    })
                    break
        return ecu_conditions


# Represents a CRD file with sections.
class CRDFile:
    """Represents a CRD file with sections."""

    def __init__(self, file_path: Path, llm_client=None):
        self.file_path = Path(file_path)
        self.llm_client = llm_client
        raw = self._read_file()
        if isinstance(raw, dict):
            self._is_json = True
            self._json_data = raw
            self.content = self._build_content_from_json(raw)
        else:
            self._is_json = False
            self._json_data = None
            self.content = raw
        self.sections = self._split_sections()

    def _build_content_from_json(self, data: Dict) -> str:
        """Build plain text content from JSON for downstream scoring."""
        parts = []
        doc = data.get("parsed_document", data)
        for sec in doc.get("sections", []):
            for el in sec.get("elements", []):
                if el.get("type") == "text" and el.get("text"):
                    parts.append(el["text"])
        return "\n\n".join(parts)

    # Read file content; returns str for .txt, dict for .json.
    def _read_file(self) -> Any:
        """Read file content. Returns str for text files, dict for JSON."""
        suffix = self.file_path.suffix.lower()
        if suffix == ".json":
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read()
        except UnicodeDecodeError:
            with open(self.file_path, "r", encoding="latin-1") as f:
                return f.read()

    def _split_sections_from_json(self) -> List[CRDSection]:
        """Build CRDSection list from parsed_document.sections."""
        sections = []
        doc = self._json_data.get("parsed_document", self._json_data)
        for sec_data in doc.get("sections", []):
            sec_id = str(sec_data.get("id", ""))
            title = sec_data.get("title", f"Section {sec_id}")
            text_elements = [
                el for el in sec_data.get("elements", [])
                if el.get("type") == "text" and (el.get("text") or "").strip()
            ]
            if not text_elements:
                continue
            paragraphs = [el["text"].strip() for el in text_elements]
            element_ids = [str(el.get("id", "")) for el in text_elements]
            content = "\n\n".join(paragraphs)
            start_line = 1
            end_line = len(element_ids)
            sections.append(
                CRDSection(
                    name=title,
                    content=content,
                    start_line=start_line,
                    end_line=end_line,
                    llm_client=self.llm_client,
                    paragraphs=paragraphs,
                    element_ids=element_ids,
                    section_id=sec_id,
                )
            )
        return sections if sections else [
            CRDSection("Entire Document", self.content, 1, 1, self.llm_client)
        ]

    # Split file into sections based on headings (txt) or parsed_document (json).
    def _split_sections(self) -> List[CRDSection]:
        """Split file into sections based on headings."""
        if self._is_json and self._json_data:
            return self._split_sections_from_json()

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
