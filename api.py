import os
import re
import json
import requests
from typing import Optional, List, Dict, Tuple, Any
from check_utils import truncate_to_word_limit

# Client for interacting with local LLM endpoint.
class LLMClient:
    """Client for interacting with local LLM endpoint (Optimized for A100 GPU)."""
    
    def __init__(self, base_url: str = None, api_key: str = None, similarity_threshold: float = 0.8, model: Optional[str] = None):
        # 1. Determine Base URL: priority order: parameter > environment variable > 11435 GPU port
        self.base_url = (
            base_url or 
            os.getenv('OLLAMA_BASE_URL') or 
            os.getenv('OLLAMA_HOST') or 
            'http://127.0.0.1:11435'
        ).rstrip('/')

        self.api_key = api_key or os.getenv('OLLAMA_API_KEY', '')
        self.model = model or os.getenv('OLLAMA_MODEL')
        
        # 2. Auto-detect model (prefer from 11435 port)
        if not self.model:
            try:
                tags_url = f"{self.base_url}/api/tags"
                response = requests.get(tags_url, timeout=5)
                if response.status_code == 200:
                    tags = response.json().get('models', [])
                    names = [str(m.get('name','')) for m in tags]
                    prefer = ([n for n in names if n.startswith('qwen3:')] or
                              [n for n in names if n.startswith('deepseek')] or
                              names)
                    self.model = prefer[0] if prefer else 'qwen3:8b'
                else:
                    self.model = 'qwen3:8b'
            except Exception:
                self.model = 'qwen3:8b'
                
        self.similarity_threshold = similarity_threshold
        self.client = None
        print(f"LLMClient initialized. Target: {self.base_url}, Model: {self.model}")
    
    def _is_append_style(self, original_text: str, result: str) -> bool:
        """Detect if result is append-style (original + new sentence) rather than mid-sentence insertion."""
        if not result or not result.strip():
            return False
        ot = original_text.strip()
        rt = result.strip()
        if re.search(r"When\s+\w+\s+sends\s+a\s+request\s*,\s*if\s+", rt, re.IGNORECASE):
            return True
        if rt.startswith(ot) and len(rt) > len(ot) * 1.2:
            return True
        return False

    def rewrite_with_llm(self, original_text: str, ears_rule: str) -> str:
        """Inject EARS rule into Original Text using minimal mid-sentence editing prompt."""
        original_text = truncate_to_word_limit(original_text, 300)
        retry_prompt_suffix = "\n\nCRITICAL: You MUST insert a clause INSIDE an existing sentence. Do NOT append a new sentence at the end."

        try:
            instruction = self._build_prompt_v2(original_text, ears_rule)
            result = self._call_ollama_api(instruction)
            if not result or not result.strip():
                return self._fallback_rewrite(original_text, ears_rule)
            if self._is_append_style(original_text, result):
                instruction = self._build_prompt_v2(original_text, ears_rule) + retry_prompt_suffix
                result = self._call_ollama_api(instruction)
            if not result or not result.strip() or self._is_append_style(original_text, result):
                return self._fallback_rewrite(original_text, ears_rule)
            return result
        except Exception as e:
            print(f"Ollama API error: {e}")
            return self._fallback_rewrite(original_text, ears_rule)

    def _build_prompt_v2(self, original_text: str, ears_rule: str) -> str:
        """Build Prompt: Issue injection with minimal mid-sentence editing."""
        return f"""You are doing "issue injection" into a CRD paragraph.

Goal:
Given (1) an EARS-style rule and (2) an original paragraph (original text),
generate ONE injected paragraph by minimally editing the original text so that it naturally introduces the rule's defect condition / issue.

Inputs
[EARS_RULE]
{ears_rule}

[ORIGINAL_TEXT]
{original_text}

Output requirements
1) Output ONLY the injected paragraph text. No explanation, no JSON.
2) Keep all existing wording unchanged as much as possible. Prefer inserting a short clause *inside an existing sentence* (mid-sentence) rather than rewriting or adding a new sentence, unless impossible.
3) The insertion point must be explicit and natural:
   - Choose a single best insertion point in the original text (preferably in the middle of a sentence).
   - Mimic the style of example1 (a "before …" clause embedded into an existing sentence).
4) Do NOT change technical terms, IDs, ECU names, signal names, figure/table references, section numbering, or formatting (line breaks and bullet symbols).
5) The injected content must reflect the rule precisely:
   - Preserve actor ECUs and step ordering (e.g., "ECU B … before ECU C …").
   - Only mention "reject / report sequence error" if it can be embedded naturally in the same sentence without making the paragraph awkward; otherwise inject only the ordering violation.
6) Use formal specification tone.

Rewrite strategy (must follow)
- Locate the single best sentence in ORIGINAL_TEXT that can host the injection.
- Insert one minimal clause (prefer "before …" / "prior to …") inside that sentence.
- Keep all other text verbatim.

Now generate the injected paragraph."""
    
    def _call_ollama_api(self, instruction: str) -> str:
        """Call native Ollama generate API with GPU force options."""
        url = f"{self.base_url}/api/generate"
        prompt = (
            "You are a Senior Automotive System Architect with 10+ years of experience. "
            "You are an expert in ISO 26262 standards and the precise language used in "
            "System Requirement Specifications (SRS).\n\n"
            + instruction
        )
        
        # Core configuration: add num_gpu to force A100 usage
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "num_predict": 1024,
                "temperature": 0.4,
                "keep_alive": "10m",   # Keep model in GPU memory longer
                "num_gpu": 99,         # Key: force all 99 layers into GPU (A100 has enough capacity)
                "num_thread": 8        # Limit CPU threads to force GPU processing
            }
        }

        # Optimization for DeepSeek
        if self.model and "deepseek" in str(self.model).lower():
            try:
                payload["options"]["think"] = "low"
            except: pass

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
            
        response = requests.post(url, json=payload, headers=headers, timeout=300, stream=True)
        response.raise_for_status()
        
        content_parts = []
        for line in response.iter_lines(decode_unicode=True):
            if not line: continue
            try:
                obj = json.loads(line)
                if 'response' in obj:
                    content_parts.append(obj['response'])
                if obj.get('done'): break
            except: continue
            
        content = ''.join(content_parts).strip()
        
        # Clean thinking process tags
        content = re.sub(r'<(thinking|think)>.*?</\1>', '', content, flags=re.DOTALL | re.IGNORECASE)
        # Extract content from markdown code blocks instead of deleting (qwen3 wraps output in ```)
        code_block = re.search(r'```(?:\w*\n)?(.*?)```', content, re.DOTALL)
        if code_block:
            content = code_block.group(1).strip()
        else:
            content = re.sub(r'```', '', content)
        # Remove common conversational prefixes
        content = re.sub(r'^(Okay|Sure|Here is|Polished paragraph:|Rewritten paragraph:).*?:\s*', '', content, flags=re.IGNORECASE | re.MULTILINE)
        return truncate_to_word_limit(content.strip(), 500)
    
    def _extract_ocr_from_ears_rule(self, ears_rule: str) -> Tuple[str, str, str]:
        """Extract O/C/R from ears_rule string for fallback. Returns (object, condition, response)."""
        text = ears_rule.strip()
        o, c, r = "System", "", ""
        # Structured format: O: ...; C: ...; R_esp: ...
        m = re.match(
            r"O\s*:\s*(.+?)\s*;\s*C\s*:\s*(.+?)\s*;\s*R_esp\s*:\s*(.+?)(?:\s*;|\s*$)",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            o, c, r = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            return o, c, r
        # Short format O; C; MUTATION with C containing THEN
        m2 = re.match(
            r"O\s*:\s*(.+?)\s*;\s*C\s*:\s*(.+?)\s*;\s*MUTATION\s*:\s*.+$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m2:
            o = m2.group(1).strip()
            c_val = m2.group(2).strip()
            if " THEN " in c_val.upper():
                parts = re.split(r"\s+THEN\s+", c_val, maxsplit=1, flags=re.IGNORECASE)
                c = parts[0].replace("IF", "").strip()
                r = parts[1].strip() if len(parts) > 1 else ""
            else:
                c, r = c_val, ""
            return o, c, r
        # Legacy IF ... THEN ... format
        if "THEN" in text:
            parts = text.split("THEN", 1)
            c = parts[0].replace("IF", "").strip()
            r = parts[1].strip() if len(parts) > 1 else ""
            return "System", c, r
        return o, c, r

    def _ensure_before_min_length(self, before_clause: str, condition: str) -> str:
        """If before clause content is too short (< 5 chars), use condition summary instead."""
        content = before_clause[len("before "):] if before_clause.startswith("before ") else before_clause
        if len(content.strip()) < 5:
            c_short = re.sub(r"\s+", " ", condition)[:80].strip().strip("\"'").strip(",")
            return "before " + c_short if c_short else "before the required step"
        return before_clause

    def _condition_to_before_clause(self, condition: str, ecu_mapping: Dict[str, str]) -> str:
        """Convert EARS condition into a short 'before ...' clause for mid-sentence insertion."""
        c = condition.strip()
        for placeholder, actual in ecu_mapping.items():
            c = c.replace(placeholder, actual)
        c = re.sub(r"\bECU [A-Z]\b", "the other ECU", c)
        # Extract "before X" if present
        m = re.search(r"\bbefore\s+(.+?)(?:\.|$)", c, re.IGNORECASE | re.DOTALL)
        if m:
            return self._ensure_before_min_length("before " + m.group(1).strip(), c)
        # "without first executing/without previously sending" -> "before [required step]"
        m = re.search(r"without\s+(?:first\s+)?(?:executing\s+the\s+step\s+)?[\"']?(.+?)[\"']?\s*(?:THEN|$)", c, re.IGNORECASE | re.DOTALL)
        if m:
            return self._ensure_before_min_length("before " + m.group(1).strip(), c)
        m = re.search(r"without\s+previously\s+(.+?)(?:,|\.|$)", c, re.IGNORECASE | re.DOTALL)
        if m:
            return self._ensure_before_min_length("before " + m.group(1).strip(), c)
        # "omits the explicit step/omits the explicit waiting step X" -> "before X" (greedy capture, strip quotes)
        m = re.search(r"omits\s+(?:the\s+explicit\s+(?:waiting\s+)?step\s+)?(.+)", c, re.IGNORECASE | re.DOTALL)
        if m:
            result = m.group(1).strip().strip("\"'")
            return self._ensure_before_min_length("before " + result, c)
        # Fallback: shorten to ~80 chars and wrap as "before ..."
        c_short = re.sub(r"\s+", " ", c)[:80].strip().strip("\"'").strip(",")
        return self._ensure_before_min_length("before " + c_short if c_short else "before the required step", c)

    def _fallback_rewrite(self, original_text: str, ears_rule: str) -> str:
        """Fallback rewrite with mid-sentence insertion when LLM is unavailable."""
        o, c, r = self._extract_ocr_from_ears_rule(ears_rule)
        text = original_text.strip()
        ecu_names = re.findall(r"\b([A-Za-z][A-Za-z \-/]* ECU)\b", text, re.IGNORECASE)
        ecu_mapping = {f"ECU {chr(65+i)}": name for i, name in enumerate(ecu_names[:4])}
        ecu_mapping.update({f"ECU {chr(97+i)}": name for i, name in enumerate(ecu_names[:4])})

        if not c:
            return truncate_to_word_limit(text, 500)
        before_clause = self._condition_to_before_clause(c, ecu_mapping)

        # Split into sentences (keep period with sentence)
        sentences = [s.strip() for s in re.split(r'(?<=\.)\s+', text) if s.strip()]
        if not sentences:
            return truncate_to_word_limit(text, 500)
        # Pick longest sentence as host
        host_idx = max(range(len(sentences)), key=lambda i: len(sentences[i]))
        host = sentences[host_idx].rstrip()
        if not host.endswith('.'):
            host += '.'
        # Insert "before ..." clause before the final period
        if host.endswith('.'):
            injected_host = host[:-1] + " " + before_clause + "."
        else:
            injected_host = host + " " + before_clause
        sentences[host_idx] = injected_host
        result = " ".join(sentences)
        return truncate_to_word_limit(result, 500)
