import re
from typing import Tuple, Optional

class EARSRule:
    """Represents a parsed EARS rule with O/C/R structure and mutation type."""
    
    def __init__(self, rule_text: str, rule_idx: int):
        self.original_text = rule_text.strip()
        self.rule_idx = rule_idx
        
        # Initialize fields
        self.object: str = ""
        self.condition: str = ""
        self.response: str = ""
        self.mutation_type: str = "llm_rewrite" # Default to legacy behavior
        
        self._parse_rule()
        self.normalized_condition = self._normalize_condition()
    
    def _parse_rule(self):
        """Parse rule string: either O/C/R_esp structured format or IF-THEN format."""
        text = self.original_text
        if re.match(r"^\s*O\s*:", text, re.IGNORECASE):
            self._parse_structured_format(text)
        elif "THEN" in text:
            self._parse_legacy_format(text)
        else:
            raise ValueError(f"Invalid EARS rule format: {self.original_text}")

    def _parse_structured_format(self, text: str):
        """Parse O: ...; C: ...; [R_esp: ...;] MUTATION: ... format. R_esp is optional."""
        # Full format: O; C; R_esp; MUTATION
        m = re.match(
            r"O\s*:\s*(.+?)\s*;\s*C\s*:\s*(.+?)\s*;\s*R_esp\s*:\s*(.+?)\s*;\s*MUTATION\s*:\s*(.+)$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            self.object = m.group(1).strip()
            self.condition = m.group(2).strip()
            self.response = m.group(3).strip()
            self.mutation_type = m.group(4).strip().lower() or "llm_rewrite"
            return
        # Short format: O; C; MUTATION (no R_esp). If C contains "THEN", split it.
        m2 = re.match(
            r"O\s*:\s*(.+?)\s*;\s*C\s*:\s*(.+?)\s*;\s*MUTATION\s*:\s*(.+)$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m2:
            self.object = m2.group(1).strip()
            c_val = m2.group(2).strip()
            self.mutation_type = m2.group(3).strip().lower() or "llm_rewrite"
            if " THEN " in c_val.upper():
                parts = re.split(r"\s+THEN\s+", c_val, maxsplit=1, flags=re.IGNORECASE)
                self.condition = parts[0].replace("IF", "").strip()
                self.response = parts[1].strip() if len(parts) > 1 else ""
            else:
                self.condition = c_val
                self.response = ""
            return
        raise ValueError(f"Invalid structured rule format: {text}")

    def _parse_legacy_format(self, text: str):
        """Parse legacy IF ... THEN ... format."""
        parts = text.split("THEN")
        if len(parts) != 2:
            raise ValueError(f"Invalid EARS rule format: {text}")
        
        self.condition = parts[0].replace("IF", "").strip()
        self.response = parts[1].strip()
        self.object = "System" # Default object for legacy rules
        self.mutation_type = "llm_rewrite" # Legacy rules rely on LLM

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
