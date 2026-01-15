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
        """Parse rule string, supporting both legacy IF-THEN and new structured format."""
        text = self.original_text
        
        # Enhanced heuristic: if it starts with or contains O: or C: at the start of a semicolon part
        if re.search(r'(^|;)\s*(O:|C:)', text):
            self._parse_structured_format(text)
        elif "THEN" in text:
            self._parse_legacy_format(text)
        else:
            raise ValueError(f"Invalid EARS rule format: {self.original_text}")

    def _parse_structured_format(self, text: str):
        """Parse structured format: O: ...; C: ...; R_esp: ...; MUTATION: ..."""
        # Split by semicolons
        parts = [p.strip() for p in text.split(';') if p.strip()]
        
        for part in parts:
            if part.startswith("O:"):
                self.object = part[2:].strip()
            elif part.startswith("C:"):
                self.condition = part[2:].strip()
            elif part.startswith("R_esp:"):
                self.response = part[6:].strip()
            elif part.startswith("MUTATION:"):
                self.mutation_type = part[9:].strip()

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
