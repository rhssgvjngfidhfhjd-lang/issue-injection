import re
import random
from typing import Optional, Tuple

class MutationEngine:
    """
    Handles deterministic mutation operations (Φ_mutate) on text.
    Supported types:
    - numeric_perturbation: Perturbs numerical values
    - step_reorder: Reorders procedural steps (numbered or list-like)
    - action_omission: Omits specific actions or sentences
    """

    @staticmethod
    def apply_mutation(text: str, mutation_type: str, rule_condition: str) -> Tuple[str, str, bool]:
        """
        Applies the specified mutation to the text.
        Returns: (mutated_text, mutation_details, success)
        """
        if mutation_type == "numeric_perturbation":
            return MutationEngine._mutate_numeric_parameter(text)
        elif mutation_type == "step_reorder":
            return MutationEngine._reorder_procedural_steps(text)
        elif mutation_type == "action_omission":
            return MutationEngine._omit_action(text, rule_condition)
        else:
            return text, "No mutation applied (unknown type)", False

    @staticmethod
    def _mutate_numeric_parameter(text: str) -> Tuple[str, str, bool]:
        """Perturbs numerical thresholds found in the text."""
        # Find numbers (integers or decimals, optionally followed by units)
        # Avoid version numbers like 1.2.3 or dates if possible
        matches = list(re.finditer(r'\b(\d+(?:\.\d+)?)\s*([a-zA-Z%]+)?\b', text))
        
        if not matches:
            return text, "No numeric parameters found", False
            
        # Pick one relevant number to mutate (heuristic: pick the one that looks like a parameter)
        # For simplicity, we pick the first one that isn't a small integer like "1" (often a step number) unless it has a unit
        target_match = None
        for m in matches:
            val = float(m.group(1))
            unit = m.group(2)
            # Skip likely step numbers (integers < 10 with no unit)
            if val < 10 and float(val).is_integer() and not unit:
                continue
            target_match = m
            break
            
        if not target_match:
            # Fallback to any number
            target_match = matches[0]

        original_val_str = target_match.group(1)
        original_val = float(original_val_str)
        unit = target_match.group(2) or ""
        
        # Perturb by -20% to -50% (making it stricter or looser)
        # We want to create a defect, so often we lower a threshold or raise it significantly
        mutation_factor = 0.5 
        new_val = original_val * mutation_factor
        
        # Format: keep int if original was int
        if float(original_val_str).is_integer():
            new_val_str = str(int(new_val))
        else:
            new_val_str = f"{new_val:.2f}"
            
        start, end = target_match.span(1)
        mutated_text = text[:start] + new_val_str + text[end:]
        
        details = f"Numeric perturbation: {original_val_str}{unit} -> {new_val_str}{unit}"
        return mutated_text, details, True

    @staticmethod
    def _reorder_procedural_steps(text: str) -> Tuple[str, str, bool]:
        """Reorders numbered steps (e.g., 1. ... 2. ...)."""
        # Regex to find numbered list items: "1. ", "1) ", "(1) "
        # We split the text by these markers
        pattern = r'(?:\b\d+\.|\b\d+\)|\(\d+\))\s'
        parts = re.split(pattern, text)
        matches = list(re.finditer(pattern, text))
        
        if len(parts) < 3: # Need at least intro + 2 steps
            return text, "Not enough steps found for reordering", False
            
        # parts[0] is the intro text
        # parts[1] is step 1 content, parts[2] is step 2 content, etc.
        # matches[0] is marker for step 1, matches[1] is marker for step 2
        
        intro = parts[0]
        steps = parts[1:]
        markers = [m.group() for m in matches]
        
        if len(steps) < 2:
             return text, "Less than 2 steps found", False
             
        # Swap step 1 and step 2
        # We keep the original markers (1., 2.) but swap the content
        step1_content = steps[0]
        step2_content = steps[1]
        
        # Reconstruct
        # Intro + Marker1 + Step2Content + Marker2 + Step1Content + Rest
        mutated_text = intro + markers[0] + step2_content + markers[1] + step1_content
        
        # Append remaining steps
        for i in range(2, len(steps)):
            mutated_text += markers[i] + steps[i]
            
        details = "Step reordering: Swapped Step 1 and Step 2"
        return mutated_text, details, True

    @staticmethod
    def _omit_action(text: str, rule_condition: str) -> Tuple[str, str, bool]:
        """Omits a sentence or clause that matches the rule condition keywords."""
        # Simple heuristic: Split into sentences, remove the one that overlaps most with rule condition
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) < 2:
            return text, "Text too short to omit action", False
            
        best_score = 0
        target_idx = -1
        
        # Tokenize rule condition
        cond_tokens = set(re.findall(r'\w+', rule_condition.lower()))
        
        for i, sent in enumerate(sentences):
            sent_tokens = set(re.findall(r'\w+', sent.lower()))
            overlap = len(cond_tokens.intersection(sent_tokens))
            if overlap > best_score:
                best_score = overlap
                target_idx = i
                
        if target_idx != -1 and best_score > 0:
            removed_sent = sentences[target_idx]
            del sentences[target_idx]
            mutated_text = " ".join(sentences)
            details = f"Action omission: Removed '{removed_sent[:30]}...'"
            return mutated_text, details, True
            
        return text, "No matching action found to omit", False
