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
the nearest paragraph(s) in the original writing style so the rule's condition (IF-part only) is
injected without contradicting the rest of the CRD.
"""

import os
import sys
import re
import argparse
import difflib
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available, using stdlib difflib")

from api import LLMClient
from ears_parsing import EARSRule
from crd_processing import CRDFile, CRDSection
from mutation_ops import MutationEngine

# Main class for EARS rule injection.
class EARSInjector:
    """Main class for EARS rule injection."""
    
    def __init__(self, rules_file: str = "EARSrules.txt", threshold: float = 0.3, similarity_threshold: float = 0.8, model_name: Optional[str] = None):
        self.rules_file = Path(rules_file)
        self.threshold = threshold
        self.similarity_threshold = similarity_threshold
        self.rules = self._parse_rules()
        self.llm_client = LLMClient(similarity_threshold=self.similarity_threshold, model=model_name)
        self.matches = []
    
    # Parse EARS rules from file.
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
    
    # Scan directory for CRD text files.
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
    
    # Find matches between rules and CRD sections.
    def find_matches(self, crd_files: List[CRDFile]) -> List[Dict]:
        """Find matches between rules and CRD sections."""
        matches = []
        
        for crd_file in crd_files:
            for section in crd_file.sections:
                # Scan for ECU components and conditions in this section
                ecu_conditions = section.scan_ecu_and_conditions()
                
                # Collect all potential matches for this section
                section_matches = []
                
                for rule in self.rules:
                    # Try to find the best match based on ECU and condition analysis
                    best_match = self._find_best_ecu_match(ecu_conditions, rule, section, crd_file.file_path.name)
                    
                    if best_match:
                        section_matches.append(best_match)
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
                                    'matched_snippet': paragraph,
                                    'section': section,
                                    'rule': rule,
                                    'paragraph': paragraph,
                                    'match_type': 'fallback'
                                }
                                section_matches.append(match)
                
                # Randomly select one match from all potential matches for this section
                if section_matches:
                    import random
                    selected_match = random.choice(section_matches)
                    matches.append(selected_match)
        
        return matches
    
    # Find best ECU-based match for rule.
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
                        'matched_snippet': paragraph,
                        'section': section,
                        'rule': rule,
                        'paragraph': paragraph,
                        'match_type': 'ecu_based'
                    }
        
        return best_match
    
    # Score how well ECU in rule matches ECU in section.
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
    
    # Analyze ECU pattern in EARS rule.
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
    
    # Analyze ECU pattern in CRD section.
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
    
    # Score how well interaction patterns match between rule and section.
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
    
    # Score how well conditions in rule match conditions in section.
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
    
    # Analyze condition pattern in EARS rule.
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
    
    # Analyze condition pattern in CRD section.
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
    
    # Score section against rule.
    def _score_section(self, section: CRDSection, rule: EARSRule) -> float:
        """Score a section against a rule."""
        # Regex matches (condition only) + fuzzy similarity
        condition_matches = len(re.findall(rule.normalized_condition, section.content, re.IGNORECASE))
        
        if RAPIDFUZZ_AVAILABLE:
            similarity = fuzz.partial_ratio(section.content.lower(), rule.condition.lower()) / 100.0
        else:
            similarity = difflib.SequenceMatcher(None, section.content.lower(), rule.condition.lower()).ratio()
        
        # Emphasize condition presence; remove response consideration
        return min(condition_matches * 0.6 + similarity * 0.4, 1.0)
    
    # Inject rules into matched sections.
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
        max_injections = 10  # Increased limit for thorough testing
        injections_done = 0
        
        # Calculate total injections needed
        inject_candidates = [m for m in sorted_matches if m['status'] == 'inject']
        print(f"Found {len(sorted_matches)} matches, {len(inject_candidates)} require injection.")
        
        for i, match in enumerate(sorted_matches):
            if match['status'] == 'inject':
                if injections_done >= max_injections:
                    # Respect limit: do not inject beyond the cap
                    match['status'] = 'limit_skipped'
                    match['injected_paragraph'] = match['paragraph']
                    injected_results.append(match)
                    continue
                try:
                    print(f"Processing injection {injections_done + 1}/{len(inject_candidates)} for rule {match['rule_idx']} in {match['crd_file']}...")
                    
                    rule = match['rule']
                    section_context = match['section'].content[:800]  # Include more context for timing analysis
                    
                    # Determine injection strategy based on mutation type
                    if rule.mutation_type != "llm_rewrite":
                        print(f"Applying explicit mutation: {rule.mutation_type}")
                        mutated_text, details, success = MutationEngine.apply_mutation(
                            match['paragraph'], 
                            rule.mutation_type, 
                            rule.condition
                        )
                        match['mutation_details'] = details
                        
                        if success:
                            # Polish the mutated text with LLM
                            rewritten = self.llm_client.polish_text(mutated_text, rule.condition)
                        else:
                            # Fallback to LLM rewrite if mutation failed (e.g. number not found)
                            print(f"Mutation failed ({details}), falling back to LLM rewrite.")
                            match['mutation_details'] += " (Fallback to LLM)"
                            rewritten = self.llm_client.rewrite_with_llm(
                                match['paragraph'],
                                rule.condition,
                                section_context
                            )
                    else:
                        # Legacy behavior
                        match['mutation_details'] = "LLM autonomous rewrite"
                        rewritten = self.llm_client.rewrite_with_llm(
                            match['paragraph'],
                            rule.condition,
                            section_context
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
    
    # Generate output files and patches.
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
    
    
    # Write injection results to Markdown file.
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
                    rule = match['rule']
                    f.write(f"### Rule {match['rule_idx']}: {match['ecu_section']}\n\n")
                    f.write(f"**Status:** {match['status']}\n\n")
                    f.write(f"**Match Score:** {match['match_score']:.3f}\n\n")
                    
                    # EARS Rule Details (Ground Truth)
                    f.write("#### EARS Rule Details (Ground Truth)\n")
                    f.write(f"- **Target Object (O):** {rule.object}\n")
                    f.write(f"- **Defect Condition (C):** {rule.condition}\n")
                    f.write(f"- **Expected Erroneous Response (R_esp):** {rule.response}\n")
                    f.write(f"- **Applied Mutation:** {rule.mutation_type}\n")
                    if 'mutation_details' in match:
                        f.write(f"- **Mutation Details:** {match['mutation_details']}\n")
                    f.write("\n")
                    
                    if 'match_type' in match:
                        f.write(f"**Match Type:** {match['match_type']}\n\n")
                    
                    f.write(f"**Location:** Lines {match['line_span']}\n\n")
                    
                    if match['status'] == 'inject':
                        f.write("**Injected Content:**\n\n")
                        f.write(f"{match['injected_paragraph']}\n\n")
                        
                        f.write("**Original Context:**\n")
                        f.write(f"{match['matched_snippet']}\n\n")
                    else:
                        f.write("**Existing Content:**\n\n")
                        f.write(f"{match['paragraph']}\n\n")
                    
                    f.write("---\n\n")
        
        print(f"Injection results written to: {output_file}")
    
    # Generate patch files and optionally apply them.
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
    
    # Run the complete EARS injection process.
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
        # Print section titles and line numbers
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


# Main entry point function.
def main():
    """Main entry point."""
    # Tee stdout/stderr to output/run.log (assumes output/ already exists)
    try:
        output_root = Path('output')
        log_file_path = output_root / 'run.log'
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, data):
                for s in self.streams:
                    try:
                        s.write(data)
                    except Exception:
                        pass
            def flush(self):
                for s in self.streams:
                    try:
                        s.flush()
                    except Exception:
                        pass
        log_fh = open(log_file_path, 'a', encoding='utf-8')
        sys.stdout = Tee(sys.stdout, log_fh)
        sys.stderr = Tee(sys.stderr, log_fh)
    except Exception:
        # Fallback silently if logging setup fails
        pass
    parser = argparse.ArgumentParser(description="Inject EARS rules into CRD files using local LLM")
    parser.add_argument("--rules", default="EARSrules.txt", help="EARS rules file (default: EARSrules.txt)")
    parser.add_argument("--crd-dir", default="/home/lexi/CRD", help="Directory containing CRD files (default: /home/lexi/CRD)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Match threshold (default: 0.3)")
    parser.add_argument("--similarity-threshold", type=float, default=0.8, help="Similarity threshold for output validation (default: 0.8)")
    parser.add_argument("--apply", action="store_true", help="Apply patches to create patched files")
    parser.add_argument("--md", action="store_true", help="Generate injected.md output")
    parser.add_argument("--secure-cleanup", action="store_true", help="Securely delete output files after processing (for confidential documents)")
    parser.add_argument("--rule-idx", type=int, help="Only use the EARS rule with this index (1-based)")
    parser.add_argument("--section-filter", help="Only consider sections whose title matches this regex (e.g., '^3-1')")
    parser.add_argument("--model", help="LLM model name (e.g., deepseek-r1:14b)")

    args = parser.parse_args()

    try:
        injector = EARSInjector(args.rules, args.threshold, args.similarity_threshold, model_name=args.model)
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
            print(f"Match threshold: {injector.threshold}")
            print(f"Similarity threshold: {injector.similarity_threshold*100:.0f}%")
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
