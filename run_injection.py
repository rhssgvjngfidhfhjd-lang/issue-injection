#!/usr/bin/env python3
"""
Run EARS Rule Injection and generate complete modified CRD file
"""

import os
import sys
from pathlib import Path
from inject_ears import EARSInjector

def main():
    print("Starting EARS Rule Injection Process...")
    print("=" * 60)
    
    # Initialize injector
    injector = EARSInjector(rules_file="EARSrules", threshold=0.3)
    
    # Run injection process
    print("Scanning CRD files...")
    crd_files = injector.scan_crd_files("CRD")
    
    if not crd_files:
        print("No CRD files found in CRD directory!")
        return
    
    print(f"Found {len(crd_files)} CRD files")
    
    # Find matches
    print("Finding matches between rules and CRD sections...")
    matches = injector.find_matches(crd_files)
    print(f"Found {len(matches)} matches")
    
    # Inject rules
    print("Injecting rules using LLM...")
    injected_matches = injector.inject_rules(matches)
    print(f"Processed {len(injected_matches)} matches")
    
    # Generate outputs
    print("Generating output files...")
    injector.generate_outputs(injected_matches, ".", apply_patches=True)
    
    # Generate complete modified CRD file
    print("Generating complete modified CRD file...")
    generate_complete_crd(crd_files, injected_matches)
    
    print("\nEARS injection complete!")
    print("Check the following files:")
    print("- matches.csv: Detailed match information")
    print("- injected.md: Injection results and context")
    print("- _patched/: Complete modified CRD files")
    print("- patches/: Patch files showing changes")

def generate_complete_crd(crd_files, injected_matches):
    """Generate complete modified CRD files with all injections applied."""
    
    # Group matches by file
    files_matches = {}
    for match in injected_matches:
        filename = match['crd_file']
        if filename not in files_matches:
            files_matches[filename] = []
        files_matches[filename].append(match)
    
    # Process each CRD file
    for crd_file in crd_files:
        filename = crd_file.file_path.name
        if filename not in files_matches:
            continue
        
        print(f"Processing {filename}...")
        
        # Read original content
        with open(crd_file.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Apply all injections for this file
        modified_content = content
        changes_applied = 0
        
        for match in files_matches[filename]:
            if match['status'] == 'inject':
                original_para = match['paragraph']
                new_para = match['injected_paragraph']
                
                if original_para != new_para:
                    # Find and replace paragraph
                    para_start = modified_content.find(original_para)
                    if para_start != -1:
                        modified_content = (
                            modified_content[:para_start] + 
                            new_para + 
                            modified_content[para_start + len(original_para):]
                        )
                        changes_applied += 1
                        print(f"  Applied injection for Rule {match['rule_idx']}")
                    else:
                        print(f"  Warning: Could not find original paragraph for Rule {match['rule_idx']}")
        
        # Write complete modified file
        output_file = Path("_patched") / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  Generated complete modified file: {output_file}")
        print(f"  Applied {changes_applied} injections")

if __name__ == "__main__":
    main()
