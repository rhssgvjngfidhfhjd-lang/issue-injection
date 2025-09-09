#!/usr/bin/env python3
"""
Run EARS Rule Injection and generate complete modified CRD file
"""

import os
import sys
import argparse
from pathlib import Path
from inject_ears import EARSInjector

def main():
    parser = argparse.ArgumentParser(description="Run EARS Rule Injection and generate complete modified CRD file")
    parser.add_argument("--rules", default="EARSrules.txt", help="EARS rules file (default: EARSrules.txt)")
    parser.add_argument("--crd-dir", default="../CRD", help="Directory containing CRD files (default: ../CRD)")
    parser.add_argument("--output-dir", default="output", help="Output directory (default: output)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Match threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    print("Starting EARS Rule Injection Process...")
    print("=" * 60)
    print(f"Rules file: {args.rules}")
    print(f"CRD directory: {args.crd_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Initialize injector
    injector = EARSInjector(rules_file=args.rules, threshold=args.threshold)
    
    # Run injection process
    print("Scanning CRD files...")
    crd_files = injector.scan_crd_files(args.crd_dir)
    
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
    injector.generate_outputs(injected_matches, args.output_dir, apply_patches=True)
    
    # Generate complete modified CRD file
    print("Generating complete modified CRD file...")
    generate_complete_crd(crd_files, injected_matches, args.output_dir)
    
    print("\nEARS injection complete!")
    print("Check the following files:")
    print(f"- {args.output_dir}/injected.md: Injection results and context")
    print(f"- {args.output_dir}/_patched/: Complete modified CRD files")
    print(f"- {args.output_dir}/patches/: Patch files showing changes")

def generate_complete_crd(crd_files, injected_matches, output_dir):
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
        output_file = Path(output_dir) / "_patched" / filename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print(f"  Generated complete modified file: {output_file}")
        print(f"  Applied {changes_applied} injections")

if __name__ == "__main__":
    main()
