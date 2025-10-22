#!/usr/bin/env python3
"""
Demo script for EARS Rule Injection Tool
This script demonstrates how to use the EARS injection tool with the sample CRD file.
Note: This demo uses the local CRD/ folder for demonstration. 
For actual usage, CRD files should be placed in ../CRD/ (same level as Issue-Injection folder).
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Demo script for EARS Rule Injection Tool")
    parser.add_argument("--rules", default="EARSrules.txt", help="EARS rules file (default: EARSrules.txt)")
    parser.add_argument("--crd-dir", default="CRD", help="Directory containing CRD files (default: CRD)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Match threshold (default: 0.3)")
    
    args = parser.parse_args()
    
    print("EARS Rule Injection Demo")
    print("=" * 40)
    print(f"Rules file: {args.rules}")
    print(f"CRD directory: {args.crd_dir}")
    print(f"Threshold: {args.threshold}")
    print()
    
    # Check if required files exist
    required_files = [args.rules, 'inject_ears.py', f'{args.crd_dir}/Sample_ECU_Function_Specification.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running the demo.")
        return 1
    
    print("All required files found")
    print("\nDemo Options:")
    print("1. Quick test (scan and match only)")
    print("2. Full injection (with LLM processing)")
    print("3. Show available EARS rules")
    print("4. Show CRD file structure")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_test(args.rules, args.crd_dir, args.threshold)
        elif choice == "2":
            run_full_injection(args.rules, args.crd_dir, args.threshold)
        elif choice == "3":
            show_ears_rules(args.rules)
        elif choice == "4":
            show_crd_structure(args.crd_dir)
        else:
            print("Invalid choice. Please run the demo again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        return 0
    except Exception as e:
        print(f"\nError during demo: {e}")
        return 1
    
    return 0

def run_quick_test(rules_file, crd_dir, threshold):
    """Run a quick test to show matching without LLM processing"""
    print("\nRunning quick test (scan and match only)...")
    
    try:
        from inject_ears import EARSInjector
        
        # Initialize injector
        injector = EARSInjector(rules_file, threshold=threshold)
        print(f"Loaded {len(injector.rules)} EARS rules")
        
        # Scan CRD files
        crd_files = injector.scan_crd_files(crd_dir)
        print(f"Found {len(crd_files)} CRD files")
        
        # Find matches
        matches = injector.find_matches(crd_files)
        print(f"Found {len(matches)} potential matches")
        
        if matches:
            print("\nMatch Summary:")
            for i, match in enumerate(matches[:3], 1):  # Show first 3 matches
                print(f"{i}. Rule {match['rule_idx']}: {match['ecu_section']}")
                print(f"   Score: {match['match_score']:.3f}, Status: {match['status']}")
        else:
            print("No matches found with current threshold")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all dependencies are installed.")

def run_full_injection(rules_file, crd_dir, threshold):
    """Run full injection with LLM processing"""
    print("\nRunning full injection (requires LLM)...")
    print("Note: This requires OpenAI API access")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Demo cancelled.")
        return
    
    try:
        from inject_ears import EARSInjector
        
        injector = EARSInjector(rules_file, threshold=threshold)
        crd_files = injector.scan_crd_files(crd_dir)
        matches = injector.find_matches(crd_files)
        
        if matches:
            print(f"Processing {len(matches)} matches...")
            injected_matches = injector.inject_rules(matches)
            injector.generate_outputs(injected_matches, ".", apply_patches=False)
            print("Injection complete! Check 'issue_injection_trace.txt' for results.")
        else:
            print("No matches found to process.")
            
    except Exception as e:
        print(f"Error during injection: {e}")

def show_ears_rules(rules_file):
    """Display available EARS rules"""
    print("\nAvailable EARS Rules:")
    print("-" * 50)
    
    try:
        with open(rules_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Truncate long rules for display
                    display_rule = line[:80] + "..." if len(line) > 80 else line
                    print(f"{i:2d}. {display_rule}")
    except Exception as e:
        print(f"Error reading EARS rules: {e}")

def show_crd_structure(crd_dir):
    """Display CRD file structure"""
    print("\nCRD File Structure:")
    print("-" * 30)
    
    try:
        from inject_ears import CRDFile
        
        crd_file = CRDFile(Path(f'{crd_dir}/Sample_ECU_Function_Specification.txt'))
        print(f"File: {crd_file.file_path.name}")
        print(f"Sections found: {len(crd_file.sections)}")
        print("\nSection breakdown:")
        for i, section in enumerate(crd_file.sections, 1):
            print(f"{i:2d}. [{section.start_line:3d}-{section.end_line:3d}] {section.name}")
            
    except Exception as e:
        print(f"Error analyzing CRD structure: {e}")

if __name__ == "__main__":
    sys.exit(main())
