#!/usr/bin/env python3
"""
Demo script for EARS Rule Injection Tool
This script demonstrates how to use the EARS injection tool with the sample CRD file.
"""

import os
import sys
from pathlib import Path

def main():
    print("EARS Rule Injection Demo")
    print("=" * 40)
    
    # Check if required files exist
    required_files = ['EARSrules', 'inject_ears.py', 'CRD/Sample_ECU_Function_Specification.txt']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present before running the demo.")
        return 1
    
    print("✅ All required files found")
    print("\nDemo Options:")
    print("1. Quick test (scan and match only)")
    print("2. Full injection (with LLM processing)")
    print("3. Show available EARS rules")
    print("4. Show CRD file structure")
    
    try:
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            run_quick_test()
        elif choice == "2":
            run_full_injection()
        elif choice == "3":
            show_ears_rules()
        elif choice == "4":
            show_crd_structure()
        else:
            print("Invalid choice. Please run the demo again.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nDemo cancelled by user.")
        return 0
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        return 1
    
    return 0

def run_quick_test():
    """Run a quick test to show matching without LLM processing"""
    print("\n🔍 Running quick test (scan and match only)...")
    
    try:
        from inject_ears import EARSInjector
        
        # Initialize injector
        injector = EARSInjector('EARSrules', threshold=0.3)
        print(f"✅ Loaded {len(injector.rules)} EARS rules")
        
        # Scan CRD files
        crd_files = injector.scan_crd_files('CRD')
        print(f"✅ Found {len(crd_files)} CRD files")
        
        # Find matches
        matches = injector.find_matches(crd_files)
        print(f"✅ Found {len(matches)} potential matches")
        
        if matches:
            print("\n📋 Match Summary:")
            for i, match in enumerate(matches[:3], 1):  # Show first 3 matches
                print(f"{i}. Rule {match['rule_idx']}: {match['ecu_section']}")
                print(f"   Score: {match['match_score']:.3f}, Status: {match['status']}")
        else:
            print("ℹ️  No matches found with current threshold")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed.")

def run_full_injection():
    """Run full injection with LLM processing"""
    print("\n🚀 Running full injection (requires LLM)...")
    print("⚠️  Note: This requires a running Ollama instance with llama3.3:latest model")
    
    confirm = input("Continue? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Demo cancelled.")
        return
    
    try:
        from inject_ears import EARSInjector
        
        injector = EARSInjector('EARSrules', threshold=0.3)
        crd_files = injector.scan_crd_files('CRD')
        matches = injector.find_matches(crd_files)
        
        if matches:
            print(f"Processing {len(matches)} matches...")
            injected_matches = injector.inject_rules(matches)
            injector.generate_outputs(injected_matches, ".", apply_patches=False)
            print("✅ Injection complete! Check 'issue_injection_trace.txt' for results.")
        else:
            print("ℹ️  No matches found to process.")
            
    except Exception as e:
        print(f"❌ Error during injection: {e}")

def show_ears_rules():
    """Display available EARS rules"""
    print("\n📜 Available EARS Rules:")
    print("-" * 50)
    
    try:
        with open('EARSrules', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Truncate long rules for display
                    display_rule = line[:80] + "..." if len(line) > 80 else line
                    print(f"{i:2d}. {display_rule}")
    except Exception as e:
        print(f"❌ Error reading EARS rules: {e}")

def show_crd_structure():
    """Display CRD file structure"""
    print("\n📁 CRD File Structure:")
    print("-" * 30)
    
    try:
        from inject_ears import CRDFile
        
        crd_file = CRDFile(Path('CRD/Sample_ECU_Function_Specification.txt'))
        print(f"File: {crd_file.file_path.name}")
        print(f"Sections found: {len(crd_file.sections)}")
        print("\nSection breakdown:")
        for i, section in enumerate(crd_file.sections, 1):
            print(f"{i:2d}. [{section.start_line:3d}-{section.end_line:3d}] {section.name}")
            
    except Exception as e:
        print(f"❌ Error analyzing CRD structure: {e}")

if __name__ == "__main__":
    sys.exit(main())
