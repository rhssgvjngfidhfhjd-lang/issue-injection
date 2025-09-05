
## Quick Start

### For New Users
1. **Run the demo script** to test the tool with sample data:
   ```bash
   python3 demo.py
   ```
   This interactive demo will guide you through the basic functionality.

2. **Check the CRD directory**: The `CRD/` folder contains a sample CRD file for testing. 
   In real usage, replace this with your actual CRD documents.

### File Structure
```
Issue-Injection/
├── inject_ears.py          # Main injection script
├── run_injection.py        # Simplified runner
├── demo.py                 # Interactive demo
├── EARSrules              # EARS rules definition
├── CRD/                   # CRD documents directory
│   └── Sample_ECU_Function_Specification.txt
└── README.md
```

## Advanced Usage

### Document Parsing Features

- The script performs intelligent document parsing based on numbered headings and section titles:
  - Supports `1-1`, `1-1-1`, `1.1`, `1.1.1` numbering styles
  - Requires space and title text after section numbers
  - Preserves uppercase module titles and `#` Markdown headings
- Automatic Table of Contents (TOC) filtering:
  - Filters out TOC/front matter based on dot leaders, short content, and numbering patterns
  - Starts processing from the first substantial numbered section (usually chapter 1)
- All parsing and matching happens in memory - no temporary files created
- Section titles and line ranges are displayed during processing for verification

### Example Output During Processing

```text
Finding matches between rules and CRD sections...
Total sections: 16
Sections in Sample_ECU_Function_Specification.txt:
- [15-20] 1. Document Information
- [25-35] 2. System Overview  
- [40-60] 3. Functional Specifications
- [65-95] 3-1. Gateway Function Control
...
```

## Configuration

- **Threshold adjustment**: Modify matching sensitivity in the scripts
- **Custom section filtering**: Use `--section-filter` regex patterns
- **Rule selection**: Use `--rule-idx` to test specific rules
- **Output formats**: Generate trace files, patches, or complete modified documents
