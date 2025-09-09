
## Quick Start

### Prerequisites
1. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Install and setup Ollama**:
   ```bash
   # Install Ollama (visit https://ollama.ai/ for installation instructions)
   # Pull the required Qwen model (optimized for this project)
   ollama pull qwen3:8b
   ```

3. **Start Ollama service**:
   ```bash
   ollama serve
   ```

### For New Users
1. **Run the demo script** to test the tool with sample data:
   ```bash
   python3 demo.py
   ```
   This interactive demo will guide you through the basic functionality.

2. **CRD File Placement**: 
   - For **demo**: Use the local `Issue-Injection/CRD/` folder (contains sample files for testing)
   - For **real usage**: Place your actual CRD files in `/home/lexi/CRD/` (default production directory)

### Project Structure
```
project-root/
├── Issue-Injection/        # Main project folder
│   ├── inject_ears.py      # Main injection script
│   ├── run_injection.py    # Simplified runner
│   ├── demo.py             # Interactive demo
│   ├── EARSrules.txt       # EARS rules definition
│   ├── requirements.txt    # Python dependencies
│   ├── CRD/                # Sample CRD files (for demo/testing only)
│   │   └── Sample_ECU_Function_Specification.txt
│   └── README.md
└── /home/lexi/CRD/         # Production CRD documents directory (default)
    └── Your_Actual_CRD_Files.txt
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

### LLM Model Settings
- **Model**: Uses `qwen3:8b` for all LLM operations
- **Anti-thinking optimization**: Prompts are optimized to prevent Qwen from outputting thinking processes
- **JSON parsing**: Robust parsing with automatic cleaning of any thinking content

### Path Configuration
- **Demo mode**: Uses local `./CRD` folder for sample files
- **Production mode**: Uses `/home/lexi/CRD` folder (default production directory)
- **Custom paths**: Use `--crd-dir` argument to specify custom CRD location

### Advanced Options
- **Threshold adjustment**: Modify matching sensitivity in the scripts
- **Custom section filtering**: Use `--section-filter` regex patterns
- **Rule selection**: Use `--rule-idx` to test specific rules
- **Output formats**: Generate trace files, patches, or complete modified documents
