# Issue Injection

## Project Overview
Issue Injection is a specialized tool for injecting EARS (Easy Approach to Requirements Syntax) rules into automotive Component Requirement Documents (CRD). The tool utilizes a deterministic Mutation Engine ($\Phi_{mutate}$) and a local LLM engine to perform controlled document corruption, enabling verifiable "ground truth" for requirement verification systems.

## Core Features
- **Deterministic Mutation Engine ($\Phi_{mutate}$)**: Implements explicit mutation operators including numeric perturbation, procedural step reordering, and action omission.
- **EARS Rule Formalization**: Supports the structured rule format $R = \langle O, C, R_{esp} \rangle$ (Object, Condition, Expected Response) for precise requirement mapping.
- **Ground Truth Traceability**: Automatically maps injected defects back to the original EARS rule components (O/C/R_esp) in the output reports.
- **Hybrid Processing**: Combines programmatic mutation for logic control with LLM polishing for linguistic fluency, ensuring document quality without losing logic precision.
- **GPU-Accelerated Inference**: Optimized for NVIDIA A100 hardware using local Ollama service and custom port binding.
- **CRD Structure Recognition**: Automated identification of ECU components and operational conditions via regex-based scanning.

## Quick Start

### Prerequisites
- **OS**: Linux (Ubuntu 22.04+ recommended)
- **Python**: 3.10+
- **LLM Runtime**: Ollama (configured for GPU access)
- **Hardware**: NVIDIA GPU (A100 80GB recommended)

### Installation
1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start GPU-Enabled LLM Service**
   Use the provided startup script to link CUDA libraries and bind the service to port 11435:
   ```bash
   bash start_ollama_gpu.sh
   ```

## Usage

### Rule Configuration
Define rules in `EARSrules.txt` using the structured format for maximum control:
```text
O: ECU_A; C: IF ECU_A sends request 1; R_esp: THEN ECU_B shall reject; MUTATION: numeric_perturbation
```
*Note: Legacy "IF-THEN" format is still supported but will default to autonomous LLM rewriting.*

### Execution
Run the main injection process:
```bash
python3 main.py --rules EARSrules.txt --crd-dir ./CRD --model qwen3:8b
```

### Environment Configuration
The tool defaults to port 11435 for the GPU-bound Ollama instance. You can override this via environment variables:
```bash
export OLLAMA_HOST=http://localhost:11435
```

### Call Local Ollama from Terminal
Use `call_ollama.py` to ask questions in the terminal; it calls local `qwen2.5-coder:32b` by default and prints the answer.

**Single question (positional):**
```bash
python3 call_ollama.py "你的问题"
```

**Pipe question from stdin:**
```bash
echo "你的问题" | python3 call_ollama.py
```

**Custom port (e.g. default Ollama 11434):**
```bash
python3 call_ollama.py --base-url http://127.0.0.1:11434 "你的问题"
```

Optional: `--stream` for streaming output, `--model` to use another model.

## Configuration Parameters
| Parameter | Description | Default |
| :--- | :--- | :--- |
| `--rules` | Path to EARS rules definition file | `EARSrules.txt` |
| `--crd-dir` | Directory containing source CRD documents | `./CRD` |
| `--output-dir` | Directory for generated results | `./output_run` |
| `--model` | LLM model name used for text polishing | `qwen3:8b` |
| `--threshold` | Minimum score threshold for rule matching | `0.3` |

## Output Specification
Upon completion, the output directory contains:
- **injected.md**: Comprehensive report including **Ground Truth Mapping** (O/C/R_esp), applied mutation types, and before/after comparisons.
- **patches/**: Standard `.patch` files recording the specific changes made to each document.
- **_patched/**: Full document copies with mutations applied for direct integration or testing.

## Notes
- **Data Security**: All processing is strictly local. No requirement data or document content is transmitted outside the host machine.
- **Logic Integrity**: Programmatic mutation ensures that core technical defects (e.g., timing changes, step omissions) are deterministic and reproducible.
- **Performance**: On A100 hardware, each injection cycle takes approximately 2-5 seconds. Ensure the Ollama service is fully initialized before running the main script.

## 
python3 main.py --crd-dir ./CRD --output-dir output
