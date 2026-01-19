#!/bin/bash
# Start Ollama with GPU support on port 11435

# Set CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/targets/x86_64-linux/lib:/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}

# Set CUDA visible devices (use GPU 0 by default, or all GPUs)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}

# Set Ollama to use GPU
export OLLAMA_NUM_GPU=${OLLAMA_NUM_GPU:-3}

# Start Ollama on port 11435
echo "Starting Ollama with GPU support on port 11435..."
echo "CUDA Library Path: $LD_LIBRARY_PATH"
echo "CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "Ollama Num GPU: $OLLAMA_NUM_GPU"

OLLAMA_HOST=0.0.0.0:11435 ollama serve
