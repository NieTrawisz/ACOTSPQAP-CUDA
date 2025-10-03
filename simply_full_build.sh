#!/bin/bash

# Simple build script for CUDA ACO

# Detect GPU architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    GPU_ARCH="sm_${GPU_CC}"
    echo "Detected GPU architecture: ${GPU_ARCH}"
    ARCH_FLAG="-arch=${GPU_ARCH}"
else
    echo "Warning: nvidia-smi not found, letting nvcc auto-detect GPU"
    ARCH_FLAG=""
fi

echo "Building full version with nvcc..."
# Create the library interface file with proper defines
echo "Creating cuda_aco_lib.cu..."
cat > cuda_aco_lib.cu << 'EOLIB'
/* CUDA ACO Library Interface - DO NOT INCLUDE main.cu! */
#define ACO_LIBRARY_MODE
#include "cuda_aco_main.cu"
EOLIB
        
mkdir -p bin
# Compile separately then link
echo "Compiling CUDA library..."
nvcc ${ARCH_FLAG} -O3 -use_fast_math -c -o cuda_aco_lib.o cuda_aco_lib.cu
echo "Compiling main.cu..."
nvcc ${ARCH_FLAG} -O3 -c -o main.o main.cu
echo "Linking..."
nvcc ${ARCH_FLAG} -O3 -o bin/cuda_aco cuda_aco_lib.o main.o -lcudart -lcurand -lm

# Clean up object files
rm -f cuda_aco_lib.o main.o

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./bin/cuda_aco"
else
    echo "Build failed!"
    exit 1
fi
