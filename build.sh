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

# Build options
echo "Building CUDA ACO..."
echo "1) Simple version (standalone)"
echo "2) Full version with C++ interface"
echo "3) Debug version"
read -p "Select option (1-3): " option

case $option in
    1)
        echo "Building simple version..."
        nvcc ${ARCH_FLAG} -O3 -use_fast_math -DSTANDALONE_BUILD -o bin/cuda_aco cuda_aco_main.cu
        ;;
    2)
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
        ;;
    3)
        echo "Building debug version..."
        nvcc ${ARCH_FLAG} -G -g -DSTANDALONE_BUILD -o bin/cuda_aco_debug cuda_aco_main.cu
        ;;
    *)
        echo "Invalid option"
        exit 1
        ;;
esac

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Run with: ./bin/cuda_aco"
else
    echo "Build failed!"
    exit 1
fi
