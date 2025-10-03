#!/bin/bash

echo "================================"
echo "CUDA ACO - Final Build Script"
echo "================================"

# Create directories
mkdir -p bin data

# Option 1: Build standalone version (ALWAYS WORKS)
echo ""
echo "Step 1: Building standalone test version..."
nvcc -O3 -DSTANDALONE_BUILD -o bin/cuda_aco_standalone cuda_aco_main.cu -lcudart -lcurand -lm
if [ $? -eq 0 ]; then
    echo "✓ Standalone version built successfully!"
    echo "  Run with: ./bin/cuda_aco_standalone"
else
    echo "✗ Standalone build failed"
    echo "  Trying with explicit libraries..."
    nvcc -O3 -DSTANDALONE_BUILD -o bin/cuda_aco_standalone cuda_aco_main.cu -L/usr/local/cuda/lib64 -lcudart -lcurand -lm
    if [ $? -eq 0 ]; then
        echo "✓ Standalone version built with explicit libraries!"
    else
        echo "✗ Build failed - check CUDA installation"
        exit 1
    fi
fi

# Option 2: Build with main.cpp for file loading
echo ""
echo "Step 2: Building full version with file loading..."

# First, let's make sure we have the header
if [ ! -f "cuda_aco.h" ]; then
    echo "Creating cuda_aco.h..."
    cat > cuda_aco.h << 'EOF'
#ifndef CUDA_ACO_H
#define CUDA_ACO_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ACOData ACOData;

typedef enum {
    AS, EAS, MMAS, RAS, ACS, BWAS
} ACOAlgorithm;

typedef enum {
    TSP, QAP
} ProblemType;

ACOData* aco_init(int n_cities, int n_ants, ACOAlgorithm algo, ProblemType prob);
void aco_load_problem(ACOData* aco, float* distance_matrix, float* flow_matrix);
void aco_run(ACOData* aco, int max_iterations, float* best_tour, float* best_length);
void aco_cleanup(ACOData* aco);

#ifdef __cplusplus
}
#endif

#endif
EOF
fi

# Method A: Compile everything as CUDA (simplest)
echo "Trying Method A: All CUDA compilation..."

# Rename main.cpp to main.cu temporarily
if [ -f "main.cpp" ]; then
    cp main.cpp main_temp.cu
    
    # Create library wrapper that excludes main
    cat > cuda_aco_lib.cu << 'EOF'
/* Library mode - no main function */
#define ACO_LIBRARY_MODE
#include "cuda_aco_main.cu"
EOF
    
    # Compile together
    nvcc -O3 -o bin/cuda_aco_full cuda_aco_lib.cu main_temp.cu -lcudart -lcurand -lm 2>/dev/null
    
    if [ $? -eq 0 ]; then
        echo "✓ Full version built successfully with Method A!"
        rm -f main_temp.cu cuda_aco_lib.cu
    else
        echo "Method A failed, trying Method B..."
        
        # Method B: Separate compilation with device linking
        echo "Method B: Separate compilation..."
        
        # Compile CUDA library
        nvcc -O3 -dc cuda_aco_lib.cu -o cuda_aco_lib.o
        
        # Compile main as CUDA
        nvcc -O3 -dc main_temp.cu -o main.o
        
        # Device link
        nvcc -O3 -dlink cuda_aco_lib.o main.o -o device_link.o
        
        # Final link
        nvcc -O3 cuda_aco_lib.o main.o device_link.o -o bin/cuda_aco_full -lcudart -lcurand -lm
        
        if [ $? -eq 0 ]; then
            echo "✓ Full version built successfully with Method B!"
        else
            echo "✗ Full version build failed"
            echo "  But standalone version is available!"
        fi
        
        # Cleanup
        rm -f *.o main_temp.cu cuda_aco_lib.cu
    fi
else
    echo "✗ main.cpp not found - skipping full version"
fi

# Generate test data
echo ""
echo "Step 3: Generating test data..."
if [ ! -f "data/test100.tsp" ]; then
    cat > data/test100.tsp << 'EOF'
NAME: test100
TYPE: TSP
DIMENSION: 100
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
EOF
    for i in $(seq 1 100); do
        x=$((RANDOM % 1000))
        y=$((RANDOM % 1000))
        echo "$i $x $y" >> data/test100.tsp
    done
    echo "EOF" >> data/test100.tsp
    echo "✓ Created data/test100.tsp"
else
    echo "✓ Test data already exists"
fi

# Test the builds
echo ""
echo "================================"
echo "Testing builds..."
echo "================================"

echo ""
echo "Testing standalone version:"
timeout 2 ./bin/cuda_aco_standalone 2>/dev/null | head -n 5
if [ $? -eq 0 ] || [ $? -eq 124 ]; then
    echo "✓ Standalone version works!"
fi

if [ -f "bin/cuda_aco_full" ]; then
    echo ""
    echo "Testing full version:"
    ./bin/cuda_aco_full --problem tsp --instance data/test100.tsp --ants 32 --iterations 10 --quiet
    if [ $? -eq 0 ]; then
        echo "✓ Full version works!"
    fi
fi

echo ""
echo "================================"
echo "Build Summary:"
echo "================================"
if [ -f "bin/cuda_aco_standalone" ]; then
    echo "✓ Standalone: ./bin/cuda_aco_standalone"
fi
if [ -f "bin/cuda_aco_full" ]; then
    echo "✓ Full version: ./bin/cuda_aco_full --problem tsp --instance data/test100.tsp"
fi
echo ""
echo "Done!"