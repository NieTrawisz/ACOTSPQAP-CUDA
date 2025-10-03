#!/bin/bash

# CUDA ACO Project Setup Script
# This script organizes the project files and prepares for compilation

echo "=== CUDA ACO Project Setup ==="
echo "Setting up project structure..."

# Create directory structure
mkdir -p src
mkdir -p include
mkdir -p data
mkdir -p build
mkdir -p bin
mkdir -p scripts
mkdir -p results

# Move or copy source files to appropriate directories
# (Assuming files are in current directory)

# If cuda_aco_main.cu exists in current directory, keep it there for now
# Or move to src/ if you prefer
if [ -f "cuda_aco_main.cu" ]; then
    echo "Found cuda_aco_main.cu"
    # Uncomment next line to move to src/
    # mv cuda_aco_main.cu src/
fi

if [ -f "cuda_aco_kernels.cuh" ]; then
    echo "Found cuda_aco_kernels.cuh"
    # You can move this to include/ if you prefer
    # mv cuda_aco_kernels.cuh include/
fi

if [ -f "main.cpp" ]; then
    echo "Found main.cpp"
    # Uncomment to move to src/
    # mv main.cpp src/
fi

# Download sample TSP instances from TSPLIB (optional)
echo "Downloading sample TSP instances..."
cd data

# Small instances for testing
for instance in "ulysses16" "ulysses22" "att48" "kroA100" "kroB100"; do
    if [ ! -f "${instance}.tsp" ]; then
        echo "Downloading ${instance}..."
        wget -q "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/${instance}.tsp.gz" 2>/dev/null
        if [ -f "${instance}.tsp.gz" ]; then
            gunzip "${instance}.tsp.gz"
            echo "  Downloaded ${instance}.tsp"
        else
            echo "  Could not download ${instance}.tsp"
        fi
    fi
done

# Generate a test instance if downloads failed
if [ ! -f "test100.tsp" ]; then
    echo "Generating test100.tsp..."
    cat > test100.tsp << EOF
NAME: test100
TYPE: TSP
DIMENSION: 100
EDGE_WEIGHT_TYPE: EUC_2D
NODE_COORD_SECTION
EOF
    for i in $(seq 1 100); do
        x=$((RANDOM % 1000))
        y=$((RANDOM % 1000))
        echo "$i $x $y" >> test100.tsp
    done
    echo "EOF" >> test100.tsp
fi

cd ..

# Create a simple build script
cat > build.sh << 'EOF'
#!/bin/bash

# Simple build script for CUDA ACO

# Detect GPU architecture
GPU_ARCH=""
if command -v nvidia-smi &> /dev/null; then
    GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
    GPU_ARCH="sm_${GPU_CC}"
    echo "Detected GPU architecture: ${GPU_ARCH}"
else
    echo "Warning: nvidia-smi not found, using default sm_70"
    GPU_ARCH="sm_70"
fi

# Build options
echo "Building CUDA ACO..."
echo "1) Simple version (standalone)"
echo "2) Full version with C++ interface"
echo "3) Debug version"
read -p "Select option (1-3): " option

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
/* CUDA ACO Library Interface - DO NOT INCLUDE main.cpp! */
#define ACO_LIBRARY_MODE
#include "cuda_aco_main.cu"
EOLIB
        
        mkdir -p bin
        # Compile separately then link
        echo "Compiling CUDA library..."
        nvcc ${ARCH_FLAG} -O3 -use_fast_math -c -o cuda_aco_lib.o cuda_aco_lib.cu
        echo "Compiling main.cpp..."
        nvcc ${ARCH_FLAG} -O3 -c -o main.o main.cpp
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
EOF

chmod +x build.sh

# Create a run script with examples
cat > run_examples.sh << 'EOF'
#!/bin/bash

# Example runs for CUDA ACO

BIN="./bin/cuda_aco"

if [ ! -f "$BIN" ]; then
    echo "Error: $BIN not found. Please build first with ./build.sh"
    exit 1
fi

echo "=== CUDA ACO Examples ==="
echo ""

# Function to run and time
run_test() {
    local desc=$1
    shift
    echo "Test: $desc"
    echo "Command: $BIN $@"
    time $BIN "$@"
    echo "---"
    echo ""
}

# Small instance with different algorithms
if [ -f "data/test100.tsp" ]; then
    run_test "AS on 100 cities" --problem tsp --instance data/test100.tsp --algorithm as --ants 100 --iterations 500
    run_test "ACS on 100 cities" --problem tsp --instance data/test100.tsp --algorithm acs --ants 100 --iterations 500
    run_test "MMAS on 100 cities" --problem tsp --instance data/test100.tsp --algorithm mmas --ants 100 --iterations 500
fi

# Parameter tuning example
if [ -f "data/kroA100.tsp" ]; then
    echo "Parameter sensitivity test on kroA100:"
    for alpha in 0.5 1.0 1.5 2.0; do
        for beta in 2.0 3.0 4.0 5.0; do
            echo "alpha=$alpha, beta=$beta"
            $BIN --problem tsp --instance data/kroA100.tsp --algorithm acs \
                 --ants 128 --iterations 200 --alpha $alpha --beta $beta --quiet
        done
    done
fi
EOF

chmod +x run_examples.sh

# Create basic test to verify CUDA is working
cat > test_cuda.cu << 'EOF'
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("Found %d CUDA device(s)\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024 * 1024));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
    }
    
    return 0;
}
EOF

# Compile and run CUDA test
echo "Checking CUDA installation..."
nvcc -o test_cuda test_cuda.cu 2>/dev/null
if [ $? -eq 0 ]; then
    ./test_cuda
    rm test_cuda
else
    echo "Warning: Could not compile CUDA test. Please check CUDA installation."
fi
rm -f test_cuda.cu

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Project structure:"
echo "  ./ - Source files and headers"
echo "  data/ - TSP/QAP instances"
echo "  build/ - Object files"
echo "  bin/ - Executables"
echo "  results/ - Output files"
echo ""
echo "Next steps:"
echo "  1. Build the project: ./build.sh"
echo "  2. Run tests: make test"
echo "  3. Run examples: ./run_examples.sh"
echo ""
echo "For custom build options, edit Makefile or use:"
echo "  make GPU_ARCH=sm_XX  (where XX is your GPU architecture)"
echo ""
echo "Use 'make help' to see all available targets"