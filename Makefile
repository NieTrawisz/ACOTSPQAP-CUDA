# Makefile for CUDA ACO Implementation
# Supports both TSP and QAP problems

# CUDA installation path (adjust if needed)
CUDA_PATH ?= /usr/local/cuda
CUDA_INC = -I$(CUDA_PATH)/include
CUDA_LIB = -L$(CUDA_PATH)/lib64

# Compiler and flags
NVCC = nvcc
CXX = g++
CUDA_FLAGS = -O3 -use_fast_math --extended-lambda -Xcompiler -fopenmp
CXX_FLAGS = -O3 -march=native -fopenmp -std=c++14 $(CUDA_INC)
LDFLAGS = $(CUDA_LIB) -lcudart -lcurand -lm -lgomp

# CUDA architecture (adjust based on your GPU)
# sm_70 for V100, sm_75 for T4/RTX 2080, sm_80 for A100, sm_86 for RTX 3090
GPU_ARCH ?= sm_70

# Directories
BUILD_DIR = build
BIN_DIR = bin

# Source files (assuming they're in current directory)
# If you have a src/ directory, uncomment the next line and comment the current SRCS
# SRCS = src/cuda_aco_lib.cu src/main.cu
SRCS = cuda_aco_lib.cu main.cu

# Object files
OBJS = $(BUILD_DIR)/cuda_aco_lib.o $(BUILD_DIR)/main.o

# Target executable
TARGET = $(BIN_DIR)/cuda_aco

# Default target
all: dirs $(TARGET)

# Create directories
dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

# Link executable - use nvcc for linking to handle CUDA dependencies
$(TARGET): $(OBJS)
	@echo "Linking: $@"
	$(NVCC) -arch=$(GPU_ARCH) $(CUDA_FLAGS) -o $@ $^ $(LDFLAGS)

# Compile CUDA source file (library version without main)
$(BUILD_DIR)/cuda_aco_lib.o: cuda_aco_lib.cu main.cu cuda_aco_kernels.cuh
	@echo "Compiling CUDA library: $<"
	$(NVCC) -arch=$(GPU_ARCH) $(CUDA_FLAGS) -dc $< -o $@

# Compile C++ source file
$(BUILD_DIR)/main.o: main.cu
	@echo "Compiling C++: $<"
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# Alternative: compile as a single CUDA file (simpler approach)
simple: dirs
	@echo "Building simple version..."
	$(NVCC) -arch=$(GPU_ARCH) $(CUDA_FLAGS) -DSTANDALONE_BUILD -o $(BIN_DIR)/cuda_aco_simple main.cu $(LDFLAGS)

# Generate test data
test-data:
	@mkdir -p data
	@echo "Generating test TSP instance..."
	@echo "NAME: test100" > data/test100.tsp
	@echo "TYPE: TSP" >> data/test100.tsp
	@echo "DIMENSION: 100" >> data/test100.tsp
	@echo "EDGE_WEIGHT_TYPE: EUC_2D" >> data/test100.tsp
	@echo "NODE_COORD_SECTION" >> data/test100.tsp
	@for i in `seq 1 100`; do \
		echo "$$i $$((RANDOM % 1000)) $$((RANDOM % 1000))" >> data/test100.tsp; \
	done
	@echo "EOF" >> data/test100.tsp

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Run basic test
test: simple test-data
	./$(BIN_DIR)/cuda_aco_simple

# Run with example (after creating proper main.cu integration)
run: $(TARGET) test-data
	./$(TARGET) --problem tsp --instance data/test100.tsp --ants 128 --iterations 100

# Profile with nvprof
profile: simple test-data
	nvprof --print-gpu-trace ./$(BIN_DIR)/cuda_aco_simple

# Profile with Nsight Compute
nsight: simple test-data
	ncu --set full -o profile ./$(BIN_DIR)/cuda_aco_simple

# Check CUDA installation
check-cuda:
	@echo "CUDA Compiler version:"
	@nvcc --version
	@echo ""
	@echo "GPU Information:"
	@nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv

# Help target
help:
	@echo "CUDA ACO Makefile targets:"
	@echo "  all          - Build the complete project"
	@echo "  simple       - Build simple standalone version"
	@echo "  clean        - Remove build artifacts"
	@echo "  test-data    - Generate test TSP instance"
	@echo "  test         - Run basic test"
	@echo "  run          - Run with example parameters"
	@echo "  profile      - Profile with nvprof"
	@echo "  nsight       - Profile with Nsight Compute"
	@echo "  check-cuda   - Check CUDA installation"
	@echo "  help         - Show this help message"
	@echo ""
	@echo "Build for specific GPU architecture:"
	@echo "  make GPU_ARCH=sm_75  # For RTX 2080/T4"
	@echo "  make GPU_ARCH=sm_80  # For A100"
	@echo "  make GPU_ARCH=sm_86  # For RTX 3090"
	@echo "  make GPU_ARCH=sm_89  # For RTX 4090"

.PHONY: all clean dirs test test-data run profile nsight check-cuda help simple