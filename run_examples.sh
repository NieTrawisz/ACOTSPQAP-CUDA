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
