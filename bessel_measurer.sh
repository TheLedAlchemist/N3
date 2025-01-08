#!/bin/bash

# Define the number of seeds and the number of parallel jobs
NUM_SEEDS=1
PARALLEL_JOBS=20
PLATFORM="cpu"

# Define the paths to the Python scripts
BESSEL_STATIC_SCRIPT="./scripts/bessel_static.py"
BESSEL_STANDARD_SCRIPT="./scripts/bessel_standard.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

# Function to run a Python script with a given seed and output directory
run_script() {
    local script=$1
    local seed=$2
    local output_dir=$3
    local lambda=$4
    mkdir -p $output_dir
    JAX_PLATFORM_NAME=$PLATFORM python $script --seed $seed --out_path $output_dir --size_influence $lambda
}


# Run the scripts for each seed in parallel
static_output_dir="${OUTPUT_BASE_DIR}/BESSEL/"
standard_output_dir="${OUTPUT_BASE_DIR}/BESSEL/"

for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    for i in $(seq 0.01 0.01 1.1); do
	run_script $BESSEL_STATIC_SCRIPT $seed $static_output_dir $i
	run_script $BESSEL_STANDARD_SCRIPT $seed $standard_output_dir $i
    done
done
