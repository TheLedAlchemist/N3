#!/bin/bash

# Define the number of seeds and the number of parallel jobs
NUM_SEEDS=2
PARALLEL_JOBS=20
PLATFORM="cpu"

# Define the paths to the Python scripts
SPIRAL_STATIC_SCRIPT="./scripts/spiral_static.py"
SPIRAL_STANDARD_SCRIPT="./scripts/spiral_standard.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

# Function to run a Python script with a given seed and output directory
run_script() {
    local script=$1
    local seed=$2
    local output_dir=$3
    mkdir -p $output_dir
    JAX_PLATFORM_NAME=$PLATFORM python $script --seed $seed --out_path $output_dir
}

export -f run_script

# Run the scripts for each seed in parallel
for seed in $(seq 0 $((NUM_SEEDS - 1))); do
    static_output_dir="${OUTPUT_BASE_DIR}/SPIRAL_STATIC/seed_${seed}/"
    standard_output_dir="${OUTPUT_BASE_DIR}/SPIRAL_STANDARD/seed_${seed}/"

    run_script $SPIRAL_STATIC_SCRIPT $seed $static_output_dir &
    run_script $SPIRAL_STANDARD_SCRIPT $seed $standard_output_dir &

    # Wait for the parallel jobs to finish before starting new ones
    if (( $((seed % PARALLEL_JOBS)) == 0 )); then
        echo "Waiting for current jobs to finish..."
        wait
    fi
done

# Wait for any remaining jobs to finish
wait

echo "All jobs completed."