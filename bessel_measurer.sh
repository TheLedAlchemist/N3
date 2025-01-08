#!/bin/bash

#BSUB -n 10
#BSUB -W 720
#BSUB -J besselTest
#BSUB -o stdout.%J
#BSUB -e stderr.%J

# Define the number of seeds and the number of parallel jobs
NUM_SEEDS=100
MAX_LAMBDA=100.00
PLATFORM="cpu"

# Define the paths to the Python scripts
BESSEL_STATIC_SCRIPT="./scripts/bessel_static.py"
BESSEL_STANDARD_SCRIPT="./scripts/bessel_standard.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
echo "Set the python path to: $PYTHONPATH"

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
    echo "Begin training on seed $seed"
    for i in $(seq 0.01 0.01 $MAX_LAMBDA); do
	run_script $BESSEL_STATIC_SCRIPT $seed $static_output_dir $i
	run_script $BESSEL_STANDARD_SCRIPT $seed $standard_output_dir $i
    done
    echo "Finished training seed $seed"
done

echo "JOB COMPLETE"
