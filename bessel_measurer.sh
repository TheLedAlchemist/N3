#!/bin/bash

#BSUB -n 100
#BSUB -W 900
#BSUB -J besselTest
#BSUB -o stdout.%J
#BSUB -e stderr.%J

# Define the number of seeds and the number of parallel jobs
NUM_SEEDS=50
PLATFORM="cpu"
MAX_PARALLEL=100

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
static_output_dir="${OUTPUT_BASE_DIR}/BESSEL"
standard_output_dir="${OUTPUT_BASE_DIR}/BESSEL"

for i in $(seq -2.0 0.01 2.0); do
    echo "Begin training on seed $seed"
    
    static_out_seed="$static_output_dir/SEED_${seed}/"
    standard_out_seed="$standard_output_dir/SEED_${seed}/"

    # Define lambda logarithmically in terms of i
    size_influence=$(awk "BEGIN {print 10^($i)}")

    # Scan through lambda logarithmically from 1e-2, 1e-1.99, ... , 0, ... , 1e+1.99, 1e+2
    for seed in $(seq 0 $((NUM_SEEDS - 1)); do
        run_script $BESSEL_STATIC_SCRIPT $seed "${static_output_dir}/SEED_${seed}" $size_influence &
        run_script $BESSEL_STANDARD_SCRIPT $seed "${standard_output_dir}/SEED_${seed}" $size_influence &

        if (( $jobs -r | wc -l ) >= MAX_PARALLEL); then
            wait -n
        fi
    done

    wait
    echo "Finished training seed $seed"
done

wait
echo "All jobs COMPLETE"
