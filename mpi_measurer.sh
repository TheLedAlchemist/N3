#!/bin/bash

#BSUB -n 2
#BSUB -W 00:05
#BSUB -R "rusage[mem=32]"
#BSUB -J mpiBesselTst
#BSUB -o out.%J
#BSUB -e err.%J

module load PrgEnv-intel

# Define the number of seeds and the number of parallel jobs
NUM_SEEDS=50
PLATFORM="cpu"

# Define the paths to the Python scripts
BESSEL_STATIC_SCRIPT="./scripts/bessel_static.py"
BESSEL_STANDARD_SCRIPT="./scripts/bessel_standard.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

export PYTHONPATH=/share/pendulums/msgill/GrowingNetworks/src:$PYTHONPATH
echo "Set the python path to: $PYTHONPATH"

# Function to run a Python script with a given seed and output directory
run_script() {
    local script=$1
    local seed=$2
    local output_dir=$3
    local lambda=$4
    mkdir -p $output_dir
    JAX_PLATFORM_NAME=$PLATFORM
    mpirun -n 1 python $script --seed $seed --out_path $output_dir --size_influence $lambda
}


# Run the scripts for each seed in parallel
static_output_dir="${OUTPUT_BASE_DIR}/BESSEL"
standard_output_dir="${OUTPUT_BASE_DIR}/BESSEL"

toPass=1

run_script $BESSEL_STANDARD_SCRIPT $toPass "${static_output_dir}/SEED_0/" 0.01
run_script $BESSEL_STANDARD_SCRIPT 5000 "${static_output_dir}/SEED_0/" 0.1

echo "Ran a single network measurement"
