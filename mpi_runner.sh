#!/bin/bash

#BSUB -n 10
#BSUB -W 00:05
#BSUB -J smallMPISweep
#BSUB -o out.%J
#BSUB -e err.%J

module load PrgEnv-intel

NUM_SEEDS=50
PARALLEL=10
PLATFORM="cpu"

# Define paths to python scripts
BESSEL_STANDARD_SCRIPT="./scripts/bessel_standard.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
echo "Set python path to $PYTHONPATH"

# Single script is JAX_PLATFORM_NAME=$PLATFORM python ${script} --seed ${i} --out_path ${direc} --size_inflence ${lambda}
vary_lambda_mpi() {
    local seed=$1
    local exe=$2
    local outdir=$3

    local start=$4
    local end=$5
    local step=$6

    local command="mpirun"

    # Consider modifying output path here
    local outd="${OUTPUT_BASE_DIR}/${outdir}/SEED_${seed}/"
    # Otherwise, ignore this block

    JAX_PLATFORM_NAME="${PLATFORM}"

    # Generate the command over the desired log spread of seeds
    for s in $(seq $start $step $end); do
        lambda=$(awk "BEGIN {print 10^($s)}")
        command+=" -n 1 python ${exe} --seed ${seed} --out_path \"${outd}/SIZE_INF_${lambda}/\" --size_influence ${lambda} :"
    done

    # Remove trailing colon
    command=${command%:}
    echo "${command}"
    eval "${command}"
}

lambdaStep=$(awk "BEGIN {print 4 / $PARALLEL}")
echo -e "\nLambda step size: ${lambdaStep}\n\n"

vary_lambda_mpi 0 $BESSEL_STANDARD_SCRIPT "DUMMY" -2 2 $lambdaStep
