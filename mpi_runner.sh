#!/bin/bash

#BSUB -n 5
#BSUB -W 00:20
#BSUB -J testing
#BSUB -o out.%J
#BSUB -e err.%J

module load PrgEnv-intel

# The lambda values [ 10^MIN, 10^MAX ] to scan through.
L_MIN=-2
L_MAX=2

# Network hyperparameters
LEARNING_RATE=$(awk 'BEGIN{printf "%.3f\n", 0.001}')
NUM_EPOCHS=10000
MAX_HIDDEN_LAYER_SIZE=10

# The number of times to sweep through the complete parameter space
NUM_PASSES=1
# The number of parallel jobs to spin up evenly across parameter space
PARALLEL=5
# The JAX platform to use
PLATFORM="cpu"

# Define paths to python scripts
BESSEL_STANDARD_SCRIPT="./scripts/bessel_standard.py"
BESSEL_STATIC_SCRIPT="./scripts/bessel_static.py"

# Define the base output directory
OUTPUT_BASE_DIR="./output"

export PYTHONPATH=$(pwd)/src:$PYTHONPATH
echo "Set python path to $PYTHONPATH"


vary_lambda_mpi() {
    local run=$1
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

    # An optional affine offset for the seeds to ensure orthogonal measurements
    seed_indexer=6000

    # Generate the command over the desired log spread of seeds
    for s in $(seq $start $step $end); do
        lambda=$(awk "BEGIN {print 10^($s)}")
        current_seed=$((seed_indexer + ((PARALLEL + 1) * (run - 1))))
        outd="${OUTPUT_BASE_DIR}/${outdir}/SEED_${current_seed}/"

        command+=" -n 1 python ${exe} --seed ${current_seed} --out_path \"${outd}/SIZE_INF_${lambda}/\" --size_influence ${lambda} "
        command+="--N_max ${MAX_HIDDEN_LAYER_SIZE} --epochs ${NUM_EPOCHS} --learning_rate ${LEARNING_RATE} :"
        seed_indexer=$((seed_indexer + 1))
    done

    # Remove trailing colon
    command=${command%:}
    echo "Scanning through ${start} to ${end} with step size 10^${step}"
    eval "${command}"
    echo "Seed ${seed} training completed!"
}

lambdaStep=$(awk "BEGIN {print $((L_MAX - L_MIN)) / $PARALLEL}")
echo -e "\nLambda step size: ${lambdaStep}\n\n"

for run in $(seq 1 1 $NUM_PASSES); do
    vary_lambda_mpi $run $BESSEL_STATIC_SCRIPT "STATIC_FINERUN_ORTHO" $L_MIN $L_MAX $lambdaStep
done

echo -e "\n\n\nAll training completed!"
