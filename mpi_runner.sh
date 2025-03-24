#!/bin/bash

#BSUB -n 5
#BSUB -W 00:20
#BSUB -J testing
#BSUB -o out.%J
#BSUB -e err.%J

module load PrgEnv-intel

# The number of nodes available to MPI
MPI_NODES=2

# The lambda values [ 10^MIN, 10^MAX ] to scan through.
L_MIN=-2
L_MAX=2

# Network hyperparameters
LEARNING_RATE=$(awk 'BEGIN{printf "%.3f\n", 0.001}')
NUM_EPOCHS=10000
MAX_HIDDEN_LAYER_SIZE=10

# Number of times to sweep through the complete lambda space
RUNS_PER_LAMBDA=10

# Logarithmic divisions of lambda space [ 10^L_MIN, 10^L_MAX ] per pass. SHOULD evenly divide MPI_NODES
PARALLEL=2
# Batch size
Loops_per_call=$(expr $MPI_NODES / $PARALLEL)
echo "We will iterate over ${Loops_per_call} row dimensions per function call"

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

    # An offset for the seeds to ensure orthogonal measurements
    seed_indexer=$(( (run - 1) * Loops_per_call * PARALLEL ))

    for _ in $(seq 1 1 $Loops_per_call); do

        # Generate the command over the desired log spread of seeds
        for s in $(seq $start $step $end); do
            lambda=$(awk "BEGIN {print 10^($s)}")
            current_seed=$((seed_indexer + ((PARALLEL) * (run - 1)) ))
            outd="${OUTPUT_BASE_DIR}/${outdir}/SEED_${current_seed}/"

            command+=" -n 1 python ${exe} --seed ${current_seed} --out_path \"${outd}\" --size_influence ${lambda} "
            command+="--N_max ${MAX_HIDDEN_LAYER_SIZE} --epochs ${NUM_EPOCHS} --learning_rate ${LEARNING_RATE} :"
            seed_indexer=$((seed_indexer + 1))
        done

    done

    # Remove trailing colon
    command=${command%:}
    echo "Scanning through ${start} to ${end} with step size 10^${step}"
    echo "${command}"
    echo "${Loops_per_call} batches of ${parallel} networks completed!"
}

# PARALLEL - 1 is to create proper fenceposting
lambdaStep=$(awk "BEGIN {print $((L_MAX - L_MIN)) / $((PARALLEL - 1))}")

echo -e "Lambda step size: ${lambdaStep}\n\n"

# I'm fairly certain I could encorporate batches out here...
# I know that if I repeat batch * (SPACE / step) times in the function,
# I should account for that somehow in this outer loop
total_runs=$(( RUNS_PER_LAMBDA ))
echo "Total runs is ${total_runs}"

for run in $(seq 1 $Loops_per_call $total_runs); do
    vary_lambda_mpi $run $BESSEL_STANDARD_SCRIPT "STATIC_FINERUN_ORTHO" $L_MIN $L_MAX $lambdaStep
done

echo -e "\n\n\nAll training completed!"
