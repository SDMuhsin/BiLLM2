#!/bin/bash

echo "Submitting job for: ptb, facebook/opt-1.3B, braq"

sbatch \
    --nodes=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=1 \
    --gpus=1 \
    --mem=12000M \
    --time=0-02:00 \
    --chdir=/scratch/sdmuhsin/BiLLM2 \
    --output=test-braq-facebookopt-1.3B-%N-%j.out \
    --wrap="
        export TRANSFORMERS_CACHE=\"./downloads\"
        export HF_HOME=\"./downloads\"
        module load python/3.10
        module load arrow/16.1.0
        source ./env/bin/activate
        echo 'Environment loaded'
        which python3
        export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
        python3 run.py facebook/opt-1.3B ptb braq --blocksize 128 --salient_metric hessian --device=\"cuda\"
    "

