#!/bin/bash

echo "Beginning run.py sbatch script submissions."

# Iterate over datasets
for dataset in wikitext2 ptb; do
    # Iterate over models
    for model in "facebook/opt-1.3B" "facebook/opt-2.7B" "facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b" "facebook/opt-66b"; do
        # Iterate over techniques
        for technique in braq rtn crb; do

            # Remove slash from model name for the output filename
            model_filename=${model//\//}

            echo "Submitting job with model: $model, dataset: $dataset, technique: $technique"
            sbatch \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task=1 \
                --gpus=1 \
                --mem=32000M \
                --time=7-00:00 \
                --chdir=/scratch/sdmuhsin/BiLLM2 \
                --output=${technique}-${model_filename}-%N-%j.out \
                --wrap="
                    export TRANSFORMERS_CACHE=\"./downloads\"
                    export HF_HOME=\"./downloads\"
                    module load python/3.10
                    module load arrow/16.1.0
                    source ./env/bin/activate
                    echo 'Environment loaded'
                    which python3
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"
                    python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device=\"cuda\"
                "
        done
    done
done

echo "All jobs submitted."

