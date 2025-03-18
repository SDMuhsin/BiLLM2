#!/bin/bash

echo "Beginning run.py sbatch script submissions."

# Iterate over datasets
for dataset in wikitext2; do
    # Iterate over models
    for model in "facebook/opt-13b" "facebook/opt-30b"; do # "facebook/opt-30b" "facebook/opt-66b"; do "huggyllama/llama-7b" "huggyllama/llama-13b" 
        # Iterate over techniques
        for bits in 2 3 4; do

            # Remove slash from model name for the output filename
            model_filename=${model//\//}

            echo "Submitting job with model: $model, dataset: $dataset, technique: $technique"
            sbatch \
                --nodes=1 \
                --ntasks-per-node=1 \
                --cpus-per-task=1 \
                --gpus=1 \
                --mem=128000M \
                --time=3-00:00 \
                --chdir=/scratch/sdmuhsin/BiLLM2 \
                --output=baseline-${bits}-${model_filename}-${dataset}-%N-%j.out \
                --wrap="
                    export TRANSFORMERS_CACHE=\"./downloads\"
                    export HF_HOME=\"./downloads\"
                    module load python/3.10
                    module load arrow/16.1.0
                    source ./env/bin/activate
                    echo 'Environment loaded'
                    which python3
                    export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"

		    python3 ./gptq/opt.py $model $dataset --wbits $bits --groupsize 128 --nearest
		    python3 ./gptq/opt.py $model $dataset --wbits $bits --groupsize 128

                "
        done
    done
done

echo "All jobs submitted."
#python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device=\"cuda\"
#python3 ./PB-LLM/gptq_pb/run.py $model $dataset $technique --low_frac 0.9 --high_bit 8 --blocksize 128 --salient_metric hessian
 
