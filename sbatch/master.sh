#!/bin/bash

echo "Submitting dynamic sbatch jobs for run.py experiments"

# Define the datasets, models, and techniques
datasets=("wikitext2" "ptb")
models=("facebook/opt-1.3B" "facebook/opt-2.7B" "facebook/opt-6.7b" "facebook/opt-13b" "facebook/opt-30b" "facebook/opt-66b")
techniques=("braq" "rtn" "crb")

# Loop over each combination
for dataset in "${datasets[@]}"; do
  for model in "${models[@]}"; do
    for technique in "${techniques[@]}"; do
      # Remove any slash from the model name for the output file
      model_clean=$(echo "$model" | tr -d '/')
      
      echo "Submitting job: dataset=$dataset, model=$model, technique=$technique"
      
      sbatch \
        --nodes=1 \
        --ntasks-per-node=1 \
        --cpus-per-task=2 \
        --gpus=1 \
        --mem=32000M \
        --time=7-00:00 \
        --output="${technique}-${model_clean}-%N-%j.out" \
        --wrap="python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device='cuda'"
    done
  done
done

echo "All jobs submitted"

