#!/bin/bash

export TRANSFORMERS_CACHE="./downloads"
export HF_HOME="./downloads"

# Define arrays for models and datasets
models=(
    #"facebook/opt-1.3b" "facebook/opt-2.7b" "facebook/opt-6.7b" 
    #"facebook/opt-13b" "facebook/opt-30b" "facebook/opt-66b"
    "huggyllama/llama-7b" "huggyllama/llama-13b" "huggyllama/llama-30b" "huggyllama/llama-65b"
)

datasets=("wikitext2" "ptb") #c4

# Iterate over each model and dataset
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running experiment with model: $model and dataset: $dataset"
        python3 run.py "$model" "$dataset" braq --blocksize 128 --salient_metric hessian --device="cpu" --just_download
    done
done

