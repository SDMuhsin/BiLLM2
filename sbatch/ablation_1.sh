#!/bin/bash

echo "Beginning run.py sbatch script submissions."

# Iterate over datasets
for dataset in wikitext2; do
    # Iterate over models
    for model in "facebook/opt-1.3b"; do
        # Iterate over techniques
        for technique in crb; do

            ##############################
            # First inner loop: Vary lam
            # Fixed corr_damp=0.1
            ##############################
            for lam in 1e-6 5e-6 1e-5 5e-5 1e-4 5e-4 1e-3 5e-3; do

                # Remove slash from model name for the output filename
                model_filename=${model//\//}

                echo "Submitting job with model: $model, dataset: $dataset, technique: $technique, corr_damp: 0.1, lam: $lam"
                sbatch \
                    --nodes=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=1 \
                    --gpus=1 \
                    --mem=32000M \
                    --time=1-00:00 \
                    --chdir=/scratch/sdmuhsin/BiLLM2 \
                    --output=${technique}-${model_filename}-${dataset}-corr0.1-lam${lam}-%N-%j.out \
                    --wrap="
                        export TRANSFORMERS_CACHE=\"./downloads\"
                        export HF_HOME=\"./downloads\"
                        module load python/3.10
                        module load arrow/16.1.0
                        source ./env/bin/activate
                        echo 'Environment loaded'
                        which python3
                        export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"

                        python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device=\"cuda\" --corr_damp 0.1 --lam $lam --skip_ppl_save
                    "
            done

            ##############################
            # Second inner loop: Vary corr_damp
            # Fixed lam=1e-5
            ##############################
            for corr in 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do

                # Remove slash from model name for the output filename
                model_filename=${model//\//}

                echo "Submitting job with model: $model, dataset: $dataset, technique: $technique, corr_damp: $corr, lam: 1e-5"
                sbatch \
                    --nodes=1 \
                    --ntasks-per-node=1 \
                    --cpus-per-task=1 \
                    --gpus=1 \
                    --mem=32000M \
                    --time=1-00:00 \
                    --chdir=/scratch/sdmuhsin/BiLLM2 \
                    --output=${technique}-${model_filename}-${dataset}-corr${corr}-lam1e-5-%N-%j.out \
                    --wrap="
                        export TRANSFORMERS_CACHE=\"./downloads\"
                        export HF_HOME=\"./downloads\"
                        module load python/3.10
                        module load arrow/16.1.0
                        source ./env/bin/activate
                        echo 'Environment loaded'
                        which python3
                        export PYTHONPATH=\"\$PYTHONPATH:\$(pwd)\"

                        python3 run.py $model $dataset $technique --blocksize 128 --salient_metric hessian --device=\"cuda\" --corr_damp $corr --lam 1e-5 --skip_ppl_save
                    "
            done

        done
    done
done

echo "All jobs submitted."

