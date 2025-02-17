#!/usr/bin/env python3
import json

def main():
    # Parameters (change these if needed)
    groupsize_param = 128
    salient_metric = "hessian"

    # Allowed models, datasets, and techniques
    allowed_models = [
        "facebook/opt-1.3b", "facebook/opt-2.7b", "facebook/opt-6.7b",
        "facebook/opt-13b", "facebook/opt-30b", "facebook/opt-66b"
    ]
    allowed_datasets = ["wikitext2", "ptb"]
    allowed_techniques = ["braq", "crb"]

    # Load the JSON file
    try:
        with open("./output/GLOBAL_PPL.json", "r") as f:
            data = json.load(f)
    except Exception as e:
        print("Error reading JSON file:", e)
        return

    # Dictionary to hold grouped results
    # Key: (model, dataset); Value: dict mapping technique -> metric_value
    results = {}

    for filepath, metric_value in data.items():
        # Remove the known prefix "./output/"
        prefix = "./output/"
        if filepath.startswith(prefix):
            stripped = filepath[len(prefix):]
        else:
            stripped = filepath

        # Instead of chopping off directory parts, we use the entire string.
        # Remove the trailing ".json"
        if stripped.endswith(".json"):
            stripped = stripped[:-5]

        # Now split the remaining string by underscores.
        # Expected format: "facebook/opt-1.3b_ptb_braq_128_hessian"
        parts = stripped.split("_")
        if len(parts) != 5:
            # Skip if the file name is not formatted as expected.
            continue

        model_str = parts[0]            # e.g., "facebook/opt-1.3b"
        dataset_str = parts[1]          # e.g., "ptb" or "wikitext2"
        technique_str = parts[2].lower()  # e.g., "braq" or "crb"
        groupsize_str = parts[3]        # e.g., "128"
        metric_str = parts[4].lower()   # e.g., "hessian"

        # Validate groupsize and salient metric
        try:
            groupsize_int = int(groupsize_str)
        except ValueError:
            continue

        if groupsize_int != groupsize_param or metric_str != salient_metric:
            continue

        # Normalize for comparison
        model_str = model_str.lower()
        dataset_str = dataset_str.lower()

        # Skip if this combination is not in our allowed lists.
        if (model_str not in allowed_models or
            dataset_str not in allowed_datasets or
            technique_str not in allowed_techniques):
            continue

        # Group by (model, dataset)
        key_group = (model_str, dataset_str)
        if key_group not in results:
            results[key_group] = {}
        results[key_group][technique_str] = metric_value

    # Print the results in a clean table format.
    header = "{:<20} {:<10} {:<15} {:<15}".format("Model", "Dataset", "braq", "crb")
    print(header)
    print("-" * len(header))
    for model in allowed_models:
        for dataset in allowed_datasets:
            key_group = (model, dataset)
            if key_group in results:
                braq_val = results[key_group].get("braq", "N/A")
                crb_val = results[key_group].get("crb", "N/A")
                print("{:<20} {:<10} {:<15} {:<15}".format(
                    model, dataset, braq_val, crb_val))
                
if __name__ == "__main__":
    main()

