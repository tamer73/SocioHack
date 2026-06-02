#!/bin/bash

# Set environment variables to reduce verbose output
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=ERROR
export NCCL_P2P_DISABLE=1
export VLLM_LOG_LEVEL=WARNING
export TRANSFORMERS_VERBOSITY=warning
export HF_HUB_VERBOSITY=error
export DEEPSPEED_LOG_LEVEL=ERROR
export PYTHONWARNINGS=ignore
export CUDA_LAUNCH_BLOCKING=0

# Function to create per-scenario config from the historical base
create_dataset_config() {
    local dataset_name="$1"
    local config_file="configs/${dataset_name}_historical.yaml"

    echo "Creating config for scenario: ${dataset_name}"

    mkdir -p configs

    # Copy the historical base config and substitute the scenario name
    cp configs/historical.yaml "$config_file"

    sed -i "s/suffix: \".*\"/suffix: \"SOCIOHACK-HISTORICAL-${dataset_name}\"/" "$config_file"
    sed -i "s/name: \"sociohack_historical\"/name: \"${dataset_name}\"/" "$config_file"

    echo "Created config: $config_file"
}

# Function to run training for a scenario
run_training() {
    local dataset_name="$1"
    local config_file="configs/${dataset_name}_historical.yaml"

    echo "=========================================="
    echo "Starting training for scenario: ${dataset_name}"
    echo "=========================================="

    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found!"
        return 1
    fi

    # Invoke the single-run launcher with the new config name
    ./scripts/train_single.sh "${dataset_name}_historical"
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=========================================="
        echo "Training completed successfully for: ${dataset_name}"
        echo "=========================================="
    else
        echo "=========================================="
        echo "Training failed for: ${dataset_name} (exit code: $exit_code)"
        echo "=========================================="
    fi
    
    return $exit_code
}

# Main execution
main() {
    echo "Starting batch training for all datasets..."
    echo "=========================================="
    
    # Get list of JSON files in data/historical (excluding copy files); extract bare name without extension
    local datasets=($(ls data/historical/*.json 2>/dev/null | grep -v "copy.json" | xargs -I{} basename {} .json))

    if [ ${#datasets[@]} -eq 0 ]; then
        echo "No scenario files found in data/historical directory!"
        exit 1
    fi
    
    echo "Found ${#datasets[@]} datasets: ${datasets[*]}"
    echo ""
    
    # Process each dataset
    local failed_datasets=()
    local skipped_datasets=()
    
    for dataset in "${datasets[@]}"; do
        # Skip scenarios where rollouts CSV is already complete (>= 60 rows).
        rollout_csv="rollouts_${dataset}.csv"
        if [ -f "$rollout_csv" ]; then
            row_count=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$rollout_csv')))" 2>/dev/null)
            if [ -n "$row_count" ] && [ "$row_count" -ge 60 ]; then
                echo "Skipping already completed scenario: $dataset (rollouts=$row_count)"
                skipped_datasets+=("$dataset")
                continue
            else
                echo "Re-running incomplete scenario: $dataset (rollouts=${row_count:-?} < 60)"
            fi
        fi

        echo "Processing dataset: $dataset"
        
        # Create config for this dataset
        create_dataset_config "$dataset"
        
        # Run training
        if run_training "$dataset"; then
            echo "Dataset $dataset completed successfully"
        else
            echo "Dataset $dataset failed"
            failed_datasets+=("$dataset")
        fi
        
        echo ""
        echo "Waiting 10 seconds before next dataset..."
        sleep 10
        echo ""
    done
    
    # Summary
    echo "=========================================="
    echo "BATCH TRAINING COMPLETED"
    echo "=========================================="
    if [ ${#skipped_datasets[@]} -gt 0 ]; then
        echo "Skipped ${#skipped_datasets[@]} already completed scenarios: ${skipped_datasets[*]}"
        echo "------------------------------------------"
    fi
    echo "Total datasets in list: ${#datasets[@]}"
    echo "Processed this run: $((${#datasets[@]} - ${#skipped_datasets[@]}))"
    echo "Successful: $((${#datasets[@]} - ${#skipped_datasets[@]} - ${#failed_datasets[@]}))"
    echo "Failed: ${#failed_datasets[@]}"
    
    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo "Failed datasets: ${failed_datasets[*]}"
        exit 1
    else
        echo "All datasets completed successfully!"
        exit 0
    fi
}

# Run main function
main "$@"
