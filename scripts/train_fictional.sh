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

# Source dataset directory
DATASET_DIR="data/fictional"

# Base config that all per-scenario configs inherit from
BASE_CONFIG="fictional"

# Function to create per-scenario config
create_dataset_config() {
    local dataset_name="$1"
    local config_file="configs/${dataset_name}_fictional.yaml"

    echo "Creating config for scenario: ${dataset_name}"

    mkdir -p configs

    cp "configs/${BASE_CONFIG}.yaml" "$config_file"

    sed -i "s/suffix: \"SOCIOHACK-FICTIONAL\"/suffix: \"SOCIOHACK-FICTIONAL-${dataset_name}\"/" "$config_file"
    sed -i "s/name: \"sociohack_fictional\"/name: \"${dataset_name}_fictional\"/" "$config_file"

    echo "Created config: $config_file"
}

# Function to run training for a scenario
run_training() {
    local dataset_name="$1"
    local config_name="${dataset_name}_fictional"
    local config_file="configs/${config_name}.yaml"

    echo "=========================================="
    echo "Starting training for scenario: ${dataset_name}"
    echo "Config: ${config_file}"
    echo "=========================================="

    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found!"
        return 1
    fi

    ./scripts/train_single.sh "${config_name}"
    
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
    echo "Starting batch training for all datasets in ${DATASET_DIR}..."
    echo "=========================================="
    echo "Source directory: ${DATASET_DIR}"
    echo "Base config: configs/${BASE_CONFIG}.yaml"
    echo "=========================================="
    
    # Get list of JSON files in dataset directory (excluding copy files)
    local datasets=($(ls ${DATASET_DIR}/*.json 2>/dev/null | grep -v "copy.json" | sed "s|${DATASET_DIR}/||" | sed 's/\.json//'))
    
    if [ ${#datasets[@]} -eq 0 ]; then
        echo "No dataset files found in ${DATASET_DIR} directory!"
        exit 1
    fi
    
    echo "Found ${#datasets[@]} datasets: ${datasets[*]}"
    echo ""
    
    # Process each dataset
    local failed_datasets=()
    local skipped_datasets=()
    
    for dataset in "${datasets[@]}"; do
        # Skip scenarios where all artifacts exist AND rollouts CSV is complete (>= 60 rows).
        local task_name="${dataset}_fictional"
        if ls loopholes_${task_name}*.json >/dev/null 2>&1 && \
           ls rollouts_${task_name}*.csv >/dev/null 2>&1 && \
           ls llm_debug_${task_name}*.csv >/dev/null 2>&1; then
            local rollout_csv=$(ls rollouts_${task_name}*.csv 2>/dev/null | head -1)
            local row_count=$(python3 -c "import pandas as pd; print(len(pd.read_csv('$rollout_csv')))" 2>/dev/null)
            if [ -n "$row_count" ] && [ "$row_count" -ge 60 ]; then
                echo "Skipping already completed scenario: $dataset (rollouts=$row_count)"
                skipped_datasets+=("$dataset")
                continue
            else
                echo "Re-running incomplete scenario: $dataset (rollouts=${row_count:-?} < 60)"
            fi
        fi

        echo "Processing dataset: $dataset"
        
        # Create per-dataset config
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
