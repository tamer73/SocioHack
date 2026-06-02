#!/bin/bash

# Set environment variables
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
export PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1

# HuggingFace home directory for storing formatted datasets.
# Defaults to ~/.cache/huggingface; override via env var.
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
OUTPUT_DIR="${HF_HOME}/datasets"

# Function to prepare a single dataset
prepare_dataset() {
    local dataset_file="$1"
    local dataset_name="$2"

    echo "=========================================="
    echo "Preparing dataset: $dataset_name"
    echo "=========================================="

    # Check if dataset file exists
    if [ ! -f "$PROJECT_ROOT/data/synthetic/$dataset_file" ]; then
        echo "Error: Dataset file $PROJECT_ROOT/data/synthetic/$dataset_file not found!"
        return 1
    fi

    # Run the format_dataset command
    cd "$PROJECT_ROOT" && \
    python3 -m src.format_dataset "$PROJECT_ROOT/data/synthetic/$dataset_file" \
        --output-dir "$OUTPUT_DIR" \
        --dataset-name "$dataset_name"

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "=========================================="
        echo "Dataset $dataset_name prepared successfully!"
        echo "=========================================="
    else
        echo "=========================================="
        echo "Failed to prepare dataset $dataset_name (exit code: $exit_code)"
        echo "=========================================="
    fi

    return $exit_code
}

# Main execution
main() {
    cd "$PROJECT_ROOT"
    echo "Starting batch dataset preparation..."
    echo "=========================================="

    # Get list of JSON files in dataset directory (excluding tmp.txt and copy files)
    local dataset_files=($(ls "$PROJECT_ROOT/data/synthetic/"*.json 2>/dev/null | grep -v "copy.json" | xargs -I{} basename {}))

    if [ ${#dataset_files[@]} -eq 0 ]; then
        echo "No dataset files found in $PROJECT_ROOT/data/synthetic directory!"
        exit 1
    fi

    echo "Found ${#dataset_files[@]} datasets to prepare:"
    for file in "${dataset_files[@]}"; do
        echo "  - $file"
    done
    echo ""

    # Process each dataset
    local failed_datasets=()
    local successful_datasets=()

    for dataset_file in "${dataset_files[@]}"; do
        # Extract dataset name (remove .json extension). The HF dataset name
        # must match what scripts/train_synthetic.sh expects: <file>_synthetic.
        local dataset_name="${dataset_file%.json}_synthetic"

        echo "Processing dataset file: $dataset_file (name: $dataset_name)"

        # Prepare the dataset
        if prepare_dataset "$dataset_file" "$dataset_name"; then
            echo "Dataset $dataset_name prepared successfully"
            successful_datasets+=("$dataset_name")
        else
            echo "Dataset $dataset_name failed to prepare"
            failed_datasets+=("$dataset_name")
        fi

        echo ""
        echo "Waiting 5 seconds before next dataset..."
        sleep 5
        echo ""
    done

    # Summary
    echo "=========================================="
    echo "BATCH DATASET PREPARATION COMPLETED"
    echo "=========================================="
    echo "Total datasets processed: ${#dataset_files[@]}"
    echo "Successful: ${#successful_datasets[@]}"
    echo "Failed: ${#failed_datasets[@]}"

    if [ ${#successful_datasets[@]} -gt 0 ]; then
        echo "Successfully prepared datasets: ${successful_datasets[*]}"
    fi

    if [ ${#failed_datasets[@]} -gt 0 ]; then
        echo "Failed datasets: ${failed_datasets[*]}"
        exit 1
    else
        echo "All datasets prepared successfully!"
        exit 0
    fi
}

# Run main function
main "$@"
