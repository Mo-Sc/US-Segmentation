#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set environment variables with paths relative to the script's directory
export nnUNet_raw="$SCRIPT_DIR/data/nnUNet_raw"
export nnUNet_preprocessed="$SCRIPT_DIR/data/nnUNet_preprocessed"
export nnUNet_results="$SCRIPT_DIR/data/nnUNet_results"

# Create the directories if they don't exist
mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"
