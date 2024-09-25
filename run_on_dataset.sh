#!/bin/bash
# Run the executable for each .mtx file stored in the dataset subfolder

EXEC="./build/matvec_mul"
DATASET_FOLDER="dataset"
PLOT_SCRIPT="plot_stats.py"

# Check if the executable exists
if [[ ! -f "$EXEC" ]]; then
    echo "Executable not found! Please compile the program using make."
    exit 1
fi

# Check if the dataset folder exists
if [[ ! -d "$DATASET_FOLDER" ]]; then
    echo "Dataset folder '$DATASET_FOLDER' not found!"
    exit 1
fi

# Find all .mtx files in the dataset folder and execute the program on each one
for file in "$DATASET_FOLDER"/*.mtx; do
    if [[ -f "$file" ]]; then
        echo "Executing: $EXEC $file"
        $EXEC "$file"
    else
        echo "No .mtx files found in '$DATASET_FOLDER'."
    fi
done

# Run the plot script if it exists
if [[ -f "$PLOT_SCRIPT" ]]; then
    echo "Executing plot_stats.py to generate plots."
    python3 "$PLOT_SCRIPT"
else
    echo "Plot script '$PLOT_SCRIPT' not found!"
fi
