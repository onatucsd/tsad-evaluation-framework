#!/bin/bash

# Loop through all files in the current directory
for file in ./scripts/WADI/*; do
    # Check if the item is a file (not a directory)
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        bash "$file"
        # Add your processing logic here
    fi
done

