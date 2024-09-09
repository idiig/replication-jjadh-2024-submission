#!/bin/bash

# Define the input file and the target word
input_file="./data/hachidaishu/hachidai.db"  # Your input file
target_word="$1"  # Target word passed as a command-line argument

# Step 1 & 2: Grep the target word in the 8th column, extract unique poemID
poem_ids=$(grep -E "$target_word" "$input_file" | awk '{split($1, arr, ":"); print arr[1] ":" arr[2]}' | sort | uniq)

# Step 3: Loop over the unique poemIDs and print rows matching the poemID
while IFS= read -r poem_id; do
    # Use awk to print rows where the first column matches the poem_id
    echo $poem_id
    echo "Raw text:"
    awk -F',' '$1 ~ /'"$poem_id"'/ {print}' "$input_file" | awk '$2 ~ /^[ACE]0/' | awk '{print $5}' | tr -d '\n'
    echo ""
    echo "Annotation:"
    awk -F',' '$1 ~ /'"$poem_id"'/ {print}' "$input_file"
done <<< "$poem_ids"

