#!/bin/bash

# Define the input file and the target words
input_file="./data/hachidaishu/hachidai.db"  # Your input file
target_words=("$@")  # All target words passed as command-line arguments

# Step 1: Start by reading the entire file
result=$(cat "$input_file")

# Step 2: Loop over each target word and filter the results using grep
for target_word in "${target_words[@]}"; do
    # Filter the result by the current target word
    result=$(echo "$result" | grep -E "$target_word")

    # Extract unique poemIDs from the filtered results (First column XX:XXXXXX:XXXX -> XX:XXXXXX)
    poem_ids=$(echo "$result" | awk '{split($1, arr, ":"); print arr[1] ":" arr[2]}' | sort | uniq)

    # Retrieve all lines related to the matched poemIDs
    result=""
    for poem_id in $poem_ids; do
        # Append all lines related to the current poem_id from the input file
        matching_lines=$(grep -E "^$poem_id" "$input_file")
        result+="$matching_lines"$'\n'
    done
done

# Step 3: After processing all target words, output the final poemIDs and their corresponding lines
poem_ids=$(echo "$result" | awk '{split($1, arr, ":"); print arr[1] ":" arr[2]}' | sort | uniq)

# Step 4: Loop over the final poem_ids and print the corresponding lines
while IFS= read -r poem_id; do
    echo "$poem_id | $(grep -E "^$poem_id" "$input_file" | awk '$2 ~ /^[ABD]0/' | awk '{print $5}' | tr -d '\n')"
    # Output all lines matching the poem_id
    grep -E "^$poem_id" "$input_file"
done <<< "$poem_ids"
