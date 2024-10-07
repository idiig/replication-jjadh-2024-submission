#!/bin/bash

# Define the list of translators
translators=("katagiri" "kubota" "matsuda" "okumura" "takeoka" "kaneko" "kojimaarai" "komachiya" "kyusojin" "ozawa")
poem_ids=(298)

# Function to process a single poemID and translator
process_poem() {
  local poemID=$1
  local translator=$2
  local log_file=$3

  echo "Running script with poemID=$poemID and translator=$translator" | tee -a "$log_file"

  # Get the poem content from file using grep, ensuring each result is on a new line
  poem=$(grep -E "01:00$poemID" data/hachidaishu/hachidai.db)

  # Get the translation content from file using grep, ensuring each result is on a new line
  translation=$(awk -v poemID="$poemID" '$3 ~ poemID {print}' data/translationExamples.txt | grep "$translator")

  # Check if poem and translation were found
  if [ -z "$poem" ]; then
    echo "No poem found for poemID=$poemID" | tee -a "$log_file"
    return
  fi

  if [ -z "$translation" ]; then
    echo "No translation found for poemID=$poemID and translator=$translator" | tee -a "$log_file"
    return
  fi

  poem_sentence=$(echo "$poem" | awk '$2 ~ /^[ABD]0/ {print $5}' | tr -d '\n')
  echo "Poem content: $poem_sentence" | tee -a "$log_file"

  translation_sentence=$(echo "$translation" | awk '$1 ~ /1/ {print}' | awk '$4 ~ /[01]/ {print $9}' | tr -d '\n')
  echo "Translation content: $translation_sentence" | tee -a "$log_file"

  # Run the Python script and pass the poem and translation as arguments
  python ./scripts/calc_match.py --poem "$poem" --translation "$translation" | tee -a "$log_file"
}

# Check if poemID or translator is provided
poemID=$1
translator=$2

# Create a log file
log_file="supplementary_materials/examples.txt"

# Helper function to format poemID with leading zeros
format_poemID() {
  printf "%04d" "$1"
}

# If poemID or translator is provided, process that specific poemID and translator
if [ -n "$poemID" ] || [ -n "$translator" ]; then
  # Default values if not provided
  poemID=$(format_poemID "${poemID:-1}")
  translator=${translator:-default_translator}
  process_poem "$poemID" "$translator" "$log_file"
else
  # If no specific poemID or translator is provided, loop through all poemID and translators
  for i in "${poem_ids[@]}"
  do
    poemID=$(format_poemID "$i")
    for translator in "${translators[@]}"
    do
      process_poem "$poemID" "$translator" "$log_file"
    done
  done
fi
