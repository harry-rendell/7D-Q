
#!/bin/bash

# TEST MERGED LIGHTCURVE SORTING ON A SINGLE FILE
# Usage: bash sort_uid_mjd_test.sh <path_to_file>

# Check if the file argument is provided
if [ -z "$1" ]; then
  echo "Error: Please provide the path to the file as the first argument."
  exit 1
fi

file_path="$1"

# Define a function to sort a file and output the sorted content
sort_file_and_output() {
    # Print the filename to the console
    echo "Sorting file: $1"
    
    # Sort the file and save the output to a temporary file
    (head -n1 "$1" && tail -n+2 "$1" | sort -t',' -n -k1,2) > "$1.tmp"
}
