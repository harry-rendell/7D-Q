#!/bin/bash

# Usage: bash sort_uid_mjd.sh <path_to_lc_files> <num_cores>

# Check if the location and num_cores arguments are provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Error: Please provide the location of the lc files as the first argument and the number of cores as the second argument."
  exit 1
fi

lc_location="$1"
num_cores="$2"

# Define a function to sort a file in place
sort_file_in_place() {
    # Print the filename to the console
    echo "Sorting file: $1"
    
    # Sort the file in place
    (head -n1 "$1" && tail -n+2 "$1" | sort -t',' -n -k1,2) > "$1.tmp"
    mv "$1.tmp" "$1"
}

# Export the function so that it can be used by subshells spawned by xargs
export -f sort_file_in_place

# Use a pipeline to generate the list of files and distribute them across cores
find "$lc_location" -maxdepth 1 -type f -name 'lc_*.csv' | \
    xargs -I{} -P "$num_cores" bash -c 'sort_file_in_place "$@"' _ {}
