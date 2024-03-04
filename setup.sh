#!/bin/bash

# Get the current directory of the script
current_directory=$(dirname "$(readlink -f "$0")")

# Define the folder path
folder_path="src"

# Define the old and new patterns
old_pattern="/home/giacomo/thesis_ws/src/"
new_pattern=$current_directory"/src/"

# Use find and sed to search and replace in all files inside the folder
find "$folder_path" -type f -exec sed -i "s|$old_pattern|$new_pattern|g" {} +