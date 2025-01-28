#!/bin/bash

# Define the source and destination directories
backup_dir="/mnt/monitor/coastal-monitoring-system/data/.backup/HR6"
original_dir="/mnt/monitor/coastal-monitoring-system/data/HR6"

# Find all image_metadata files in the backup directory and copy them back to the original directory
find "$backup_dir" -type f -name "image_metadata.json" | while read -r backup_file; do
    # Get the relative path of the backup file
    rel_path="${backup_file#$backup_dir/}"

    # Define the corresponding original file path
    original_file="$original_dir/$rel_path"
    
    # Copy the backup file to the original location
    rsync -av "$backup_file" "$original_file"
done