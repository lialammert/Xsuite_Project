#!/bin/sh
# Check if a script path was provided as an argument
if [ "$1" = "" ]; then
    echo "No script provided. Exiting."
    exit 1
fi

# Activate the Python virtual environment
#source ~/miniforge3/bin/activate
source /afs/cern.ch/work/l/llammert/public/myenv/bin/activate

# Run the Python script passed as the first argument
python "$1"
