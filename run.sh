#!/bin/bash
# Helper script to run the thermal camera with virtual environment

# Activate virtual environment
source venv/bin/activate

# Run the thermal camera application with provided arguments
python3 src/tc001v4.2.py "$@"
