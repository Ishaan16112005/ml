#!/bin/bash

# run.sh - Execute the Machine Learning project end-to-end

set -e

echo "========================================="
echo "Machine Learning Project - End-to-End Execution"
echo "========================================="

# Activate virtual environment if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo "Virtual environment activated."
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed."
    exit 1
fi

echo ""
echo "[1/4] Installing dependencies from requirements.txt..."
python3 -m pip install -q -r requirements.txt

echo ""
echo "[2/4] Checking if survey data exists..."
if [ ! -f "src/london_weather.csv" ]; then
    echo "Error: london_weather.csv not found in current directory."
    echo "Please ensure the data file is present."
    exit 1
fi

echo "Data file found: london_weather.csv"

echo ""
echo "[3/4] Executing the project notebook..."
cd notebooks
jupyter nbconvert --to notebook --execute project.ipynb --output project_output.ipynb
cd ..

echo ""
echo "========================================="
echo "✓ Project execution completed successfully!"
echo "Results saved to:"
echo "  - src/london_weather_reasult.csv"
echo "  - notebooks/project_output.ipynb (executed notebook)"
echo "========================================="