#!/bin/bash
set -e

VENV_DIR=".venv"

echo "=== Setting up Virtual Environment ==="

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
else
    echo "Virtual environment $VENV_DIR already exists."
fi

source $VENV_DIR/bin/activate

echo "Upgrading pip and installing dependencies..."
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "Error: requirements.txt not found!"
    exit 1
fi

echo "=== Environment Setup Completed Successfully ==="