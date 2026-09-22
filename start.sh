#!/bin/bash

# Activate the virtual environment
if [ ! -d "$HOME/project/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $HOME/project/venv
fi
source $HOME/project/venv/bin/activate
# Install dependencies
pip install -r $HOME/project/requirements.txt
# Run the main script
$HOME/project/venv/bin/python $HOME/project/gimbal.py
