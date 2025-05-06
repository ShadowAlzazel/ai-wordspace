#!/bin/bash

# Check if venv exists, if not, create it
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python -m venv venv
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
if [ -f "requirements.txt" ]; then
  echo "Installing dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "requirements.txt not found. Skipping dependency installation."
fi

echo "Environment setup complete."
