#!/bin/bash

set -e

echo "Setting up Fireworks AI Text-to-SQL Take-Home..."
echo ""

# Create virtual environment and install dependencies.
# Prefer `uv` (faster), fall back to stdlib `venv` + `pip` if not installed.
if command -v uv &> /dev/null; then
    echo "Creating virtual environment with uv..."
    uv venv
    source .venv/bin/activate
    uv pip install -e .
else
    echo "uv not found — falling back to python3 -m venv + pip."
    if ! command -v python3 &> /dev/null; then
        echo "Error: neither uv nor python3 is on PATH."
        echo "Install uv (https://github.com/astral-sh/uv) or Python 3.11+."
        exit 1
    fi
    python3 -m venv .venv
    source .venv/bin/activate
    pip install --quiet --upgrade pip
    pip install --quiet -e .
fi

# Download the Chinook database
echo ""
echo "Downloading Chinook database..."
mkdir -p data

if [ -f "data/Chinook.db" ]; then
    echo "Removing existing data/Chinook.db..."
    rm data/Chinook.db
fi

curl -s https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql | sqlite3 data/Chinook.db

if [ -f "data/Chinook.db" ]; then
    echo "Successfully created data/Chinook.db"
else
    echo "Error: Failed to create database"
    exit 1
fi

echo ""
echo "Setup complete!"
echo ""
echo "To get started:"
echo "  1. Activate the virtual environment: source .venv/bin/activate"
echo "  2. Set your FIREWORKS_API_KEY environment variable"
echo "  3. Start building your solution"
echo ""