#!/bin/bash

# Stock Forecasting Application Startup Script

echo "Starting Stock Forecasting Application..."

# Check if MongoDB is running
if ! pgrep -x "mongod" > /dev/null; then
    echo "MongoDB is not running. Please start MongoDB first."
    echo "Run: mongod"
    exit 1
fi

# Install dependencies if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Starting Flask application..."
python backend/app.py
