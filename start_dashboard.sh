#!/bin/bash
# Simple shell script to start the dashboard

echo "Starting AgriGraph AI Dashboard..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Start the web server
python3 start_dashboard.py

