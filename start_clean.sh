#!/bin/bash
# Start server on a clean port (5002) to avoid conflicts

cd "$(dirname "$0")"

echo "🧹 Starting server on clean port 5002..."
echo ""

# Activate venv if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start on port 5002 (which should be free)
python3 start_dashboard.py 5002

