#!/bin/bash
# Script to kill processes using ports 5000-5009 (for AgriGraph AI)

echo "🔍 Checking for processes on ports 5000-5009..."
echo ""

PORTS=(5000 5001 5002 5003 5004 5005 5006 5007 5008 5009)
KILLED=0

for port in "${PORTS[@]}"; do
    PID=$(lsof -ti :$port 2>/dev/null)
    if [ ! -z "$PID" ]; then
        PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null)
        echo "⚠️  Port $port is in use by PID $PID ($PROCESS_NAME)"
        
        # Check if it's a Python/Flask process (safe to kill)
        if [[ "$PROCESS_NAME" == *"python"* ]] || [[ "$PROCESS_NAME" == *"Python"* ]] || [[ "$PROCESS_NAME" == *"flask"* ]]; then
            echo "   Killing Python/Flask process on port $port..."
            kill $PID 2>/dev/null
            sleep 1
            # Force kill if still running
            if kill -0 $PID 2>/dev/null; then
                echo "   Force killing..."
                kill -9 $PID 2>/dev/null
            fi
            KILLED=$((KILLED + 1))
            echo "   ✅ Port $port is now free"
        else
            echo "   ⚠️  Skipping (not a Python process - might be system process)"
        fi
    else
        echo "✅ Port $port is free"
    fi
done

echo ""
if [ $KILLED -gt 0 ]; then
    echo "✅ Killed $KILLED process(es). Ports should be available now."
else
    echo "ℹ️  No Python processes found on these ports."
fi

echo ""
echo "🚀 You can now start the server:"
echo "   source venv/bin/activate"
echo "   python3 start_dashboard.py"

