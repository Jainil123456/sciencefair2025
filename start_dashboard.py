#!/usr/bin/env python3
"""
Simple launcher that starts the web server and opens the browser.
"""

import webbrowser
import time
import threading
import sys
import os
import subprocess

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to activate virtual environment if it exists
venv_path = os.path.join(os.path.dirname(__file__), 'venv')
if os.path.exists(venv_path):
    # Check if we're already in venv
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not activated. Attempting to use venv...")
        print("💡 Tip: Run 'source venv/bin/activate' first for best results")

from agrigraph_ai.web_app import run_web_app

def open_browser(port=5000, max_attempts=10):
    """Open browser after a short delay to let server start."""
    # Wait a bit longer for server to fully start
    time.sleep(4)
    url = f'http://127.0.0.1:{port}'
    
    # Try to verify server is running before opening browser
    import urllib.request
    for attempt in range(max_attempts):
        try:
            response = urllib.request.urlopen(url, timeout=2)
            if response.getcode() == 200:
                print(f"\n🌐 Server is ready! Opening browser to {url}...")
                try:
                    webbrowser.open(url)
                    return
                except Exception as e:
                    print(f"⚠️  Could not open browser automatically: {e}")
                    print(f"   Please manually open: {url}")
                    return
        except Exception:
            time.sleep(1)
    
    print(f"\n⚠️  Server may not be ready yet. Please manually open: {url}")

if __name__ == '__main__':
    # Check for --kill-ports flag
    kill_ports = '--kill-ports' in sys.argv
    if kill_ports:
        sys.argv.remove('--kill-ports')
        print("🔧 Killing processes on ports 5000-5009...")
        try:
            subprocess.run([sys.executable, 'kill_ports.py'], check=False)
            print("Waiting 2 seconds for ports to be released...\n")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️  Could not kill ports: {e}\n")
    
    # Get port from command line or use default
    requested_port = 5000
    if len(sys.argv) > 1:
        try:
            requested_port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default 5000.")
    
    print("🔧 Initializing AgriGraph AI Dashboard...")
    print("📦 Checking dependencies...")
    
    # Check critical imports
    try:
        import torch
        import flask
        import plotly
        print("✅ All dependencies found")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\nPlease install dependencies:")
        print("  source venv/bin/activate")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    
    # Start browser in a separate thread (will use actual port from server)
    # We'll update this after server starts
    browser_port = [requested_port]  # Use list to allow modification
    
    def start_browser():
        open_browser(browser_port[0])
    
    browser_thread = threading.Thread(target=start_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the web server (this will block)
    try:
        # The run_web_app function will find an available port
        # and print it, but we need to capture it for the browser
        # For now, browser will try the requested port
        run_web_app(host='127.0.0.1', port=requested_port, debug=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        print("\n🔧 Troubleshooting:")
        print("1. Make sure virtual environment is activated: source venv/bin/activate")
        print("2. Check if dependencies are installed: pip install -r requirements.txt")
        print("3. Try a different port: python3 start_dashboard.py 5001")
        print("4. Check if port is in use: lsof -i :5000")
        sys.exit(1)

