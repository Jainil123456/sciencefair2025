# Debugging Server Issues

## Common Issues and Solutions

### Issue 1: "ModuleNotFoundError: No module named 'torch'"

**Cause:** Virtual environment not activated or dependencies not installed.

**Solution:**
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python3 -c "import torch; print('OK')"
```

### Issue 2: Port 5000 Already in Use

**Cause:** macOS Control Center or another application using port 5000.

**Solution:**
- The server will automatically try ports 5001, 5002, etc.
- Check the console output for the actual port being used
- Or manually specify a port: `python3 start_dashboard.py 5001`

### Issue 3: Server Starts But Browser Doesn't Open

**Cause:** Browser opening may fail silently.

**Solution:**
- Check the console for the URL (e.g., `http://127.0.0.1:5000`)
- Manually open that URL in your browser
- The server is running even if browser doesn't open automatically

### Issue 4: "Connection Refused" in Browser

**Cause:** Server didn't start properly or crashed.

**Solution:**
1. Check terminal for error messages
2. Verify virtual environment is activated
3. Check if dependencies are installed
4. Try starting with explicit port: `python3 start_dashboard.py 5001`

### Issue 5: Template Not Found

**Cause:** Template directory not found.

**Solution:**
- Verify `templates/dashboard.html` exists
- Check file permissions
- The web_app.py should create the directory automatically

## Step-by-Step Debugging

### Step 1: Verify Environment
```bash
# Check Python version (should be 3.8+)
python3 --version

# Check if venv is activated
which python3  # Should show venv path if activated

# Check dependencies
python3 -c "import torch, flask, plotly; print('All OK')"
```

### Step 2: Test Server Import
```bash
source venv/bin/activate
python3 -c "from agrigraph_ai.web_app import app; print('Import successful')"
```

### Step 3: Check Port Availability
```bash
# Check if port 5000 is in use
lsof -i :5000

# Or try a different port
python3 start_dashboard.py 5001
```

### Step 4: Start Server with Verbose Output
```bash
source venv/bin/activate
python3 start_dashboard.py
```

Look for:
- ✅ "Starting server on http://..."
- ❌ Any error messages
- Port number being used

### Step 5: Test in Browser
1. Note the URL from console (e.g., `http://127.0.0.1:5001`)
2. Open that URL manually
3. Check browser console (F12) for JavaScript errors
4. Check Network tab for failed requests

## Manual Server Start

If automatic launcher fails, start manually:

```bash
# Terminal 1: Activate venv
source venv/bin/activate

# Terminal 1: Start server
python3 -m agrigraph_ai.web_app

# Terminal 2: Or use Flask directly
export FLASK_APP=agrigraph_ai.web_app
export FLASK_ENV=development
flask run --port=5001
```

## Checking Server Status

```bash
# Check if server is running
curl http://127.0.0.1:5000/

# Or check process
ps aux | grep "python.*web_app"

# Check port usage
lsof -i :5000
```

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Missing dependencies | `pip install -r requirements.txt` |
| `Address already in use` | Port taken | Use different port or kill process |
| `Template not found` | Missing template file | Check `templates/` directory |
| `Connection refused` | Server not running | Start server first |
| `500 Internal Server Error` | Code error | Check server logs |

## Getting Help

If issues persist:
1. Check the full error traceback in terminal
2. Verify all files are present (templates, config, etc.)
3. Try starting with minimal setup:
   ```bash
   source venv/bin/activate
   python3 -m flask --app agrigraph_ai.web_app run --port=5001
   ```

