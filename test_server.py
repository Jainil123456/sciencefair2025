#!/usr/bin/env python3
"""
Diagnostic script to test server setup and identify issues.
"""

import sys
import os

def test_imports():
    """Test if all required modules can be imported."""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    modules = {
        'torch': 'PyTorch',
        'torch_geometric': 'PyTorch Geometric',
        'flask': 'Flask',
        'plotly': 'Plotly',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib'
    }
    
    failed = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - OK")
        except ImportError as e:
            print(f"❌ {name:20s} - FAILED: {e}")
            failed.append(name)
    
    return len(failed) == 0

def test_web_app():
    """Test if web app can be imported."""
    print("\n" + "=" * 60)
    print("Testing Web App Import...")
    print("=" * 60)
    
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from agrigraph_ai.web_app import app
        print("✅ Web app imported successfully")
        print(f"✅ Flask app created: {app}")
        return True
    except Exception as e:
        print(f"❌ Failed to import web app: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_templates():
    """Test if template files exist."""
    print("\n" + "=" * 60)
    print("Testing Template Files...")
    print("=" * 60)
    
    template_dir = os.path.join(os.path.dirname(__file__), 'templates')
    template_file = os.path.join(template_dir, 'dashboard.html')
    
    if os.path.exists(template_dir):
        print(f"✅ Template directory exists: {template_dir}")
    else:
        print(f"❌ Template directory missing: {template_dir}")
        return False
    
    if os.path.exists(template_file):
        print(f"✅ Template file exists: {template_file}")
        size = os.path.getsize(template_file)
        print(f"   File size: {size} bytes")
        return True
    else:
        print(f"❌ Template file missing: {template_file}")
        return False

def test_port(port=5000):
    """Test if a port is available."""
    print("\n" + "=" * 60)
    print(f"Testing Port {port}...")
    print("=" * 60)
    
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(('127.0.0.1', port))
        sock.close()
        print(f"✅ Port {port} is available")
        return True
    except OSError as e:
        print(f"❌ Port {port} is in use: {e}")
        print(f"   Trying alternative ports...")
        
        for p in range(5001, 5010):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.bind(('127.0.0.1', p))
                sock.close()
                print(f"✅ Port {p} is available (use this instead)")
                return True
            except OSError:
                continue
        
        print("❌ No available ports found between 5000-5009")
        return False

def test_config():
    """Test if config can be loaded."""
    print("\n" + "=" * 60)
    print("Testing Configuration...")
    print("=" * 60)
    
    try:
        from agrigraph_ai.config import Config
        print(f"✅ Config loaded")
        print(f"   Num nodes: {Config.NUM_NODES}")
        print(f"   Field size: {Config.FIELD_WIDTH}x{Config.FIELD_HEIGHT}")
        print(f"   Output dir: {Config.OUTPUT_DIR}")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 60)
    print("AgriGraph AI - Server Diagnostic Tool")
    print("=" * 60)
    print()
    
    results = {
        'imports': test_imports(),
        'web_app': test_web_app(),
        'templates': test_templates(),
        'port': test_port(),
        'config': test_config()
    }
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = all(results.values())
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test}")
    
    print()
    if all_passed:
        print("✅ All tests passed! Server should work.")
        print("\nTo start the server:")
        print("  source venv/bin/activate")
        print("  python3 start_dashboard.py")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  1. Activate virtual environment: source venv/bin/activate")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Check template files exist in templates/ directory")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())

