#!/usr/bin/env python3
"""
Kill processes using ports 5000-5009 (for AgriGraph AI server).
Only kills Python/Flask processes, not system processes.
"""

import subprocess
import sys
import time

def get_process_on_port(port):
    """Get PID of process using a port."""
    try:
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split('\n')[0]
    except Exception:
        pass
    return None

def get_process_name(pid):
    """Get name of process by PID."""
    try:
        # Try multiple methods to get process name
        for cmd in [
            ['ps', '-p', str(pid), '-o', 'comm='],
            ['ps', '-p', str(pid), '-o', 'command='],
            ['lsof', '-p', str(pid), '-a', '-c', 'python']
        ]:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                name = result.stdout.strip().split('\n')[0]
                if name:
                    return name
    except Exception:
        pass
    return None

def is_python_process(pid, process_name=None):
    """Check if process is a Python process."""
    if process_name:
        name_lower = process_name.lower()
        if 'python' in name_lower or 'flask' in name_lower:
            return True
    
    # Also check by checking the command line
    try:
        result = subprocess.run(
            ['ps', '-p', str(pid), '-o', 'command='],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            cmd = result.stdout.strip()
            if 'python' in cmd.lower() or 'flask' in cmd.lower() or 'web_app' in cmd.lower() or 'start_dashboard' in cmd.lower():
                return True
    except Exception:
        pass
    
    return False

def kill_process(pid, force=False):
    """Kill a process by PID."""
    try:
        signal = '-9' if force else '-15'
        subprocess.run(['kill', signal, str(pid)], timeout=2)
        return True
    except Exception:
        return False

def main():
    force = '--force' in sys.argv
    if force:
        print("⚠️  FORCE MODE: Will kill all processes on ports, not just Python")
        print("=" * 60)
    else:
        print("🔍 Checking for processes on ports 5000-5009...")
        print("=" * 60)
    
    ports = list(range(5000, 5010))
    killed = 0
    
    for port in ports:
        pid = get_process_on_port(port)
        if pid:
            process_name = get_process_name(pid)
            print(f"⚠️  Port {port} is in use by PID {pid} ({process_name})")
            
            # Check if it's a Python process
            is_python = is_python_process(pid, process_name)
            
            if is_python:
                print(f"   Killing Python/Flask process on port {port}...")
                if kill_process(pid):
                    time.sleep(0.5)
                    # Check if still running
                    if get_process_on_port(port):
                        print(f"   Force killing...")
                        kill_process(pid, force=True)
                        time.sleep(0.5)
                    
                    if not get_process_on_port(port):
                        print(f"   ✅ Port {port} is now free")
                        killed += 1
                    else:
                        print(f"   ❌ Failed to free port {port}")
                else:
                    print(f"   ❌ Failed to kill process")
            else:
                # For non-Python processes, ask user or skip
                if '--force' in sys.argv:
                    print(f"   ⚠️  Force killing non-Python process (PID {pid})...")
                    if kill_process(pid, force=True):
                        time.sleep(0.5)
                        if not get_process_on_port(port):
                            print(f"   ✅ Port {port} is now free")
                            killed += 1
                else:
                    print(f"   ⚠️  Skipping PID {pid} (not a Python process - use --force to kill anyway)")
        else:
            print(f"✅ Port {port} is free")
    
    print("=" * 60)
    if killed > 0:
        print(f"✅ Killed {killed} process(es). Ports should be available now.")
    else:
        print("ℹ️  No Python processes found on these ports.")
    
    print("\n🚀 You can now start the server:")
    print("   source venv/bin/activate")
    print("   python3 start_dashboard.py")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

