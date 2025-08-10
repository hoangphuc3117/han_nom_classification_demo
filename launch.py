#!/usr/bin/env python3
"""
🚀 One-click launcher for Hán Nôm Classification Demo
"""
import subprocess
import sys
import webbrowser
import time
import socket

def check_port(port):
    """Check if port is available."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def main():
    print("🏛️ Hán Nôm Classification Demo")
    print("🚀 One-click Launcher")
    print("=" * 40)
    
    # Check if port is available
    port = 8501
    if not check_port(port):
        print(f"⚠️  Port {port} is already in use!")
        port = 8502
        print(f"🔄 Switching to port {port}")
    
    try:
        print("🚀 Starting demo...")
        
        # Launch Streamlit in background
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.port", str(port),
            "--server.address", "localhost",
            "--server.headless", "false"
        ])
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Open browser
        url = f"http://localhost:{port}"
        print(f"🌐 Opening browser at: {url}")
        webbrowser.open(url)
        
        print("✅ Demo is running!")
        print("⏹️  Press Ctrl+C to stop")
        print("-" * 40)
        
        # Wait for process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
