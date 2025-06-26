"""
Simple script to run the Streamlit benchmark explorer.

Usage:
    python run_streamlit.py
    
This will start the Streamlit server on an available port.
"""

import subprocess
import sys
import socket
from pathlib import Path

def find_available_port(start_port=8501, max_port=8510):
    """
    Find an available port starting from start_port.
    
    Args:
        start_port (int): Port to start checking from
        max_port (int): Maximum port to check
        
    Returns:
        int: Available port number
        
    Raises:
        RuntimeError: If no available port is found
    """
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except socket.error:
            continue
    
    raise RuntimeError(f"No available ports found between {start_port} and {max_port}")

def kill_existing_streamlit():
    """
    Attempt to kill any existing Streamlit processes.
    """
    try:
        # Try to kill existing streamlit processes (Unix/Linux/Mac)
        subprocess.run(["pkill", "-f", "streamlit"], capture_output=True)
        print("🔄 Killed existing Streamlit processes")
    except (subprocess.CalledProcessError, FileNotFoundError):
        # pkill not available or no processes to kill
        pass

def main():
    """Run the Streamlit application."""
    
    # Check if the benchmark data exists
    data_file = Path("benchmark/parsed_benchmark_data.json")
    if not data_file.exists():
        print("❌ Benchmark data not found!")
        print("📋 Please run the following command first:")
        print("   python parse_benchmark.py")
        print("\n💡 This will generate the parsed_benchmark_data.json file needed for the explorer.")
        return 1
    
    # Try to find an available port
    try:
        print("Finding available port...")
        port = find_available_port()
        print(f"🚀 Starting Benchmark Explorer on port {port}...")
    except RuntimeError as e:
        print(f"❌ Port Error: {e}")
        print("\n🔧 To fix this:")
        print("1. Close other Streamlit apps or applications using ports 8501-8510")
        print("2. Or run: pkill -f streamlit")
        print("3. Then try again")
        return 1
    
    print(f"📊 Data file found: benchmark/parsed_benchmark_data.json")
    print(f"🌐 Opening browser at http://localhost:{port}")
    print("\n⚡ Use Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Run streamlit with the available port
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "explore_benchmark.py",
            #"--server.headless", "false",
            #"--server.port", str(port)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running Streamlit: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure Streamlit is installed: pip install streamlit")
        print("2. Check if another app is using the port")
        print("3. Try killing existing processes: pkill -f streamlit")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Shutting down Benchmark Explorer...")
        return 0

#if __name__ == "__main__":
#   exit(main()) 
main()