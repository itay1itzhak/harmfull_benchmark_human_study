"""
Simple script to run the Streamlit benchmark explorer.

Usage:
    python run_streamlit.py
    
This will start the Streamlit server on the default port (8501).
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit application."""
    
    # Check if the benchmark data exists
    data_file = Path("benchmark/parsed_benchmark_data.json")
    if not data_file.exists():
        print("Benchmark data not found!")
        print("Please run the following command first:")
        print("   python parse_benchmark.py")
        print("\n This will generate the parsed_benchmark_data.json file needed for the explorer.")
        return 1
    
    print("Starting Benchmark Explorer...")
    print("Data file found: benchmark/parsed_benchmark_data.json")
    print("Opening browser at http://localhost:8501")
    print("\n Use Ctrl+C to stop the server")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "explore_benchmark.py",
            "--server.headless", "false",
            "--server.port", "8501"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f" Error running Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n Shutting down Benchmark Explorer...")
        return 0

if __name__ == "__main__":
    exit(main()) 