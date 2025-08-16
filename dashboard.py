#!/usr/bin/env python3

import os
from dotenv import load_dotenv

load_dotenv()

def launch_dashboard():
    # Launch the TruLens dashboard to view evaluation results
    print("Launching TruLens Dashboard...")
    print("Dashboard will open in your default web browser")
    print("Press Ctrl+C to stop the dashboard")
    
    try:
        from evaluation import tru
        tru.run_dashboard()
    except Exception as e:
        print(f"Error launching dashboard: {e}")

if __name__ == "__main__":
    launch_dashboard() 