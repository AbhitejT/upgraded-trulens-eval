#!/usr/bin/env python3

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def create_env_file():
    env_file = Path(".env")
    if not env_file.exists():
        print("\nCreating .env file...")
        with open(env_file, "w") as f:
            f.write("# OpenAI API Key\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
        print("✅ .env file created")
        print("⚠️  Please update .env with your actual OpenAI API key")
    else:
        print("✅ .env file already exists")

def main():
    print("TruLens RAG Evaluation System Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not Path("venv").exists():
        print("\nCreating virtual environment...")
        if not run_command("python -m venv venv", "Creating virtual environment"):
            sys.exit(1)
    else:
        print("✅ Virtual environment already exists")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:  # Unix/Linux/macOS
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{activate_cmd} && {pip_cmd} install -r requirements.txt", "Installing dependencies"):
        print("\n⚠️  Installation failed. You may need to install dependencies manually:")
        print("   source venv/bin/activate")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Create test documents directory
    test_docs = Path("test_documents")
    if not test_docs.exists():
        test_docs.mkdir()
        print("✅ test_documents directory created")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Update .env with your OpenAI API key")
    print("2. Place your documents in test_documents/ folder")
    print("3. Run: python main.py")
    print("4. View results: python dashboard.py")

if __name__ == "__main__":
    main() 