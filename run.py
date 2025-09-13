#!/usr/bin/env python3
"""
Financial PDF Extractor - Main Runner Script
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path
import webbrowser
import time

def check_requirements():
    """Check if all required packages are installed."""
    try:
        import streamlit
        import pandas
        import plotly
        import openai
        import pdf2image
        import PIL
        print("✅ All required packages are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has required variables."""
    env_path = Path(".env")
    if not env_path.exists():
        print("⚠️  .env file not found. Creating template...")
        create_env_template()
        return False
    
    # Check if OpenAI API key is set
    with open(env_path, 'r') as f:
        content = f.read()
        if "OPENAI_API_KEY=your_openai_api_key_here" in content:
            print("⚠️  Please update your OpenAI API key in the .env file")
            return False
    
    print("✅ Environment file configured.")
    return True

def create_env_template():
    """Create a template .env file."""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
APP_TITLE=Financial PDF Extractor & Analyzer
MAX_FILES=23
MAX_FILE_SIZE_MB=50

# Processing Configuration
PDF_DPI=200
MAX_TOKENS=2000
TEMPERATURE=0.1

# Debug Mode (set to True for development)
DEBUG=False
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("📝 Created .env template file. Please update with your API key.")

def run_streamlit_app():
    """Run the Streamlit application."""
    try:
        print("🚀 Starting Financial PDF Extractor...")
        print("📊 Opening browser in 3 seconds...")
        
        # Start Streamlit in a subprocess
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment then open browser
        time.sleep(3)
        webbrowser.open("http://localhost:8501")
        
        # Wait for the process
        process.wait()
        
    except KeyboardInterrupt:
        print("\n⏹️  Application stopped by user.")
        process.terminate()
    except Exception as e:
        print(f"❌ Error starting application: {e}")

def main():
    """Main function to run the application."""
    print("=" * 60)
    print("📊 Financial PDF Extractor & Consolidation System")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} detected.")
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    # Check environment file
    env_ready = check_env_file()
    
    if not env_ready:
        print("\n📋 Setup Instructions:")
        print("1. Update the .env file with your OpenAI API key")
        print("2. Get your API key from: https://platform.openai.com/api-keys")
        print("3. Replace 'your_openai_api_key_here' with your actual key")
        print("4. Run this script again")
        sys.exit(1)
    
    print("\n🎯 All checks passed! Starting application...")
    print("💡 The application will open in your default browser.")
    print("💡 Press Ctrl+C to stop the application.")
    print("-" * 60)
    
    # Run the application
    run_streamlit_app()

if __name__ == "__main__":
    main()