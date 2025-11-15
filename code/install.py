import subprocess
import sys
import os

def install_requirements():
    """Install dependencies from requirements.txt."""
    req_file = "requirements.txt"
    
    if not os.path.exists(req_file):
        print(f"❌ {req_file} not found. Please create it first.")
        sys.exit(1)

    try:
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install from requirements.txt
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("✅ All dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()

