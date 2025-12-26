# setup-torch.py (auxiliary script)
import sys
import subprocess

def install_pytorch():
    """Install PyTorch according to platform"""
    if sys.platform == "darwin":  # macOS
        # macOS with Apple Silicon (M1/M2)
        subprocess.run([
            "pip", "install", 
            "torch", "torchvision", 
            "--index-url", "https://download.pytorch.org/whl/cpu"
        ])
    elif sys.platform == "win32":  # Windows
        # Windows with CUDA (if you have GPU)
        subprocess.run([
            "pip", "install", 
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
    else:  # Linux
        # Linux with CUDA
        subprocess.run([
            "pip", "install", 
            "torch", "torchvision",
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])

if __name__ == "__main__":
    install_pytorch()