#!/bin/bash

# RMBG2 does not support Python 3.13 (yet), so we will find the 
# highest installed Python version below 3.13 and use that
PYTHON=$(which -a python3.{0..12} | head -n 1)

if [ -z "$PYTHON" ]; then
    echo "No Python version below 3.13 found."
    exit 1
fi

# Create the virtual environment
$PYTHON -m venv venv
echo "Virtual environment created with $($PYTHON --version)"

# Activate the virtual environment
source venv/bin/activate
echo "Virtual environment activated"

# Install required dependencies for the script
pip install --upgrade pip
pip install huggingface-hub

# Check if the OS is macOS or Linux
os_name="$(uname)"
echo $os_name

# Common packages to install with specific versions
common_packages=("pillow==11.0.0" "kornia==0.7.4" "transformers==4.46.2" "timm==1.0.11")

if [ "$os_name" == "Darwin" ]; then
    echo "The operating system is macOS."
    # Install required packages with specific versions
    pip install torch torchvision
    pip install "${common_packages[@]}"
elif [ "$os_name" == "Linux" ]; then
    # Check if CUDA is installed
    if command -v nvidia-smi &> /dev/null; then
        echo "The operating system is Linux with CUDA installed."
        # Install required packages with CUDA support and specific versions
        pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
    else
        echo "The operating system is Linux but CUDA is not installed."
        # Install required packages without CUDA support and specific versions
        pip install torch==2.5.1 torchvision==0.20.1
    fi
    pip install "${common_packages[@]}"
else
    echo "Unknown operating system."
fi

# Download required files using Hugging Face Hub
echo "Downloading required files from Hugging Face repository..."
python -c "
from huggingface_hub import hf_hub_download

repo_id = 'briaai/RMBG-2.0'
files = [
    'BiRefNet_config.py',
    'birefnet.py',
    'config.json',
    'model.safetensors'
]

for file in files:
    print(f'Downloading {file}...')
    hf_hub_download(repo_id=repo_id, filename=file, local_dir='.', local_dir_use_symlinks=False)
"
echo "All files downloaded successfully."

echo -e "\e[35mActivate the venv before running with 'source venv/bin/activate'\e[0m"
