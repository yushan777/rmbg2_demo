#!/bin/bash

# create the venv
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# Check if the OS is macOS or Linux
ios_name="$(uname)"

# Common packages to install with specific versions
common_packages=("pillow==11.0.0" "kornia==0.7.4" "transformers==4.46.2" "timm==1.0.11")

if [ "$ios_name" == "Darwin" ]; then
    echo "The operating system is macOS."
    # Install required packages with specific versions
    pip install torch==2.5.1+cu12.4 torchvision==0.20.1+cu12.4
    pip install "${common_packages[@]}"
elif [ "$ios_name" == "Linux" ]; then
    # Check if CUDA is installed
    if command -v nvidia-smi &> /dev/null; then
        echo "The operating system is Linux with CUDA installed."
        # Install required packages with CUDA support and specific versions
        pip install torch==2.5.1+cu12.4 torchvision==0.20.1+cu12.4 torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
    else
        echo "The operating system is Linux but CUDA is not installed."
        # Install required packages without CUDA support and specific versions
        pip install torch==2.5.1 torchvision==0.20.1
    fi
    pip install "${common_packages[@]}"
else
    echo "Unknown operating system."
fi

# Download required files into the current directory
wget https://huggingface.co/briaai/RMBG-2.0/resolve/main/BiRefNet_config.py
wget https://huggingface.co/briaai/RMBG-2.0/resolve/main/birefnet.py
wget https://huggingface.co/briaai/RMBG-2.0/resolve/main/config.json
wget https://huggingface.co/briaai/RMBG-2.0/resolve/main/model.safetensors
