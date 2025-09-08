# Use the NVIDIA CUDA base image with Ubuntu 22.04
FROM nvidia/cuda:12.2.2-base-ubuntu22.04 

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including Python, OpenCV, ZBar, and FFmpeg
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar-dev \
    apt-get clean

# Copy the current directory contents into the container at /app
COPY . .

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support for CUDA 12.4
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Expose the application port
EXPOSE 8000

# Command to run the application
CMD ["python3", "-m", "app.main"]
