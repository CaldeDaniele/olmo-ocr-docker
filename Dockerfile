FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    poppler-utils \
    wget \
    git \
    build-essential \
    python3-pip \
    curl \
    software-properties-common \
    && add-apt-repository multiverse \
    && apt-get update \
    && apt-get install -y ttf-mscorefonts-installer \
    fonts-crosextra-caladea fonts-crosextra-carlito gsfonts lcdf-typetools

# Accept Microsoft font license terms
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh

# Set conda in PATH
ENV PATH="/opt/conda/bin:${PATH}"

# Create conda environment
RUN conda create -y -n olmocr python=3.11 \
    && echo "source activate olmocr" > ~/.bashrc

# Activate conda environment and install olmocr
SHELL ["/bin/bash", "-c"]
RUN source activate olmocr \
    && git clone https://github.com/allenai/olmocr.git \
    && cd olmocr \
    && pip install -e .

# Install sglang with flashinfer for GPU
RUN source activate olmocr \
    && pip install sgl-kernel==0.0.3.post1 --force-reinstall --no-deps \
    && pip install "sglang[all]==0.4.2" --find-links https://flashinfer.ai/whl/cu124/torch2.4/flashinfer/

# Install Gradio
RUN source activate olmocr \
    && pip install gradio

# Copy the Gradio app
COPY app.py /app/app.py

# Expose port for Gradio
EXPOSE 7860

# Set working directory
WORKDIR /app

# Startup command
CMD ["bash", "-c", "source activate olmocr && python app.py"]