
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    torchaudio==2.1.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir "numpy<2.0"

RUN pip install --no-cache-dir \
    "tensorflow[and-cuda]==2.15.1" \
    gin-config
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY FrontEnd/leaf_pytorch ./leaf_pytorch
COPY leaf-audio/leaf_audio ./leaf_audio
COPY Model ./Model
# Only the model definitions -- the rest of the ConvNeXt repo is its ImageNet
# training harness, which Run8 does not use.
COPY ConvNeXt/models ./ConvNeXt/models
COPY Run5.py .
COPY Run6.py .
COPY Run7.py .
COPY Run8.py .
# Run5 = leaf_pytorch + VitGlobal; Run6 = TF leaf-audio + VitCnnGlobal;
# Run8 = leaf_pytorch + ViT or ConvNeXt (MODEL=convnext).

CMD ["python", "Run5.py"]
