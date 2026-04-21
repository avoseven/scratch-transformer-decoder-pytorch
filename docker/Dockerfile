# docker/Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*  # updateにより保存された，不要になったパッケージリストを削除して、Dockerイメージのサイズを小さくする

COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt