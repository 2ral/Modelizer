FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

LABEL maintainer="turalevro@gmail.com"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /root

RUN mkdir -p /root/.cache \
    && mkdir -p /root/datasets \
    && mkdir -p /root/src/modelizer \
    && mkdir -p /root/scripts

COPY LICENSE.txt /root/
COPY pyproject.toml /root/
COPY src/ /root/src/
COPY scripts/ /root/scripts/

RUN apt update && apt upgrade -y && apt install -y \
    python3 \
    python3-pip \
    nano \
    zip \
    unzip \
    && apt clean \
    && apt autoremove --purge -y \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.12 /usr/bin/python

RUN pip install --no-cache-dir --upgrade --break-system-packages setuptools \
    && pip install --break-system-packages --no-cache-dir -e . \
    && pip install --break-system-packages --no-cache-dir https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu130torch2.11-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl \
    && (rm -rf /root/.cache/* || true) \
    && (rm -rf /root/src/modelizer.egg-info || true)
