FROM ubuntu:18.04

WORKDIR /home

ENV DEBIAN_FRONTEND noninteractive
# Core Linux Deps
RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --fix-missing --no-install-recommends \
    apt-utils         \
    build-essential   \
    curl              \
    wget              \
    git               \
    cmake             \
    libgl1-mesa-glx   \
    protobuf-compiler \
                                 && \
    apt-get clean                && \
    rm -rf /var/lib/apt/lists/*  && \
    apt-get clean && rm -rf /tmp/* /var/tmp/*
ENV DEBIAN_FRONTEND noninteractive

ARG PYTHON=python3
ARG PIP=pip3

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    ${PYTHON}     \
    ${PYTHON}-pip \
    ${PYTHON}-dev

RUN pip3 install --upgrade pip

RUN ${PIP} --no-cache-dir install --upgrade \
    pip           \
    setuptools    \
    opencv-python \
    numpy==1.18.0 \
    tensorflow==1.15.4

RUN alias python=python3

RUN git clone https://github.com/tensorflow/models.git    && \
    cd models/research                                    && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf1/setup.py .           && \
    python3 -m pip install --use-feature=2020-resolver .  && \
    python3 object_detection/builders/model_builder_tf1_test.py