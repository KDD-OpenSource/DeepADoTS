FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

ARG USERNAME=docker

RUN apt-get update && \
  # We need add-apt-repository
  apt-get install -y --no-install-recommends software-properties-common && \
  # We need Python3.6
  add-apt-repository ppa:jonathonf/python-3.6 -y && \
  apt-get update && \
  apt-get install -y python3-pip python3.6 git python3.6-tk && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/* && \
  apt-get autoremove

ADD . /repo

RUN python3.6 -m pip --no-cache-dir install -r repo/requirements.txt
RUN python3.6 -m pip --no-cache-dir install tensorflow-gpu

RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc

ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=$CUDA_HOME
ENV PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64

WORKDIR /repo