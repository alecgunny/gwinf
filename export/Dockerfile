ARG tag=20.12
ARG dev=false
FROM nvcr.io/nvidia/pytorch:${tag}-py3 AS base
ADD requirements.txt requirements.txt

FROM base AS false
RUN echo "git+https://github.com/alecgunny/exportlib.git" >> requirements.txt

FROM base AS true
ADD exportlib /opt/exportlib
RUN pip install -e /opt/exportlib

FROM $dev
RUN pip install nvidia-pyindex && \
      pip install -r requirements.txt && \
      rm requirements.txt

ADD export.py export.py

RUN TARGET_DIR=/usr/local/cuda-11.1/targets/x86_64-linux && \
        LINK_DIR=/usr/local/cuda/compat/lib.real && \
        ln -s $TARGET_DIR/lib/libcudart.so.11.0 $LINK_DIR/libcudart.so.10.1 && \
        ln -s $TARGET_DIR/lib/libcublas.so.11 $LINK_DIR/libcublas.so.10 && \
        ln -s $TARGET_DIR/lib/libcusolver.so.11 $LINK_DIR/libcusolver.so.10 && \
        ln -s $TARGET_DIR/lib/libcusparse.so.11 $LINK_DIR/libcusparse.so.10 && \
        ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.8 $LINK_DIR/libcudnn.so.7
