ARG tag=20.12
ARG dev=false
FROM nvcr.io/nvidia/tritonserver:${tag}-py3-sdk AS base
ADD docker/client/requirements.txt requirements.txt

FROM base AS false
RUN echo "git+https://github.com:alecgunny/tsinfer.git@consolidate" >> requirements.txt

FROM base AS true
VOLUME /opt/tsinfer
ADD tsinfer /opt/tsinfer
RUN pip install -e /opt/tsinfer

FROM $dev
RUN pip install -r requirements.txt
