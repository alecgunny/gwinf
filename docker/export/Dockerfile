ARG tag=20.12
ARG dev=false
FROM nvcr.io/nvidia/pytorch:${tag}-py3 AS base
ADD docker/export/requirements.txt requirements.txt

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
ENTRYPOINT ["python", "export.py"]
CMD [""]
