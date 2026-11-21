# Pinned: the base image by digest and every package by version and SHA-256, so a rebuild gets
# the same environment. test_data.h5 is not copied in: the adapter mounts it read-only at /workspace/test_data.h5.
FROM python:3.11-slim@sha256:07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535

COPY docker_requirements.txt .
RUN pip install --no-cache-dir --require-hashes --only-binary=:all: -r docker_requirements.txt

WORKDIR /workspace

COPY process_data.py test_util.py /workspace/
ENV PYTHONPATH="/workspace" MPLBACKEND=Agg MPLCONFIGDIR=/tmp HOME=/tmp
USER 65534:65534
