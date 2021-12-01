FROM tensorflow/tensorflow:latest-gpu

COPY requirements.txt /tmp/pip-tmp/

RUN pip3 install --upgrade pip \
    && pip3 --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp

RUN curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    libopencv-dev \
    python3-opencv

ENV TZ Asia/Tokyo
ENV PYTHONPATH "${PYTHONPATH}:/workspaces/craft-tensorflow"
EXPOSE 8888 6006
