#!/bin/bash
# RunPod serverless start command (base image: runpod/pytorch). Everything heavy lives on the network volume.
set -e
V=/runpod-volume
export HF_HOME=$V/hf PYTHONPATH=$V/pip:$PYTHONPATH PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p $V/hf $V/pip
REPO=${WORKER_REPO:-https://raw.githubusercontent.com/daniel11bb/recut-vision-worker/main}
if [ ! -f $V/pip/.ok ] || [ "$(curl -fsSL $REPO/requirements.txt | md5sum)" != "$(cat $V/pip/.req 2>/dev/null)" ]; then
  curl -fsSL $REPO/requirements.txt -o /tmp/req.txt
  pip install -q --target $V/pip --upgrade -r /tmp/req.txt
  md5sum < /tmp/req.txt > $V/pip/.req; touch $V/pip/.ok
fi
curl -fsSL $REPO/handler.py -o /handler.py
exec python -u /handler.py
