#!/bin/bash
# RunPod serverless start command (base image: runpod/pytorch). Models + pip cache live on the network volume.
# Never exits on an error: if the real handler can't load, a fallback handler answers every job with the startup log.
V=/runpod-volume
export HF_HOME=$V/hf PIP_CACHE_DIR=$V/pipcache PIP_DISABLE_PIP_VERSION_CHECK=1
mkdir -p $V/hf $V/pipcache
LOG=$V/start.log; : > $LOG
exec > >(tee -a $LOG) 2>&1
echo "== start $(date -u) image=$(cat /etc/os-release | grep PRETTY | cut -d= -f2) python=$(python -V 2>&1) torch=$(python -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>&1)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>&1 | head -1
REPO=${WORKER_REPO:-https://raw.githubusercontent.com/daniel11bb/recut-vision-worker/main}
curl -fsSL $REPO/requirements.txt -o /tmp/req.txt || echo "!! could not fetch requirements"
pip install -q -r /tmp/req.txt 2>&1 | tail -5
curl -fsSL $REPO/handler.py -o /handler.py || echo "!! could not fetch handler"
python -c "import runpod, transformers; print('runpod', runpod.__version__, 'transformers', transformers.__version__)" 2>&1
if python -c "import runpod" 2>/dev/null; then
  echo "== launching handler"
  python -u /handler.py
  echo "!! handler exited with $?"
fi
echo "== fallback handler (reports the log)"
pip install -q runpod 2>&1 | tail -2
python -u - <<'PY'
import runpod
def h(job):
    try: log = open("/runpod-volume/start.log").read()[-6000:]
    except Exception as e: log = str(e)
    return {"error": "worker failed to start", "log": log}
runpod.serverless.start({"handler": h})
PY
