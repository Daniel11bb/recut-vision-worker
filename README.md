# recut-vision-worker

RunPod serverless worker for Recut Studio: open-vocabulary object detection (Grounding DINO) and figure masks (SAM).

- `handler.py` — the RunPod handler (see its docstring for the request / response shape).
- `start.sh` — what the endpoint runs on a base PyTorch image: installs the requirements into the network volume once
  (`/runpod-volume/pip`), caches models there (`HF_HOME=/runpod-volume/hf`), pulls the latest `handler.py` from this repo, runs it.
- `requirements.txt` — Python packages (torch comes with the base image).

Deployed by the Recut Studio owner's setup (2026-10-29); the client is `core/vision.py` in recut-studio.
